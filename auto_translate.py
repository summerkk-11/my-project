#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DeepSeek API 将日文 SRT 翻译成中文。

安装依赖：
    pip install openai

用法：
    python auto_translate.py "日文字幕.srt"
    python auto_translate.py "日文字幕.srt" --output "chinese_subtitle.srt"

说明：
1. 脚本默认每 25 条字幕调用一次 API，绝不会把整份长字幕一次性发送。
2. 程序会验证模型返回的字幕数量、序号和时间轴，发现格式错误会自动重试。
3. API Key 不要提交到 GitHub，也不要把写着真实 Key 的脚本发给别人。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


# ============================================================================
# API 配置：填入自己的 DeepSeek API Key 后直接运行
# ============================================================================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "在这里填写你的 DeepSeek API Key")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 每块字幕条数。要求是 20～30 条，25 条在上下文长度和翻译连贯性之间较平衡。
DEFAULT_CHUNK_SIZE = 25

# 初次调用失败后，最多再重试 3 次。
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 120


@dataclass
class SubtitleCue:
    """一条 SRT 字幕。"""

    index: str
    timecode: str
    text: str

    def to_srt(self) -> str:
        """按标准 SRT 格式序列化；正文内部可以有换行。"""
        return f"{self.index}\n{self.timecode}\n{self.text}".rstrip()


# 秒的小数部分固定为三位；小时允许超过两位，以兼容超长视频。
TIMECODE_PATTERN = re.compile(
    r"^\s*"
    r"\d{2,}:\d{2}:\d{2},\d{3}"
    r"\s*-->\s*"
    r"\d{2,}:\d{2}:\d{2},\d{3}"
    r"(?:\s+.*)?$"
)


def parse_srt(content: str) -> list[SubtitleCue]:
    """
    严格但兼容常见 SRT 变体地解析字幕。

    每条 cue 的结构：
        序号
        时间轴
        正文（可以有多行）
    """
    # 统一换行，去掉 UTF-8 BOM，避免 Windows/Unix 混用导致空块。
    content = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n+", content.strip())
    cues: list[SubtitleCue] = []

    for block_number, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if not any(line.strip() for line in lines):
            continue

        if len(lines) < 2:
            raise ValueError(f"第 {block_number} 个字幕块不完整")

        # 标准 SRT 的序号和北京时间轴通常在前两行。
        # 为兼容少见的空行，前 3 行中查找时间轴。
        time_line_index = next(
            (
                i
                for i, line in enumerate(lines[:3])
                if TIMECODE_PATTERN.match(line)
            ),
            None,
        )
        if time_line_index is None:
            raise ValueError(
                f"第 {block_number} 个字幕块找不到有效时间轴："
                f"{lines[:2]}"
            )
        if time_line_index == 0:
            raise ValueError(
                f"第 {block_number} 个字幕块缺少数字序号"
            )

        index = lines[time_line_index - 1].strip()
        if not index.isdigit():
            raise ValueError(
                f"第 {block_number} 个字幕块的序号不是数字：{index!r}"
            )

        text = "\n".join(lines[time_line_index + 1 :]).strip()
        if not text:
            raise ValueError(
                f"第 {block_number} 个字幕块的正文为空"
            )

        cues.append(
            SubtitleCue(
                index=index,
                timecode=lines[time_line_index].strip(),
                text=text,
            )
        )

    if not cues:
        raise ValueError("SRT 中没有解析到任何字幕。")
    return cues


def split_cues(
    cues: list[SubtitleCue], chunk_size: int
) -> list[list[SubtitleCue]]:
    """按照固定 cue 数量切块，不按字符数硬切，避免破坏一条字幕。"""
    if chunk_size < 1:
        raise ValueError("chunk_size 必须大于 0")
    return [
        cues[offset : offset + chunk_size]
        for offset in range(0, len(cues), chunk_size)
    ]


def serialize_cues(cues: list[SubtitleCue]) -> str:
    """将 cue 列表拼成等待翻译的 SRT 文本。"""
    return "\n\n".join(cue.to_srt() for cue in cues) + "\n"


def strip_optional_code_fence(text: str) -> str:
    """
    只清除模型偶尔添加的外层 ```srt/``` 代码围栏。

    不修改序号、时间轴或正文内容。真正格式仍然会在后面严格校验。
    """
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def validate_translation(
    original: list[SubtitleCue],
    translated: list[SubtitleCue],
) -> None:
    """
    强制验证模型输出：
    - cue 数量必须一致；
    - 每条序号必须一致；
    - 每条时间轴必须逐字符一致。

    这里故意不做“自动修复序号/时间轴”，因为自动猜测很容易造成整部字幕错位。
    格式不合格时由上层重新请求模型。
    """
    if len(translated) != len(original):
        raise ValueError(
            f"字幕数量不一致：期望 {len(original)} 条，实际 {len(translated)} 条"
        )

    for position, (source, result) in enumerate(
        zip(original, translated), start=1
    ):
        if source.index != result.index:
            raise ValueError(
                f"第 {position} 条序号被改动："
                f"{source.index!r} -> {result.index!r}"
            )
        if source.timecode != result.timecode:
            raise ValueError(
                f"第 {position} 条时间轴被改动：\n"
                f"原文: {source.timecode}\n"
                f"返回: {result.timecode}"
            )
        if not result.text.strip():
            raise ValueError(f"第 {position} 条译文为空")


def build_prompt(chunk: list[SubtitleCue]) -> tuple[str, str]:
    """构造强制格式约束，返回 system prompt 和 user prompt。"""
    source_srt = serialize_cues(chunk)
    system_prompt = (
        "你是专业的日语字幕翻译器。"
        "你只做忠实翻译，不解释、不总结、不续写、不删除信息。"
        "严禁把字幕合并、拆分、重排或添加原文不存在的内容。"
    )
    user_prompt = f"""请把下面 SRT 字幕的正文从日语翻译为简体中文。

必须遵守以下规则，任何一条都不能违反：
1. 禁止输出任何解释、说明、前后缀或 Markdown 代码围栏。
2. 必须原封不动保留每一条的数字序号和完整时间轴。
3. 只翻译时间轴下方的正文；序号、箭头、冒号、逗号和毫秒都不能改。
4. 不得合并、拆分、增加或删除任何一条字幕。
5. 专有名词、人物称呼和语气必须保持一致。
6. 原文已经标记为 [无法识别] 的内容，译文也原样输出 [无法识别]。
7. 不要根据上下文臆造原文没有出现的内容。
8. 最终输出必须可以直接保存为合法的 SRT 文件。

待翻译 SRT：
{source_srt}"""
    return system_prompt, user_prompt


def translate_chunk(
    client: OpenAI,
    chunk: list[SubtitleCue],
    chunk_number: int,
    chunk_total: int,
) -> list[SubtitleCue]:
    """
    翻译一个字幕块。

    初次调用失败后最多重试 MAX_RETRIES 次。API 异常和格式校验失败都会触发重试。
    超过重试次数时抛出异常，由调用方生成带失败标记的安全回退字幕。
    """
    system_prompt, user_prompt = build_prompt(chunk)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            print(
                f"[API] 第 {chunk_number}/{chunk_total} 块，"
                f"第 {attempt}/{MAX_RETRIES + 1} 次请求，"
                f"共 {len(chunk)} 条字幕",
                flush=True,
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                stream=False,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("API 返回了空内容")

            cleaned = strip_optional_code_fence(content)
            translated = parse_srt(cleaned)
            validate_translation(chunk, translated)
            return translated

        except Exception as exc:
            last_error = exc
            print(f"[警告] 本块处理失败：{exc}", file=sys.stderr, flush=True)
            if attempt <= MAX_RETRIES:
                wait_seconds = 2 ** attempt
                print(
                    f"[重试] {wait_seconds} 秒后重试……",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"第 {chunk_number} 块在 {MAX_RETRIES + 1} 次尝试后仍失败：{last_error}"
    )


def safe_fallback(chunk: list[SubtitleCue]) -> list[SubtitleCue]:
    """
    API 最终失败时，不丢字幕、不破坏时间轴。

    保留原文和原时间码，并在正文前标记“翻译失败”，方便之后单独重翻。
    """
    return [
        SubtitleCue(
            index=cue.index,
            timecode=cue.timecode,
            text=f"[翻译失败，以下为日文原文]\n{cue.text}",
        )
        for cue in chunk
    ]


def write_srt(cues: list[SubtitleCue], output_path: Path) -> None:
    """
    原子写入 SRT：
    先写临时文件，再替换目标文件，避免中途断电留下半个文件。
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_content = "\r\n\r\n".join(cue.to_srt() for cue in cues) + "\r\n"

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}_",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(handle)
    temporary_path = Path(temporary_name)

    try:
        # UTF-8 BOM 对 Windows 播放器、以及一些不支持无 BOM UTF-8 的 App 更稳。
        temporary_path.write_bytes(final_content.encode("utf-8-sig"))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="调用 DeepSeek API 将日文 SRT 翻译为中文"
    )
    parser.add_argument("input_srt", help="输入日文 SRT 文件路径")
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径；默认在输入文件同目录生成 chinese_subtitle.srt",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"每次发送的字幕条数，默认 {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 SRT 和分块，不调用 API",
    )
    args = parser.parse_args()

    if not 20 <= args.chunk_size <= 30:
        print(
            "[提示] chunk-size 建议保持在 20～30 条之间。",
            file=sys.stderr,
        )

    input_path = Path(args.input_srt).expanduser().resolve()
    if not input_path.is_file():
        print(f"[错误] 输入文件不存在：{input_path}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name("chinese_subtitle.srt")
    )

    try:
        cues = parse_srt(input_path.read_text(encoding="utf-8-sig"))
        chunks = split_cues(cues, args.chunk_size)
    except Exception as exc:
        print(f"[错误] SRT 解析失败：{exc}", file=sys.stderr)
        return 2

    print(
        f"[信息] 共 {len(cues)} 条字幕，分成 {len(chunks)} 块，"
        f"每块最多 {args.chunk_size} 条。",
        flush=True,
    )
    if args.dry_run:
        for index, chunk in enumerate(chunks, start=1):
            print(
                f"  块 {index}: {chunk[0].index}～{chunk[-1].index}，"
                f"{len(chunk)} 条"
            )
        return 0

    if API_KEY.startswith("在这里填写"):
        print(
            "[错误] 请先在脚本上方填写 API_KEY，"
            "或设置环境变量 DEEPSEEK_API_KEY。",
            file=sys.stderr,
        )
        return 2

    # 重试完全由本脚本控制；关闭 OpenAI 客户端自带重试，避免次数叠加。
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    translated_cues: list[SubtitleCue] = []
    for chunk_number, chunk in enumerate(chunks, start=1):
        try:
            translated_cues.extend(
                translate_chunk(
                    client,
                    chunk,
                    chunk_number=chunk_number,
                    chunk_total=len(chunks),
                )
            )
        except Exception as exc:
            print(f"[错误] {exc}", file=sys.stderr, flush=True)
            translated_cues.extend(safe_fallback(chunk))

    try:
        write_srt(translated_cues, output_path)
    except Exception as exc:
        print(f"[错误] 输出文件写入失败：{exc}", file=sys.stderr)
        return 2

    print(f"[完成] 中文 SRT 已保存：{output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
