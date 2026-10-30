#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from quality_postprocess import process as postprocess_quality


ROOT = Path(__file__).resolve().parent
PROGRAM = ROOT / "Faster-Whisper-XXL"
EXE = PROGRAM / "faster-whisper-xxl.exe"
FFMPEG = PROGRAM / "ffmpeg.exe"
MODELS = PROGRAM / "_models"
MERGER = ROOT / "merge_srt.ps1"
TRANSLATOR = ROOT / "translate_srt.py"
QUALITY_CONFIG = ROOT / "quality_pipeline_config.json"
SOURCE_DIR = ROOT / "手机上传"
JOBS_DIR = ROOT / "batch_jobs_quality"
OUTPUT_DIR = ROOT / "字幕成果_高质量"
LOG_PATH = ROOT / "高质量识别进度.txt"


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def media_duration(path):
    result = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-nostdin", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr
    )
    if not match:
        raise RuntimeError(f"无法读取音频时长: {path.name}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def split_audio(source, job_dir, duration, part_seconds):
    expected = math.ceil(duration / part_seconds)
    existing = sorted(job_dir.glob("part_*.wav"))
    if len(existing) == expected:
        log(f"  切片已存在，跳过切分: {len(existing)} 段")
        return existing

    log(f"  正在切成 {expected} 个 15 分钟 WAV 片段")
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "-1",
        "-f",
        "segment",
        "-segment_time",
        str(part_seconds),
        "-reset_timestamps",
        "1",
        str(job_dir / "part_%03d.wav"),
    ]
    result = subprocess.run(
        command,
        cwd=PROGRAM,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError("FFmpeg 切分失败: " + result.stderr[-2000:])
    parts = sorted(job_dir.glob("part_*.wav"))
    if len(parts) != expected:
        raise RuntimeError(f"切片数量异常: {len(parts)} != {expected}")
    return parts


def stream_command(command, environment):
    process = subprocess.Popen(
        command,
        cwd=PROGRAM,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def recognize_source(source, environment, config):
    final_path = OUTPUT_DIR / f"{source.stem}_ja.srt"
    if final_path.exists() and final_path.stat().st_size > 0:
        log(f"已存在，跳过: {final_path.name}")
        return True

    model_name = str(config["model"])
    part_seconds = int(config.get("audio_segment_seconds", 900))
    duration = media_duration(source)
    log(
        f"开始 {source.name}，时长 {duration / 3600:.2f} 小时，"
        f"大小 {source.stat().st_size / 2**20:.1f} MiB"
    )
    job_dir = JOBS_DIR / source.stem
    job_dir.mkdir(parents=True, exist_ok=True)
    parts = split_audio(source, job_dir, duration, part_seconds)

    marker_suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
    pending = [
        part
        for part in parts
        if not (
            job_dir / f"{part.stem}.{marker_suffix}.done"
        ).exists()
    ]
    if pending:
        log(f"  待识别分段: {len(pending)}/{len(parts)}")
        command = [
            str(EXE),
            *[str(part) for part in pending],
            "--device",
            "cpu",
            "--model",
            model_name,
            "--model_dir",
            str(MODELS),
            "--language",
            "ja",
            "--task",
            "transcribe",
            "--threads",
            str(config.get("threads", 8)),
            "--compute_type",
            str(config.get("compute_type", "int8")),
            "--condition_on_previous_text",
            str(config.get("condition_on_previous_text", False)),
            "--initial_prompt",
            str(config.get("initial_prompt") or "None"),
            "--vad_filter",
            str(config.get("vad_filter", True)),
            "--hallucination_silence_threshold",
            str(config.get("hallucination_silence_threshold", 2.5)),
            "--output_format",
            "srt",
            "json",
            "--output_dir",
            str(job_dir),
            "--beep_off",
        ]
        hotwords = config.get("hotwords")
        if hotwords:
            command.extend(["--hotwords", str(hotwords)])
        exit_code = stream_command(command, environment)
        if exit_code != 0:
            raise RuntimeError(f"识别程序退出码 {exit_code}")
        confidence = config.get("min_confidence", {})
        for part in pending:
            srt_path = job_dir / f"{part.stem}.srt"
            json_path = job_dir / f"{part.stem}.json"
            report_path = job_dir / f"{part.stem}.quality.json"
            if not json_path.exists():
                raise RuntimeError(f"缺少置信度 JSON: {json_path.name}")
            postprocess_quality(
                json_path,
                srt_path,
                report_path,
                min_word_median=float(
                    confidence.get("word_median", 0.55)
                ),
                min_word_probability=float(
                    confidence.get("word_minimum", 0.03)
                ),
                max_no_speech=float(
                    confidence.get("max_no_speech", 0.60)
                ),
                max_compression_ratio=float(
                    confidence.get("max_compression_ratio", 2.40)
                ),
                min_avg_logprob=float(
                    confidence.get("min_avg_logprob", -1.0)
                ),
                suspicious_terms=tuple(config.get("suspicious_terms", [])),
                replacements=dict(config.get("domain_replacements", {})),
            )
            (job_dir / f"{part.stem}.{marker_suffix}.done").write_text(
                "ok\n", encoding="ascii"
            )
    else:
        log("  所有分段均已识别，继续合并")

    log("  正在平移并合并时间轴")
    merge = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(MERGER),
            "-PartDirectory",
            str(job_dir),
            "-OutputFile",
            str(final_path),
            "-FFprobe",
            str(PROGRAM / "ffprobe.exe"),
            "-FFmpeg",
            str(FFMPEG),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if merge.returncode != 0:
        raise RuntimeError("SRT 合并失败: " + merge.stderr[-2000:])
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError("最终字幕为空")
    log(f"  完成: {final_path}")

    if config.get("outputs", {}).get("chinese_srt", True):
        log("  正在生成中文和双语 SRT")
        translation_command = [
            sys.executable,
            str(TRANSLATOR),
            str(final_path),
            "--min-translation-logprob",
            str(config.get("min_translation_logprob", -0.9)),
        ]
        translation = subprocess.run(
            translation_command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if translation.returncode != 0:
            log(
                "  翻译失败，日文字幕已保留: "
                + translation.stderr[-2000:]
            )
        else:
            log("  中文和双语字幕已生成")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        help="可选：只处理指定音频；省略时处理手机上传目录中的全部音频",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8-sig"))
    for path in (EXE, FFMPEG, MERGER, TRANSLATOR):
        if not path.exists():
            raise SystemExit(f"缺少必要文件: {path}")
    model_name = str(config["model"])
    model_file = MODELS / f"faster-whisper-{model_name}" / "model.bin"
    if not model_file.exists():
        raise SystemExit(
            f"缺少 {model_name} 模型: {model_file}"
        )

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HOME": str(ROOT / "_cache" / "huggingface"),
            "TORCH_HOME": str(ROOT / "_cache" / "torch"),
            "XDG_CACHE_HOME": str(ROOT / "_cache"),
            "TEMP": str(ROOT / "_temp"),
            "TMP": str(ROOT / "_temp"),
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": str(config.get("threads", 8)),
        }
    )

    extensions = {
        ".mp3",
        ".m4a",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".amr",
        ".wma",
        ".mp4",
        ".mkv",
    }
    if args.inputs:
        sources = []
        for value in args.inputs:
            path = Path(value)
            if not path.is_absolute():
                path = SOURCE_DIR / path
            path = path.resolve()
            if not path.is_file():
                raise SystemExit(f"输入文件不存在: {path}")
            if path.suffix.lower() not in extensions:
                raise SystemExit(f"不支持的格式: {path.suffix}")
            sources.append(path)
        sources = sorted(set(sources))
    else:
        sources = sorted(
            path
            for path in SOURCE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
    if not sources:
        raise SystemExit(f"没有找到支持的音频: {SOURCE_DIR}")
    log(f"批量任务开始，共 {len(sources)} 个文件")
    failed = []
    started = time.monotonic()
    for index, source in enumerate(sources, start=1):
        log(f"===== 文件 {index}/{len(sources)} =====")
        try:
            recognize_source(source, environment, config)
        except Exception as error:
            failed.append(source.name)
            log(f"失败 {source.name}: {error}")

    elapsed = time.monotonic() - started
    if failed:
        log(
            f"批量任务结束，耗时 {elapsed / 60:.1f} 分钟；"
            f"失败: {', '.join(failed)}"
        )
        return 1
    log(f"全部完成，耗时 {elapsed / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
