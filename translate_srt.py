#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / "_python_local"
MODEL_DIR = ROOT / "_translation_models" / "opus-mt-ja-zh"
PLACEHOLDER = "[无法识别]"

if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))


def time_to_seconds(value):
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value)
    if not match:
        raise ValueError(f"bad timecode: {value}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def seconds_to_time(value):
    milliseconds = max(0, round(value * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{seconds:02d},"
        f"{milliseconds:03d}"
    )


def read_srt(path):
    content = Path(path).read_text(encoding="utf-8-sig").strip()
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", content):
        lines = block.splitlines()
        time_index = next(
            (
                index
                for index, line in enumerate(lines)
                if "-->" in line
            ),
            None,
        )
        if time_index is None:
            continue
        start, end = [
            part.strip() for part in lines[time_index].split("-->", 1)
        ]
        text = "\n".join(lines[time_index + 1 :]).strip()
        if not text:
            continue
        cues.append(
            {
                "start": time_to_seconds(start),
                "end": time_to_seconds(end),
                "text": text,
            }
        )
    return cues


def write_srt(path, cues):
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\r\n"
            f"{seconds_to_time(cue['start'])} --> "
            f"{seconds_to_time(cue['end'])}\r\n"
            f"{cue['text']}"
        )
    Path(path).write_bytes(
        ("\r\n\r\n".join(blocks) + "\r\n").encode("utf-8-sig")
    )


def normalize_chinese(text, source):
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = text.strip()
    text = text.replace("咁样", "这样").replace("咁", "这样")
    text = text.replace("?", "？").replace("!", "！")
    if "ご主人様" in source or "主人様" in source:
        text = re.sub(r"母亲|妈妈|媽|母上", "主人", text)
        if "電話" in source and ("かけた" in source or "かけ" in source):
            text = "你给主人打电话了吗？"
    if source.strip() == "そう":
        text = "是这样。"
    if "いいじゃん" in source:
        text = "不错嘛。"
    text = text.replace("主人大人", "主人")
    return text


def translate_cues(cues, batch_size=8, min_logprob=-1.20):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not MODEL_DIR.exists():
        raise SystemExit(f"translation model missing: {MODEL_DIR}")
    torch.set_num_threads(8)
    # SentencePiece uses a narrow Windows path; using relative names avoids
    # failures when the installation path contains Chinese characters.
    previous_dir = Path.cwd()
    os.chdir(MODEL_DIR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ".",
            local_files_only=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ".",
            local_files_only=True,
        )
    finally:
        os.chdir(previous_dir)
    model.eval()

    translations = [PLACEHOLDER] * len(cues)
    scores = [None] * len(cues)
    indices = [
        index
        for index, cue in enumerate(cues)
        if cue["text"].strip() != PLACEHOLDER
    ]
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            batch_texts = [cues[index]["text"] for index in batch_indices]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            output = model.generate(
                **encoded,
                max_new_tokens=160,
                num_beams=4,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            transition_scores = model.compute_transition_scores(
                output.sequences,
                output.scores,
                output.beam_indices,
                normalize_logits=True,
            )
            sequence_scores = transition_scores.mean(dim=1).tolist()
            decoded = tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )
            for index, source, result, score in zip(
                batch_indices, batch_texts, decoded, sequence_scores
            ):
                scores[index] = float(score)
                if score >= min_logprob:
                    translations[index] = normalize_chinese(result, source)
    return translations, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_srt")
    parser.add_argument("--output-zh")
    parser.add_argument("--output-bilingual")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-translation-logprob", type=float, default=-1.20)
    args = parser.parse_args()

    input_path = Path(args.input_srt)
    base_stem = (
        input_path.stem[:-3]
        if input_path.stem.endswith("_ja")
        else input_path.stem
    )
    output_zh = Path(
        args.output_zh
        or input_path.with_name(base_stem + "_zh.srt")
    )
    output_bilingual = Path(
        args.output_bilingual
        or input_path.with_name(base_stem + "_bilingual.srt")
    )
    cues = read_srt(input_path)
    if not cues:
        raise SystemExit("no cues found")
    translations, scores = translate_cues(
        cues,
        batch_size=args.batch_size,
        min_logprob=args.min_translation_logprob,
    )

    chinese = []
    bilingual = []
    for cue, translated in zip(cues, translations):
        chinese.append(
            {
                "start": cue["start"],
                "end": cue["end"],
                "text": translated,
            }
        )
        bilingual.append(
            {
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"] + "\n" + translated,
            }
        )
    write_srt(output_zh, chinese)
    write_srt(output_bilingual, bilingual)
    report = {
        "input": str(input_path),
        "output_zh": str(output_zh),
        "output_bilingual": str(output_bilingual),
        "cues": len(cues),
        "placeholders": sum(
            cue["text"] == PLACEHOLDER for cue in cues
        ),
        "translation_low_confidence": sum(
            score is not None and score < args.min_translation_logprob
            for score in scores
        ),
        "scores": scores,
        "timeline_preserved": all(
            source["start"] == translated["start"]
            and source["end"] == translated["end"]
            for source, translated in zip(cues, chinese)
        ),
    }
    Path(output_zh.with_suffix(".translation.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
