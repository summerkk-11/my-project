#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from pathlib import Path


PLACEHOLDER = "[无法识别]"
DEFAULT_SUSPICIOUS = ("注射", "お注射", "針", "病院", "医療", "患者")
DEFAULT_REPLACEMENTS = {"お母さん": "ご主人様", "ママ": "ご主人様"}


def normalized_text(text):
    return re.sub(r"[\s、。！？!?,.…「」『』（）()]+", "", text).lower()


def timecode(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )


def segment_confidence(segment):
    probabilities = [
        float(word.get("probability", 0.0))
        for word in (segment.get("words") or [])
        if word.get("word")
    ]
    minimum = min(probabilities, default=0.0)
    median = statistics.median(probabilities) if probabilities else 0.0
    return minimum, median


def process(
    input_json,
    output_srt,
    report_json,
    min_word_median=0.55,
    min_word_probability=0.03,
    max_no_speech=0.60,
    max_compression_ratio=2.40,
    min_avg_logprob=-1.00,
    repetition_window=6,
    suspicious_terms=DEFAULT_SUSPICIOUS,
    replacements=DEFAULT_REPLACEMENTS,
):
    data = json.loads(Path(input_json).read_text(encoding="utf-8-sig"))
    segments = data.get("segments") or []
    recent = []
    cleaned = []
    report = {
        "input_segments": len(segments),
        "kept_segments": 0,
        "low_confidence": 0,
        "repetitions": 0,
        "suspicious_terms": 0,
        "replaced_terms": 0,
        "placeholder_segments": 0,
        "flags": [],
    }

    for index, segment in enumerate(segments):
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        if not text:
            continue
        original = text
        for source, target in replacements.items():
            if source in text:
                text = text.replace(source, target)
                report["replaced_terms"] += 1

        minimum, median = segment_confidence(segment)
        flags = []
        if float(segment.get("no_speech_prob", 0)) > max_no_speech:
            flags.append("no_speech")
        if float(segment.get("compression_ratio", 0)) > max_compression_ratio:
            flags.append("compression")
        if float(segment.get("avg_logprob", 0)) < min_avg_logprob:
            flags.append("logprob")
        if median and median < min_word_median:
            flags.append("low_word_median")
        if minimum < min_word_probability and median < 0.80:
            flags.append("low_word_min")

        normalized = normalized_text(text)
        start = float(segment.get("start", 0))
        recent = [
            item for item in recent if start - item[0] <= 30
        ][-repetition_window:]
        repeated = any(
            item[1] == normalized for item in recent
        )
        if repeated:
            flags.append("repetition")
        recent.append((start, normalized))

        suspicious = [
            term for term in suspicious_terms if term in text
        ]
        if suspicious:
            flags.extend(f"suspicious:{term}" for term in suspicious)

        if flags:
            report["flags"].append(
                {
                    "index": index,
                    "start": start,
                    "end": float(segment.get("end", start)),
                    "original": original,
                    "candidate": text,
                    "word_min": minimum,
                    "word_median": median,
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": segment.get("no_speech_prob"),
                    "compression_ratio": segment.get("compression_ratio"),
                    "flags": flags,
                }
            )

        if "low_word_median" in flags or "low_word_min" in flags:
            report["low_confidence"] += 1
        if "repetition" in flags:
            report["repetitions"] += 1
        if suspicious:
            report["suspicious_terms"] += 1

        if flags:
            text = PLACEHOLDER
            report["placeholder_segments"] += 1
        else:
            report["kept_segments"] += 1

        cleaned.append(
            {
                "start": start,
                "end": float(segment.get("end", start)),
                "text": text,
            }
        )

    blocks = []
    for index, cue in enumerate(cleaned, start=1):
        blocks.append(
            f"{index}\r\n"
            f"{timecode(cue['start'])} --> {timecode(cue['end'])}\r\n"
            f"{cue['text']}"
        )
    Path(output_srt).write_bytes(
        ("\r\n\r\n".join(blocks) + "\r\n").encode("utf-8-sig")
    )
    Path(report_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_srt")
    parser.add_argument("report_json")
    parser.add_argument("--min-word-median", type=float, default=0.55)
    parser.add_argument("--min-word-probability", type=float, default=0.03)
    parser.add_argument("--max-no-speech", type=float, default=0.60)
    parser.add_argument("--max-compression-ratio", type=float, default=2.40)
    parser.add_argument("--min-avg-logprob", type=float, default=-1.00)
    parser.add_argument("--config")
    args = parser.parse_args()
    overrides = {}
    if args.config:
        config = json.loads(
            Path(args.config).read_text(encoding="utf-8-sig")
        )
        confidence = config.get("min_confidence", {})
        overrides = {
            "min_word_median": confidence.get("word_median", 0.55),
            "min_word_probability": confidence.get("word_minimum", 0.03),
            "max_no_speech": confidence.get("max_no_speech", 0.60),
            "max_compression_ratio": confidence.get(
                "max_compression_ratio", 2.40
            ),
            "min_avg_logprob": confidence.get("min_avg_logprob", -1.0),
            "suspicious_terms": tuple(config.get("suspicious_terms", [])),
            "replacements": dict(config.get("domain_replacements", {})),
        }
    process(
        args.input_json,
        args.output_srt,
        args.report_json,
        min_word_median=overrides.get(
            "min_word_median", args.min_word_median
        ),
        min_word_probability=overrides.get(
            "min_word_probability", args.min_word_probability
        ),
        max_no_speech=overrides.get(
            "max_no_speech", args.max_no_speech
        ),
        max_compression_ratio=overrides.get(
            "max_compression_ratio", args.max_compression_ratio
        ),
        min_avg_logprob=overrides.get(
            "min_avg_logprob", args.min_avg_logprob
        ),
        suspicious_terms=overrides.get(
            "suspicious_terms", DEFAULT_SUSPICIOUS
        ),
        replacements=overrides.get(
            "replacements", DEFAULT_REPLACEMENTS
        ),
    )


if __name__ == "__main__":
    main()
