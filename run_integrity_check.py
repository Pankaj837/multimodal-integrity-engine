#!/usr/bin/env python
"""Run the multimodal integrity check over a folder of submissions.

Usage:
    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports

With no arguments it processes ./sample_submissions into ./reports.
Writes, for every input file, <file>.json and <file>.md, plus summary.json and
summary.md (aggregate table and peer-similarity matrix) and run.log.

Exit codes: 0 = finished (individual bad files are flagged, not fatal),
            2 = invalid arguments / input folder missing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Quiet third-party noise before heavy imports; our own logging stays informative.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal submission integrity & plagiarism detection (PDF, DOCX, MP4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", default="./sample_submissions",
                        help="Folder with submissions (searched recursively).")
    parser.add_argument("--output_dir", default="./reports", help="Where reports are written.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="YAML configuration file.")
    parser.add_argument("--reference_dir", default=None, help="Override the external reference corpus folder.")
    parser.add_argument("--template_file", default=None, help="Optional known prompt/template to always discount.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Override compute device.")
    parser.add_argument("--no_ai", action="store_true", help="Skip AI-generation detection (faster).")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def setup_logging(level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logfile = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    logfile.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level), handlers=[console, logfile], force=True)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "faster_whisper", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print(f"error: input folder not found: {input_dir}", file=sys.stderr)
        return 2

    setup_logging(args.log_level, output_dir)
    log = logging.getLogger("integrity")

    from integrity.config import load_config
    from integrity.pipeline import IntegrityPipeline
    from integrity.reporter import write_reports

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.reference_dir:
        config.reference_dir = str(Path(args.reference_dir).resolve())
    if args.template_file:
        config.boilerplate.template_file = str(Path(args.template_file).resolve())
    if args.device:
        config.models.device = args.device
    if args.no_ai:
        config.ai.enabled = False

    started = time.perf_counter()
    run = IntegrityPipeline(config, ROOT).run(input_dir)
    summary = write_reports(run, output_dir)

    log.info("Finished in %.1fs - reports in %s", time.perf_counter() - started, output_dir)
    print()
    print(f"{'File':34} {'Status':18} {'Risk':14} {'Peer%':>6} {'Ext%':>6} {'AI%':>6}")
    print("-" * 88)

    def fmt(value):
        return "-" if value is None else f"{value:.1f}"

    for doc in summary["documents"]:
        print(
            f"{doc['document_id'][:34]:34} {doc['status']:18} {doc['total_integrity_risk']:14} "
            f"{fmt(doc['peer_similarity_percentage']):>6} {fmt(doc['external_similarity_percentage']):>6} "
            f"{fmt(doc['ai_generated_probability']):>6}"
        )
    print(f"\nSummary: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
