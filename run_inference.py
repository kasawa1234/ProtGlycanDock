#!/usr/bin/env python3
"""Run ProtGlycanDock inference for a directory of JSON inputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import threading
from pathlib import Path
from queue import Queue

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json_dir", type=Path, required=True, help="Directory containing inference JSON files.")
    parser.add_argument("--dump_dir", type=Path, default=Path("outputs/inference/default"), help="Inference output directory.")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("release_data/checkpoint"), help="Directory containing <model_name>.pt.")
    parser.add_argument("--model_name", default="protenix_base_default_v0.5.0", help="Checkpoint/model config name.")
    parser.add_argument("--type", default="default", help="Run type. Include 'glycan' to enable glycan features.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs used as a worker pool.")
    parser.add_argument("--seeds", default="42", help="Comma-separated inference seeds passed to inference.sh.")
    parser.add_argument("--split_file", type=Path, default=None, help="Optional JSON or pickle mapping sample ID to split name.")
    parser.add_argument("--split_name", default="test", help="Split selected from --split_file.")
    parser.add_argument("--use_msa", default="true", choices=["true", "false"], help="Whether to use MSA features.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"], help="Inference dtype.")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-sample timeout in seconds.")
    parser.add_argument("--inference_script", type=Path, default=Path(__file__).with_name("inference.sh"), help="Path to inference.sh.")
    return parser.parse_args()


def load_split(split_file: Path | None) -> dict[str, str] | None:
    if split_file is None:
        return None
    if split_file.suffix == ".json":
        with split_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with split_file.open("rb") as handle:
        return pickle.load(handle)


def collect_json_files(json_dir: Path, split: dict[str, str] | None, split_name: str) -> list[Path]:
    files = sorted(json_dir.glob("*.json"))
    if split is None:
        return files
    return [path for path in files if split.get(path.stem) == split_name]


def run_one(json_file: Path, gpu_id: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "MODEL_NAME": args.model_name,
            "TYPE": args.type,
            "DUMP_DIR": str(args.dump_dir),
            "CHECKPOINT_DIR": str(args.checkpoint_dir),
            "SEEDS": args.seeds,
            "USE_MSA": args.use_msa,
            "DTYPE": args.dtype,
        }
    )
    cmd = ["bash", str(args.inference_script), str(json_file)]
    sample = json_file.stem
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        LOGGER.error("Timed out while processing %s on GPU %s", sample, gpu_id)
        return {"status": "timeout", "sample": sample}

    if result.returncode == 0:
        return {"status": "success", "sample": sample}

    message = result.stderr.strip() or result.stdout.strip()
    if "out of memory" in message.lower() or "oom" in message.lower():
        LOGGER.warning("OOM while processing %s on GPU %s", sample, gpu_id)
        return {"status": "oom", "sample": sample}

    LOGGER.error("Failed %s on GPU %s: %s", sample, gpu_id, message)
    return {"status": "error", "sample": sample, "message": message}


def worker(files: Queue[Path], gpus: Queue[str], args: argparse.Namespace, results: list[dict[str, str]], lock: threading.Lock, pbar: tqdm) -> None:
    while True:
        try:
            json_file = files.get_nowait()
        except Exception:
            return
        gpu_id = gpus.get()
        try:
            result = run_one(json_file, gpu_id, args)
            with lock:
                results.append(result)
                pbar.update(1)
        finally:
            gpus.put(gpu_id)
            files.task_done()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    split = load_split(args.split_file)
    json_files = collect_json_files(args.json_dir, split, args.split_name)
    if not json_files:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must contain at least one GPU ID")

    files: Queue[Path] = Queue()
    for path in json_files:
        files.put(path)
    gpus: Queue[str] = Queue()
    for gpu in gpu_ids:
        gpus.put(gpu)

    results: list[dict[str, str]] = []
    lock = threading.Lock()
    with tqdm(total=len(json_files), desc="Processing JSON files", unit="file") as pbar:
        threads = [threading.Thread(target=worker, args=(files, gpus, args, results, lock, pbar)) for _ in gpu_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    summary_path = args.dump_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info("Saved inference summary to %s", summary_path)


if __name__ == "__main__":
    main()
