#!/usr/bin/env python3
"""
push_to_kaggle.py — upload the SFT dataset and both notebooks to Kaggle.

    export KAGGLE_API_TOKEN=KGAT_...      # the new-style token authenticates as Bearer
    export KAGGLE_USERNAME=fardeensadab   # legacy authenticate() still demands these two,
    export KAGGLE_KEY=unused              # but the HTTP layer prefers KAGGLE_API_TOKEN
    python scripts/push_to_kaggle.py            # create or version everything
    python scripts/push_to_kaggle.py --data     # dataset only
    python scripts/push_to_kaggle.py --kernels  # notebooks only

Note on auth: a `KGAT_…` token is NOT the legacy `kaggle.json` key. Passing it as
KAGGLE_KEY gives 200s on public reads (which need no auth at all) and then 401s on
every write. It must go in KAGGLE_API_TOKEN so kagglesdk sends `Authorization: Bearer`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = os.environ.get("KAGGLE_USERNAME", "fardeensadab")

DATASET_ID = f"{USER}/deepseek-persuasion-sft"
TRAIN_KERNEL_ID = f"{USER}/persuasion-01-distil-deepseek-into-qwen3-4b"
# NB: Kaggle derives the slug from the title, so this id must match the slug of
# "Persuasion 02 — Arena: base vs fine-tuned" or every push creates a second kernel.
EVAL_KERNEL_ID = f"{USER}/persuasion-02-arena-base-vs-fine-tuned"
RESIST_TRAIN_KERNEL_ID = f"{USER}/persuasion-03-distil-grok-resistance"
RESIST_EVAL_KERNEL_ID = f"{USER}/persuasion-04-arena-resistance"


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def push_dataset(a) -> None:
    # Stage into a temp dir rather than uploading kaggle_dataset/ in place: running the test
    # suite leaves a __pycache__ behind, and Kaggle would happily ship the .pyc files.
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "payload"
        shutil.copytree(ROOT / "kaggle_dataset", src,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))
        (src / "dataset-metadata.json").write_text(json.dumps({
            "title": "DeepSeek Persuasion SFT",
            "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2), encoding="utf-8")

        try:
            a.dataset_status(DATASET_ID)
            exists = True
        except Exception:                                            # noqa: BLE001
            exists = False

        if exists:
            print(f"versioning dataset {DATASET_ID} …")
            a.dataset_create_version(str(src), version_notes="update", dir_mode="zip", quiet=False)
        else:
            print(f"creating dataset {DATASET_ID} …")
            a.dataset_create_new(str(src), public=False, dir_mode="zip", quiet=False)
    print(f"  https://www.kaggle.com/datasets/{DATASET_ID}")


def push_kernel(a, kernel_id: str, title: str, notebook: Path,
                dataset_sources: list[str], kernel_sources: list[str]) -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(notebook, td / notebook.name)
        meta = {
            "id": kernel_id,
            "title": title,
            "code_file": notebook.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": dataset_sources,
            "competition_sources": [],
            "kernel_sources": kernel_sources,
            "model_sources": [],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"pushing kernel {kernel_id} …")
        try:
            print(" ", a.kernels_push(str(td)))
        except Exception as e:                                       # noqa: BLE001
            # A kernel_source that has never been run has no output yet and is rejected.
            if kernel_sources:
                print(f"  retrying without kernel_sources ({e})")
                meta["kernel_sources"] = []
                (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(" ", a.kernels_push(str(td)))
            else:
                raise
        print(f"  https://www.kaggle.com/code/{kernel_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--kernels", action="store_true")
    # A push COMMITS AND RUNS the notebook. Pushing 01 when only 02 changed restarts a
    # multi-hour training job for nothing, so make it possible to target one kernel.
    ap.add_argument("--only", choices=["train", "eval", "resist-train", "resist-eval"],
                    help="push just one notebook (default: all four)")
    args = ap.parse_args()
    do_all = not (args.data or args.kernels or args.only)

    a = api()
    if do_all or args.data:
        push_dataset(a)
    if do_all or args.kernels or args.only:
        want = args.only
        if want in (None, "train"):
            push_kernel(a, TRAIN_KERNEL_ID, "Persuasion 01 — Distil DeepSeek into Qwen3-4B",
                        ROOT / "notebooks" / "01_train_sft.ipynb", [DATASET_ID], [])
        if want in (None, "eval"):
            push_kernel(a, EVAL_KERNEL_ID, "Persuasion 02 — Arena: base vs fine-tuned",
                        ROOT / "notebooks" / "02_eval_arena.ipynb", [DATASET_ID], [TRAIN_KERNEL_ID])
        if want in (None, "resist-train"):
            push_kernel(a, RESIST_TRAIN_KERNEL_ID, "Persuasion 03 — Distil Grok resistance",
                        ROOT / "notebooks" / "03_train_resistance.ipynb", [DATASET_ID], [])
        if want in (None, "resist-eval"):
            # needs BOTH adapters: the notebook-01 persuader attacks, the notebook-03 resistor defends
            push_kernel(a, RESIST_EVAL_KERNEL_ID, "Persuasion 04 — Arena: resistance",
                        ROOT / "notebooks" / "04_eval_resistance.ipynb", [DATASET_ID],
                        [TRAIN_KERNEL_ID, RESIST_TRAIN_KERNEL_ID])


if __name__ == "__main__":
    main()
