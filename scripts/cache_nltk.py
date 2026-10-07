"""Provision and verify the NLTK data required by pinecone-text BM25."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nltk


REQUIRED_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}


def nltk_data_directory() -> Path:
    configured = os.getenv("NLTK_DATA")
    return Path(configured) if configured else Path.home() / "nltk_data"


def verify_resources(target: Path) -> None:
    target = target.resolve()
    if str(target) not in nltk.data.path:
        nltk.data.path.insert(0, str(target))

    missing = []
    for package, resource_path in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(resource_path, paths=[str(target)])
        except LookupError:
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Missing required NLTK resources: " + ", ".join(sorted(missing))
        )


def provision_resources(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for package in REQUIRED_RESOURCES:
        downloaded = nltk.download(
            package,
            download_dir=str(target),
            quiet=False,
            raise_on_error=True,
        )
        if not downloaded:
            raise RuntimeError(f"NLTK downloader did not provision {package}")
    verify_resources(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    target = nltk_data_directory()
    if args.verify_only:
        verify_resources(target)
    else:
        provision_resources(target)

    print(f"NLTK resources ready at {target}")


if __name__ == "__main__":
    main()
