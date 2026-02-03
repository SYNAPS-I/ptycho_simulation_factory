#!/usr/bin/env python3
import argparse
import os
import warnings
from typing import List

from PIL import Image
from tqdm import tqdm


def read_nonempty_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def resolve_path(root_dir: str, entry: str) -> str:
    if os.path.isabs(entry):
        return entry
    return os.path.join(root_dir, entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check object images by attempting to open them with PIL."
    )
    parser.add_argument("object_file_list", type=str)
    parser.add_argument("root_dir", type=str)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first error instead of continuing.",
    )
    args = parser.parse_args()

    entries = read_nonempty_lines(args.object_file_list)
    total = len(entries)
    failures = 0

    for idx, entry in enumerate(tqdm(entries, desc="Checking images", unit="file")):
        path = resolve_path(args.root_dir, entry)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with Image.open(path) as img:
                    img.load()
            truncated = any(
                "Truncated File Read" in str(w.message) for w in caught
            )
            if truncated:
                failures += 1
                print(f"[{idx + 1}/{total}] WARN: {path} -> Truncated File Read")
                if args.stop_on_error:
                    break
        except Exception as exc:
            failures += 1
            print(f"[{idx + 1}/{total}] ERROR: {path} -> {exc}")
            if args.stop_on_error:
                break

    print(f"Checked {total} files, {failures} failures.")


if __name__ == "__main__":
    main()
