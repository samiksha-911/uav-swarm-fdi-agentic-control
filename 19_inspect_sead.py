#!/usr/bin/env python3
"""
STEP 19 — SEAD DATASET INSPECTION
=================================

Purpose:
    Inspect mapping.json and "processed data.rar" WITHOUT extracting
    the 551+ MB archive.

This step DOES NOT:
    - modify Step 15
    - train a model
    - merge SEAD with the Basic dataset
    - extract the large archive

It only determines:
    1. where the SEAD files are
    2. archive contents / file types / sizes
    3. mapping.json structure
    4. possible class/run information
    5. whether SEAD can realistically augment the current 7 classes

Run from anywhere:
    python 19_inspect_sead.py
"""

from pathlib import Path
import json
import subprocess
import shutil
import re
from collections import Counter

HOME = Path.home()
SEARCH_DIRS = [
    HOME / "Downloads",
    HOME / "Desktop",
    HOME / "Documents",
    HOME / "uav_fdi_data",
]

RAR_NAMES = {"processed data.rar", "processed_data.rar", "processed-data.rar"}
JSON_NAMES = {"mapping.json"}

def find_files():
    rar_files = []
    json_files = []

    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        try:
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.lower() in {x.lower() for x in RAR_NAMES}:
                    rar_files.append(p)
                elif p.name.lower() in {x.lower() for x in JSON_NAMES}:
                    json_files.append(p)
        except PermissionError:
            pass

    # Also check home one level deeper if not found.
    if not rar_files:
        try:
            for p in HOME.rglob("*.rar"):
                if p.is_file() and "processed" in p.name.lower():
                    rar_files.append(p)
        except PermissionError:
            pass

    return sorted(set(rar_files)), sorted(set(json_files))


def print_mapping_summary(path):
    print("\n" + "-" * 70)
    print("2. MAPPING.JSON")
    print("-" * 70)
    print("Path:", path)
    print("Size:", path.stat().st_size / 1024, "KB")

    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print("Could not parse JSON:", repr(e))
        return

    print("Top-level type:", type(obj).__name__)

    if isinstance(obj, dict):
        print("Top-level keys:", list(obj.keys())[:100])
        print("Number of top-level keys:", len(obj))

        # Print compact structural information.
        for k, v in list(obj.items())[:30]:
            if isinstance(v, dict):
                print(f"  {k!r}: dict ({len(v)} keys)")
                print("      sample keys:", list(v.keys())[:8])
            elif isinstance(v, list):
                print(f"  {k!r}: list ({len(v)} items)")
                if v:
                    print("      first item type:", type(v[0]).__name__)
                    print("      first item:", repr(v[0])[:300])
            else:
                print(f"  {k!r}: {type(v).__name__} = {repr(v)[:200]}")

    elif isinstance(obj, list):
        print("Number of items:", len(obj))
        for i, item in enumerate(obj[:10]):
            print(f"  [{i}] {type(item).__name__}: {repr(item)[:500]}")

    # Search serialized mapping for likely labels/classes/run identifiers.
    raw = json.dumps(obj, ensure_ascii=False)
    patterns = [
        r"Compass",
        r"RC Failure",
        r"GPS Failure",
        r"Gyro Failure",
        r"Accelerometer Failure",
        r"Barometer Failure",
        r"No Failure",
        r"failure",
        r"fault",
        r"class",
        r"label",
        r"run",
        r"flight",
    ]

    print("\nKeyword counts:")
    for pat in patterns:
        print(f"  {pat:24s}: {len(re.findall(pat, raw, flags=re.I))}")


def inspect_rar(path):
    print("\n" + "-" * 70)
    print("3. PROCESSED DATA ARCHIVE")
    print("-" * 70)
    print("Path:", path)
    print("Size: %.2f MB" % (path.stat().st_size / (1024 * 1024)))

    unrar = shutil.which("unrar")
    if not unrar:
        print("ERROR: 'unrar' is not installed/in PATH.")
        print("Install with:")
        print("  sudo apt update && sudo apt install unrar")
        return

    print("\nArchive test:")
    result = subprocess.run(
        [unrar, "t", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    print(result.stdout[-5000:])

    print("\nArchive listing (first 120 entries):")
    result = subprocess.run(
        [unrar, "lb", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )

    lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    if not lines:
        print("No archive entries returned.")
        return

    for line in lines[:120]:
        print(" ", line)

    print("\nTotal archive entries:", len(lines))

    ext_counter = Counter()
    for name in lines:
        suffix = Path(name).suffix.lower() or "[no extension]"
        ext_counter[suffix] += 1

    print("\nFile extension distribution:")
    for ext, n in ext_counter.most_common():
        print(f"  {ext:15s}: {n}")

    print("\nIMPORTANT:")
    print("Do NOT extract the whole archive yet.")
    print("The next step depends on the archive structure shown above.")


def main():
    print("=" * 70)
    print("STEP 19 — SEAD DATASET INSPECTION")
    print("=" * 70)
    print("\nNo model training. No data merging. No large extraction.")

    rar_files, json_files = find_files()

    print("\n1. FILE LOCATION")
    print("-" * 70)

    print("\nRAR candidates:")
    if rar_files:
        for p in rar_files:
            print(" ", p)
    else:
        print("  NOT FOUND")

    print("\nmapping.json candidates:")
    if json_files:
        for p in json_files:
            print(" ", p)
    else:
        print("  NOT FOUND")

    if not json_files:
        print("\nWARNING: mapping.json was not found automatically.")

    if json_files:
        # Prefer Downloads/mapping.json if present.
        preferred = [p for p in json_files if p.parent == HOME / "Downloads"]
        mapping = preferred[0] if preferred else json_files[0]
        print_mapping_summary(mapping)

    if rar_files:
        # Prefer exact expected name.
        preferred = [p for p in rar_files if p.name == "processed data.rar"]
        rar = preferred[0] if preferred else rar_files[0]
        inspect_rar(rar)

    print("\n" + "=" * 70)
    print("STEP 19 COMPLETE")
    print("=" * 70)
    print("\nPaste the COMPLETE output here.")
    print("Do not extract or rename anything yet.")
    print("We will use the actual SEAD structure to write Step 20.")


if __name__ == "__main__":
    main()
