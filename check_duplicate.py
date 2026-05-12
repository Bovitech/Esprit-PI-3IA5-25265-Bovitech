"""
check_duplicates.py
Detects duplicate or near-duplicate images across your dataset.
Run this AFTER merging new data, BEFORE retraining.

Usage: python check_duplicates.py
"""

import os
import json
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict

DATA_DIR   = "data"
SPLITS     = ["train", "valid", "test"]
HASH_SIZE  = 8       # perceptual hash size (higher = stricter matching)
MAX_DIST   = 5       # hamming distance threshold (0=identical, ≤5=near-duplicate)


def hash_all_images(data_dir, splits):
    print("🔍 Hashing all images...\n")
    records = []
    for split in splits:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            images = [p for p in cls_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            print(f"   {split:6s} / {cls_dir.name:20s} : {len(images):>4d} images")
            for img_path in images:
                try:
                    img  = Image.open(img_path).convert("RGB")
                    h    = imagehash.phash(img, hash_size=HASH_SIZE)
                    records.append({
                        "path" : str(img_path),
                        "split": split,
                        "cls"  : cls_dir.name,
                        "hash" : str(h),
                    })
                except Exception as e:
                    print(f"   ⚠️  Could not read {img_path.name}: {e}")
    return records


def find_duplicates(records, max_dist):
    print("\n🔎 Finding duplicates...\n")
    hashes = [(r, imagehash.hex_to_hash(r["hash"])) for r in records]

    groups   = []
    visited  = set()

    for i, (ri, hi) in enumerate(hashes):
        if i in visited:
            continue
        group = [ri]
        visited.add(i)
        for j, (rj, hj) in enumerate(hashes):
            if j <= i or j in visited:
                continue
            if hi - hj <= max_dist:
                group.append(rj)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    return groups


def report(groups, records):
    total_images = len(records)

    # Cross-split duplicates (train↔test, train↔valid) — most dangerous
    cross_split = [g for g in groups
                   if len({r["split"] for r in g}) > 1]

    # Within-split duplicates — wasted data but not dangerous
    same_split  = [g for g in groups
                   if len({r["split"] for r in g}) == 1]

    print("=" * 60)
    print("  DUPLICATE REPORT")
    print("=" * 60)
    print(f"  Total images scanned : {total_images}")
    print(f"  Duplicate groups     : {len(groups)}")
    print(f"  ├─ Cross-split (⚠️ dangerous) : {len(cross_split)}")
    print(f"  └─ Same-split  (ℹ️ redundant) : {len(same_split)}")
    print()

    if cross_split:
        print("⚠️  CROSS-SPLIT DUPLICATES (these inflate your accuracy!):")
        print("   You should DELETE the copies from valid/ and test/\n")
        for i, group in enumerate(cross_split, 1):
            print(f"   Group {i}:")
            for r in group:
                print(f"     [{r['split']:5s} / {r['cls']:20s}]  {Path(r['path']).name}")
        print()
    else:
        print("✅ No cross-split duplicates found — your splits are clean.\n")

    if same_split:
        print("ℹ️  SAME-SPLIT DUPLICATES (redundant but not harmful):")
        for i, group in enumerate(same_split[:10], 1):   # show first 10
            print(f"   Group {i}  [{group[0]['split']} / {group[0]['cls']}]:")
            for r in group:
                print(f"     {Path(r['path']).name}")
        if len(same_split) > 10:
            print(f"   ... and {len(same_split)-10} more groups")
        print()

    # Per-class count summary
    print("=" * 60)
    print("  CLASS SIZE SUMMARY (after merge)")
    print("=" * 60)
    from collections import Counter
    for split in SPLITS:
        split_records = [r for r in records if r["split"] == split]
        counts = Counter(r["cls"] for r in split_records)
        if not counts:
            continue
        print(f"\n  {split.upper()}:")
        for cls, cnt in sorted(counts.items()):
            bar     = "█" * min(cnt // 5, 40)
            warning = "  ⚠️  very few!" if cnt < 20 else ""
            print(f"    {cls:<20s} {cnt:>4d}  {bar}{warning}")

    # Save report
    report_data = {
        "total_images"    : total_images,
        "cross_split_dupes": [[r["path"] for r in g] for g in cross_split],
        "same_split_dupes" : [[r["path"] for r in g] for g in same_split],
    }
    with open("outputs/duplicate_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n📄 Full report saved to outputs/duplicate_report.json")


def remove_cross_split_duplicates(groups):
    """
    Automatically removes the test/valid copy when a duplicate
    is found across splits (keeps the train copy).
    Only runs if you uncomment the call below.
    """
    removed = 0
    for group in groups:
        splits_in_group = {r["split"]: r for r in group}
        # keep train, remove from valid/test
        for split in ["valid", "test"]:
            if split in splits_in_group and "train" in splits_in_group:
                path = Path(splits_in_group[split]["path"])
                path.unlink()
                print(f"   🗑  Removed: {path}")
                removed += 1
    print(f"\n✅ Removed {removed} duplicate files from valid/test.")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    records = hash_all_images(DATA_DIR, SPLITS)
    groups  = find_duplicates(records, MAX_DIST)
    report(groups, records)

    # ── Uncomment to AUTO-DELETE cross-split duplicates ──
    # cross_split = [g for g in groups if len({r["split"] for r in g}) > 1]
    # if cross_split:
    #     confirm = input("\nAuto-delete cross-split dupes from valid/test? (yes/no): ")
    #     if confirm.strip().lower() == "yes":
    #         remove_cross_split_duplicates(cross_split)