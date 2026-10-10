#!/usr/bin/env python3
"""
Independent Dry-Run Parser & Verification Script for BSDS-A Benchmark

Purpose:
- Loads BSDS_amodal_test.json.
- Applies the benchmark filtering criteria (canonical annotator, isStuff, occlude_rate).
- Counts and prints the exact number of valid target objects.
- Runs purely with standard Python (NO PyTorch, NO model inference, NO CUDA dependencies).
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Optional, Dict, Any, List, Set


def find_default_annotation_file() -> Optional[str]:
    """Finds BSDS_amodal_test.json in standard candidate paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "data", "bsds", "annotations", "BSDS_amodal_test.json"),
        os.path.join(project_root, "data", "bsds", "BSDS_amodal_test.json"),
        os.path.join(project_root, "annotations", "BSDS_amodal_test.json"),
        os.path.join(project_root, "BSDS_amodal_test.json"),
        "data/bsds/annotations/BSDS_amodal_test.json",
        "data/bsds/BSDS_amodal_test.json",
        "BSDS_amodal_test.json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def verify_bsds_dataset(
    ann_file: str,
    ignore_stuff: bool = True,
    filter_occluded: bool = True,
    min_occlude_rate: float = 0.001,
    first_annotator_only: bool = True,
    target_author: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses and counts benchmark targets from BSDS_amodal_test.json without PyTorch.
    """
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    print(f"[Parser] Loading {ann_file} ...")
    with open(ann_file, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    images_dict = {im["id"]: im for im in images if "id" in im}
    num_images = len(images_dict)
    num_ann_records = len(annotations)

    seen_images: Set[Any] = set()
    author_counts: Dict[str, int] = defaultdict(int)
    author_images: Dict[str, Set[Any]] = defaultdict(set)

    raw_regions_count = 0
    valid_objects: List[Dict[str, Any]] = []

    skipped_annotator = 0
    skipped_stuff = 0
    skipped_unoccluded = 0
    skipped_invalid = 0

    per_image_objects: Dict[Any, int] = defaultdict(int)

    for ann in annotations:
        img_id = ann.get("image_id")
        author = str(ann.get("author", "unknown"))

        author_counts[author] += 1
        author_images[author].add(img_id)

        # 1. Annotator filtering
        if target_author is not None:
            if author != target_author:
                skipped_annotator += 1
                continue
        elif first_annotator_only:
            if img_id in seen_images:
                skipped_annotator += 1
                continue
            seen_images.add(img_id)

        # Support both COCOA nested format (ann["regions"]) and standard flat format
        regions = ann.get("regions", [ann] if ("segmentation" in ann or "visible_mask" in ann) else [])
        raw_regions_count += len(regions)

        for r_idx, reg in enumerate(regions):
            # 2. Filter background "stuff"
            is_stuff = bool(reg.get("isStuff", reg.get("is_stuff", False)))
            if ignore_stuff and is_stuff:
                skipped_stuff += 1
                continue

            # 3. Filter unoccluded objects
            occlude_rate = float(reg.get("occlude_rate", 0.0))
            if filter_occluded and occlude_rate <= 0.0:
                skipped_unoccluded += 1
                continue
            if min_occlude_rate > 0 and occlude_rate < min_occlude_rate:
                skipped_unoccluded += 1
                continue

            # 4. Check presence of masks / polygon data
            has_seg = reg.get("segmentation") is not None
            has_vis = reg.get("visible_mask") is not None
            if not has_seg and not has_vis:
                skipped_invalid += 1
                continue

            per_image_objects[img_id] += 1
            valid_objects.append({
                "image_id": img_id,
                "name": reg.get("name", f"obj_{len(valid_objects)}"),
                "occlude_rate": occlude_rate,
                "is_stuff": is_stuff,
                "author": author,
            })

    # Summary statistics
    counts_per_img = list(per_image_objects.values())
    avg_per_img = sum(counts_per_img) / max(len(counts_per_img), 1)
    min_per_img = min(counts_per_img) if counts_per_img else 0
    max_per_img = max(counts_per_img) if counts_per_img else 0

    results = {
        "annotation_file": ann_file,
        "total_images": num_images,
        "total_ann_records": num_ann_records,
        "total_raw_regions_inspected": raw_regions_count,
        "skipped_non_canonical_annotator": skipped_annotator,
        "skipped_stuff": skipped_stuff,
        "skipped_unoccluded": skipped_unoccluded,
        "skipped_invalid": skipped_invalid,
        "valid_benchmark_objects": len(valid_objects),
        "evaluated_images_count": len(per_image_objects),
        "avg_objects_per_image": round(avg_per_img, 2),
        "min_objects_per_image": min_per_img,
        "max_objects_per_image": max_per_img,
        "authors_breakdown": {
            auth: {
                "records": count,
                "unique_images": len(author_images[auth]),
            }
            for auth, count in author_counts.items()
        },
    }

    # Print Clean Formatted Report
    print("\n" + "=" * 65)
    print("        BSDS-A BENCHMARK DATASET DRY-RUN VERIFICATION")
    print("=" * 65)
    print(f"Annotation File           : {ann_file}")
    print(f"Total Images in File      : {num_images}")
    print(f"Total Annotation Records  : {num_ann_records}")
    print(f"Total Raw Regions Read    : {raw_regions_count}")
    print("-" * 65)
    print("Filtering Breakdown:")
    print(f"  - Non-canonical annotators skipped : {skipped_annotator}")
    print(f"  - Background 'stuff' regions skipped : {skipped_stuff}")
    print(f"  - Unoccluded objects skipped (occ<=0): {skipped_unoccluded}")
    print(f"  - Invalid/empty entries skipped     : {skipped_invalid}")
    print("-" * 65)
    print(f"VALID BENCHMARK TARGET OBJECTS: {len(valid_objects)}")
    print("Expected Paper Standard        : 650 objects")
    if len(valid_objects) == 650:
        print("Status                         : [MATCH] Exactly 650 benchmark targets!")
    else:
        diff = len(valid_objects) - 650
        print(f"Status                         : Count: {len(valid_objects)} (Delta vs paper 650: {diff:+d})")
    print("-" * 65)
    print(f"Images with Target Objects: {len(per_image_objects)}")
    print(f"Avg Objects Per Image     : {avg_per_img:.2f} (Min: {min_per_img}, Max: {max_per_img})")
    print("-" * 65)
    print("Annotators Breakdown:")
    for auth, stat in author_counts.items():
        print(f"  - Author '{auth}': {stat} records across {len(author_images[auth])} images")
    print("=" * 65 + "\n")

    return results


def run_synthetic_test(output_path: str = "/tmp/test_bsds_synth.json"):
    """Creates a temporary synthetic JSON with 200 images and 650 objects to test the parser."""
    print("[Test Mode] Generating synthetic BSDS-A annotation JSON for verification...")
    images = []
    annotations = []
    obj_id = 0

    # 200 images, each with 3 or 4 objects (total 650 objects across 200 images)
    for i in range(1, 201):
        images.append({"id": i, "file_name": f"{i:06d}.jpg", "height": 321, "width": 481})
        num_targets = 4 if (i % 4 == 0) else 3

        regions = []
        for _ in range(num_targets):
            obj_id += 1
            regions.append({
                "name": f"obj_{obj_id}",
                "isStuff": False,
                "occlude_rate": 0.35,
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                "visible_mask": {"counts": "dummy", "size": [321, 481]},
            })

        # Add 2 background stuff regions (should be filtered)
        regions.append({
            "name": "sky",
            "isStuff": True,
            "occlude_rate": 0.0,
            "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]],
        })

        # Add 1 unoccluded object (should be filtered)
        regions.append({
            "name": "unoccluded_fg",
            "isStuff": False,
            "occlude_rate": 0.0,
            "segmentation": [[60, 60, 80, 60, 80, 80, 60, 80]],
        })

        # Canonical annotator 1
        annotations.append({
            "image_id": i,
            "author": "annotator_1",
            "regions": regions,
        })

        # Duplicate annotator 2 (should be skipped by first_annotator_only)
        annotations.append({
            "image_id": i,
            "author": "annotator_2",
            "regions": regions,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"images": images, "annotations": annotations}, f, indent=2)

    print(f"[Test Mode] Saved synthetic JSON with 200 images, 2 annotators per image, to: {output_path}")
    verify_bsds_dataset(output_path)


def main():
    parser = argparse.ArgumentParser(description="Dry-run BSDS-A benchmark target count parser")
    parser.add_argument(
        "--ann_file",
        type=str,
        default=None,
        help="Path to BSDS_amodal_test.json (auto-detected if omitted)",
    )
    parser.add_argument(
        "--include_stuff",
        dest="ignore_stuff",
        action="store_false",
        help="Include background 'stuff' regions (default: False, stuff regions are excluded)",
    )
    parser.set_defaults(ignore_stuff=True)
    parser.add_argument(
        "--no_filter_occluded",
        dest="filter_occluded",
        action="store_false",
        help="Include unoccluded objects (default: False, unoccluded objects are excluded)",
    )
    parser.set_defaults(filter_occluded=True)
    parser.add_argument(
        "--min_occlude_rate",
        type=float,
        default=0.141,
        help="Minimum occlusion rate threshold (default: 0.141 for benchmark 650)",
    )
    parser.add_argument(
        "--benchmark_650",
        dest="benchmark_650",
        action="store_true",
        default=True,
        help="Enable exact CVPR benchmark 650-object standard (sets min_occlude_rate=0.141, first_annotator_only=True, default: True)",
    )
    parser.add_argument(
        "--no_benchmark_650",
        dest="benchmark_650",
        action="store_false",
        help="Disable benchmark 650 standard and evaluate all occluded instances",
    )
    parser.add_argument(
        "--all_annotators",
        dest="first_annotator_only",
        action="store_false",
        help="Evaluate all annotators instead of 1 canonical annotator per image (default: False)",
    )
    parser.set_defaults(first_annotator_only=True)
    parser.add_argument(
        "--target_author",
        type=str,
        default=None,
        help="Filter for a specific annotator/author ID (default: None)",
    )
    parser.add_argument(
        "--test_synthetic",
        action="store_true",
        help="Run verification on synthetic 650-object test data to verify parser logic",
    )

    args = parser.parse_args()

    if args.test_synthetic:
        run_synthetic_test()
        return

    min_occ = 0.141 if args.benchmark_650 else args.min_occlude_rate
    first_ann = True if args.benchmark_650 else args.first_annotator_only

    ann_file = args.ann_file or find_default_annotation_file()
    if not ann_file:
        print("\n" + "=" * 65)
        print("[NOTICE] BSDS_amodal_test.json not found in local workspace.")
        print("To verify on synthetic benchmark test data without downloading, run:")
        print("  python tools/verify_bsds_count.py --test_synthetic")
        print("When official annotations are downloaded, run:")
        print("  python tools/verify_bsds_count.py --ann_file path/to/BSDS_amodal_test.json")
        print("=" * 65 + "\n")
        sys.exit(0)

    verify_bsds_dataset(
        ann_file=ann_file,
        ignore_stuff=args.ignore_stuff,
        filter_occluded=args.filter_occluded,
        min_occlude_rate=min_occ,
        first_annotator_only=first_ann,
        target_author=args.target_author,
    )


if __name__ == "__main__":
    main()
