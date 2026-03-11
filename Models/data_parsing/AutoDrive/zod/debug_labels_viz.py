#!/usr/bin/env python3
"""
Debug labels: no inference. Uses cipo_radar output (bbox, azimuth) + labels (distance, speed).
BEV shows radar points + one labeled dot (blue/yellow) at labeled distance/azimuth.
Verify: labeled dot should align with radar returns if labels are correct.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ZOD_ROOT = Path("/home/pranavdoma/Downloads/zod")


def radar_spherical_to_cartesian(pts):
    az = pts["azimuth_angle"].astype(np.float64)
    el = pts["elevation_angle"].astype(np.float64)
    rg = pts["radar_range"].astype(np.float64)
    x = rg * np.cos(el) * np.cos(az)
    y = rg * np.cos(el) * np.sin(az)
    z = rg * np.sin(el)
    return x, y, z


def draw_bev_with_labeled_dot(xy, labeled_x, labeled_y, scale=6, x_range=(0, 100), y_range=(-30, 30)):
    """BEV: gray radar points + one colored dot at (labeled_x, labeled_y)."""
    bev_h = int((x_range[1] - x_range[0]) * scale)
    bev_w = int((y_range[1] - y_range[0]) * scale)
    bev = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 28

    def to_pixel(x, y):
        row = int((x_range[1] - x) * scale)
        col = int((y_range[1] - y) * scale)
        return np.clip(row, 0, bev_h - 1), np.clip(col, 0, bev_w - 1)

    for x in range(0, 101, 20):
        r, c = to_pixel(x, y_range[0])
        cv2.line(bev, (c, r), (bev_w - 1, r), (55, 55, 55), 1)
        cv2.putText(bev, f"{x}m", (5, r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    for y in range(-30, 31, 10):
        r, c = to_pixel(x_range[0], y)
        cv2.line(bev, (c, 0), (c, bev_h - 1), (55, 55, 55), 1)

    # All radar points (gray)
    for i in range(len(xy)):
        r, c = to_pixel(xy[i, 0], xy[i, 1])
        cv2.circle(bev, (c, r), 2, (85, 85, 85), -1)

    # Labeled dot (yellow) - where label says the object is
    if labeled_x is not None and labeled_y is not None:
        r, c = to_pixel(labeled_x, labeled_y)
        cv2.circle(bev, (c, r), 3, (0, 255, 255), -1)
        cv2.circle(bev, (c, r), 4, (255, 255, 255), 1)

    # Ego
    r0, c0 = to_pixel(0, 0)
    cv2.circle(bev, (c0, r0), 8, (0, 255, 255), -1)
    cv2.circle(bev, (c0, r0), 10, (255, 255, 255), 2)

    cv2.putText(bev, "Radar BEV", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bev, "yellow = labeled position", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(bev, "gray = radar points", (5, bev_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return bev


def main():
    parser = argparse.ArgumentParser(description="Debug labels: bbox + labeled dot in BEV, no inference")
    parser.add_argument("--sequence", type=str, default="000330")
    parser.add_argument("--zod-root", type=str, default=str(ZOD_ROOT))
    parser.add_argument("--labels-dir", type=str, default=None)
    parser.add_argument("--cipo-radar", type=str, default=None, help="cipo_radar.json path (has bbox, azimuth)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zod = Path(args.zod_root)
    seq = args.sequence
    labels_dir = Path(args.labels_dir) if args.labels_dir else zod / "labels" / seq
    _ZOD_SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_ZOD_SCRIPT_DIR))
    from zod_utils import get_images_blur_dir
    img_dir = get_images_blur_dir(zod, seq)
    assoc_path = zod / "associations" / f"{seq}_associations.json"

    cipo_path = Path(args.cipo_radar) if args.cipo_radar else Path(__file__).parent / "output" / seq / "cipo_radar.json"
    if not cipo_path.exists():
        cipo_path = Path(__file__).parent / f"cipo_radar_{seq}.json"
    if not cipo_path.exists():
        print(f"cipo_radar not found: {cipo_path}. Run run_cipo_radar first.")
        return 1

    if not labels_dir.exists():
        print(f"Labels dir not found: {labels_dir}. Run full pipeline first.")
        return 1
    if not assoc_path.exists():
        print(f"Associations not found: {assoc_path}. Run step1 first.")
        return 1

    with open(cipo_path) as f:
        cipo_data = json.load(f)
    with open(assoc_path) as f:
        assoc = json.load(f)

    cipo_map = {r["image"]: r for r in cipo_data["results"]}
    img_to_rec = {r["image"]: r for r in assoc["associations"]}
    radar_data = np.load(assoc["radar_npy_path"], allow_pickle=True)

    # Filter to images that have CIPO in cipo_radar output
    cipo_images = [r["image"] for r in cipo_data["results"] if r.get("cipo_detected", True)]
    if not cipo_images:
        cipo_images = list(cipo_map.keys())

    label_files = list(labels_dir.glob("*.json"))
    if not label_files:
        print(f"No labels in {labels_dir}")
        return 1

    # Pick from images that have both label and cipo_radar entry
    valid_stems = [f.stem for f in label_files if f"{f.stem}.jpg" in cipo_map]
    if not valid_stems:
        valid_stems = [f.stem for f in label_files]

    random.seed(args.seed)
    n = min(args.n, len(valid_stems))
    selected_stems = random.sample(valid_stems, n)

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output" / seq / "debug_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for stem in selected_stems:
        img_name = f"{stem}.jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            print(f"Skip {stem}: image not found")
            continue

        rec = img_to_rec.get(img_name)
        if not rec:
            print(f"Skip {stem}: no association")
            continue

        cipo_rec = cipo_map.get(img_name, {})
        with open(labels_dir / f"{stem}.json") as f:
            label = json.load(f)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip {stem}: failed to load")
            continue

        # Draw bbox if CIPO detected
        if cipo_rec.get("cipo_detected") and "bbox" in cipo_rec:
            x1, y1, x2, y2 = [int(v) for v in cipo_rec["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "CIPO", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Label overlay
        curvature = label.get("curvature")
        dist = label.get("distance_to_in_path_object")
        speed = label.get("speed_of_in_path_object")
        curv_str = f"{curvature:.4f}" if curvature is not None and isinstance(curvature, (int, float)) else ("-" if curvature is None else str(curvature))

        lines = [
            "LABEL:",
            f"curvature: {curv_str}",
            f"distance: {dist} m" if dist is not None else "distance: -",
            f"speed: {speed} m/s" if speed is not None else "speed: -",
        ]
        cv2.rectangle(img, (0, 0), (img.shape[1], 110), (40, 40, 40), -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (20, 28 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"seq {seq}  {stem}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Radar points for BEV
        pts = radar_data[radar_data["timestamp"] == rec["radar_timestamp_ns"]]
        x, y, z = radar_spherical_to_cartesian(pts)
        mask = (z >= -0.5) & (z <= 1.0)
        xy = np.column_stack([x[mask], y[mask]])

        # Labeled dot position: (dist * cos(az), dist * sin(az)) in radar frame
        labeled_x, labeled_y = None, None
        az_deg = cipo_rec.get("azimuth_radar_deg")
        if dist is not None and az_deg is not None:
            az_rad = np.deg2rad(az_deg)
            labeled_x = dist * np.cos(az_rad)
            labeled_y = dist * np.sin(az_rad)

        bev = draw_bev_with_labeled_dot(xy, labeled_x, labeled_y)

        h, w = img.shape[:2]
        target_h = 640
        img_s = cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)
        bev_s = cv2.resize(bev, (img_s.shape[1] // 2, target_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.hstack([img_s, bev_s])

        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), canvas)
        n_saved += 1
        print(f"Saved {out_path.name}")

    print(f"\nDone: {n_saved} images -> {out_dir}")
    print("Verify: yellow dot (labeled position) should align with gray radar returns if labels are correct")
    return 0


if __name__ == "__main__":
    exit(main())
