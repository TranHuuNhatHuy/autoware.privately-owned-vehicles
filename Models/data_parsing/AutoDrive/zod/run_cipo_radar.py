#!/usr/bin/env python3
"""
Run AutoSpeed on ZOD images, compute CIPO azimuth in camera frame,
transform to radar frame, associate with nearest radar cluster.
Output: distance (m), speed (m/s) per image.
"""

import json
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add repo root for Models import
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image

try:
    from Models.inference.auto_speed_infer_50deg import (
        AutoSpeed50Infer,
        center_crop_50deg_resize,
        pixel_to_h_angle_deg_50,
    )
except ImportError:
    from inference.auto_speed_infer_50deg import (
        AutoSpeed50Infer,
        center_crop_50deg_resize,
        pixel_to_h_angle_deg_50,
    )

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


ZOD_ROOT = Path("/home/pranavdoma/Downloads/zod")
MODEL_PATH = Path(__file__).resolve().parents[4] / "VisionPilot/ROS2/data/models/autodrive.pt"

_LAT_BUFFER_M = 0.5  # ±0.5m lateral buffer for CIPO-radar association and clustering

_ZOD_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_ZOD_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_ZOD_SCRIPT_DIR))
from zod_utils import get_images_blur_dir, get_calibration_path


def parse_image_timestamp(fname: Path) -> int:
    stem = fname.stem
    parts = stem.split("_")
    ts_str = "_".join(parts[2:])
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)




def radar_spherical_to_cartesian(pts):
    """Radar: X forward, Y left, Z up. ZOD angles in radians."""
    az = pts["azimuth_angle"].astype(np.float64)
    el = pts["elevation_angle"].astype(np.float64)
    rg = pts["radar_range"].astype(np.float64)
    x = rg * np.cos(el) * np.cos(az)
    y = rg * np.cos(el) * np.sin(az)
    z = rg * np.sin(el)
    return x, y, z


def pixel_to_h_angle_deg(u: float, W: float, H: float, hfov_deg: float) -> float:
    """
    Horizontal angle (deg) from optical axis.
    H_angle = ((u - W/2) / (W/2)) * (HFOV/2)
    """
    return ((u - W / 2) / (W / 2)) * (hfov_deg / 2)


def cam_dir_to_radar_azimuth(h_angle_deg: float, cam_ext: np.ndarray, radar_ext: np.ndarray) -> float:
    """
    Transform camera horizontal angle to radar azimuth (radians).
    Camera: X right, Y down, Z forward. H_angle from optical axis (Z).
    dir_cam = (sin(h), 0, cos(h)) in camera frame.
    Transform via extrinsics: dir_radar = R_radar^T @ R_cam @ dir_cam
    azimuth_radar = atan2(dir_radar[1], dir_radar[0])
    """
    h_rad = np.deg2rad(h_angle_deg)
    dir_cam = np.array([np.sin(h_rad), 0.0, np.cos(h_rad)])
    R_cam = np.array(cam_ext)[:3, :3]
    R_radar = np.array(radar_ext)[:3, :3]
    dir_world = R_cam @ dir_cam
    dir_radar = R_radar.T @ dir_world
    return float(np.arctan2(dir_radar[1], dir_radar[0]))


def _polar_vel_dist(a, b, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5):
    """Polar+velocity distance: range ~4m, lateral ~0.5m, velocity ~1.5 m/s. a,b are (range, azimuth, range_rate)."""
    dr = abs(a[0] - b[0])
    r_avg = (a[0] + b[0]) / 2
    daz = abs(np.angle(np.exp(1j * (a[1] - b[1]))))
    d_lateral = r_avg * abs(np.sin(daz)) if r_avg > 0 else 0.0
    dv = abs(a[2] - b[2])
    return np.sqrt((dr / range_scale) ** 2 + (d_lateral / lat_buffer) ** 2 + (dv / vel_scale) ** 2)


def get_radar_clusters(radar_data, ts_ns: int, z_min=-0.5, z_max=1.0, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5, min_samples=2):
    """Filter z (-0.5 to 1m: ground to car roof), cluster with DBSCAN in (range, azimuth, range_rate) using lateral buffer."""
    pts = radar_data[radar_data["timestamp"] == ts_ns]
    if len(pts) == 0:
        return []
    x, y, z = radar_spherical_to_cartesian(pts)
    mask = (z >= z_min) & (z <= z_max)
    pts_f = pts[mask]
    rg = pts_f["radar_range"].astype(np.float64)
    az = pts_f["azimuth_angle"].astype(np.float64)
    rr = pts_f["range_rate"].astype(np.float64)
    polar_vel = np.column_stack([rg, az, rr])
    if len(polar_vel) == 0 or DBSCAN is None:
        return []
    metric = lambda a, b: _polar_vel_dist(a, b, range_scale, lat_buffer, vel_scale)
    labels = DBSCAN(eps=1.0, min_samples=min_samples, metric=metric).fit(polar_vel).labels_
    clusters = []
    for lbl in set(labels):
        if lbl < 0:
            continue
        m = labels == lbl
        clusters.append({
            "azimuth": float(np.mean(pts_f["azimuth_angle"][m])),
            "range": float(np.mean(pts_f["radar_range"][m])),
            "range_rate": float(np.mean(pts_f["range_rate"][m])),
        })
    return clusters


def find_nearest_cluster_lateral(clusters, azimuth_radar: float, lat_buffer_m: float = 0.5):
    """
    Filter clusters within ±lat_buffer_m lateral distance from CIPO ray.
    Perpendicular distance = r * |sin(az_cluster - az_cipo)| <= lat_buffer_m.
    Among those, pick the one with minimum range (closest along the azimuth ray).
    """
    if not clusters:
        return None
    in_cone = []
    for c in clusters:
        daz = abs(np.angle(np.exp(1j * (c["azimuth"] - azimuth_radar))))
        d_lateral = c["range"] * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            in_cone.append(c)
    if not in_cone:
        return None
    return min(in_cone, key=lambda c: c["range"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="000479")
    parser.add_argument("--zod-root", type=str, default=str(ZOD_ROOT), help="ZOD dataset root")
    parser.add_argument("--output", type=str, default=None, help="Output path for cipo_radar.json (default: cipo_radar_{seq}.json)")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Path to AutoSpeed model (autodrive.pt)")
    args = parser.parse_args()
    seq = args.sequence
    zod = Path(args.zod_root)
    img_dir = get_images_blur_dir(zod, seq)
    calib_path = get_calibration_path(zod, seq)
    radar_dir = zod / "radar_front" / "sequences" / seq / "radar_front"

    if not img_dir.exists():
        print(f"Image dir not found: {img_dir}")
        return
    if not calib_path.exists():
        print(f"Calibration not found: {calib_path}")
        return

    # Associations
    assoc_path = zod / "associations" / f"{seq}_associations.json"
    if not assoc_path.exists():
        print(f"Run step1 first: python zod/scripts/step1_timestamp_association.py --sequence {seq}")
        return

    with open(assoc_path) as f:
        assoc = json.load(f)
    with open(calib_path) as f:
        calib = json.load(f)["FC"]
    W, H = calib["image_dimensions"][0], calib["image_dimensions"][1]
    hfov_deg = calib["field_of_view"][0]
    cam_ext = np.array(calib["extrinsics"])
    radar_ext = np.array(calib["radar_extrinsics"])

    radar_data = np.load(assoc["radar_npy_path"], allow_pickle=True)
    model = AutoSpeed50Infer(str(args.model_path))

    out_path = Path(args.output) if args.output else Path(__file__).parent / f"cipo_radar_{seq}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_saved = False

    # Pass 1: forward pass - cluster matches only (no forward tracking)
    results = []
    for rec in assoc["associations"]:
        img_path = img_dir / rec["image"]
        ts_ns = rec.get("image_timestamp_ns")
        if not img_path.exists():
            results.append({"image": rec["image"], "cipo_detected": False, "distance_m": None, "speed_ms": None})
            continue

        img = Image.open(img_path).convert("RGB")
        crop_img, crop_info = center_crop_50deg_resize(img, W, H, hfov_deg)
        sample_path = str(out_path.parent / "model_input_sample.png") if not sample_saved else None
        if sample_path:
            sample_saved = True
            print(f"Saved model input sample -> {sample_path}")
        preds = model.inference(crop_img, crop_info, W, H, save_sample_path=sample_path)

        # CIPO: Level 1 and 2 only (dangerous/most in path). Exclude Level 3 (cyan, less in-path).
        CIPO_CLASSES = (1, 2)
        cipo = [p for p in preds if int(p[5]) in CIPO_CLASSES]
        if not cipo:
            results.append({"image": rec["image"], "cipo_detected": False, "distance_m": None, "speed_ms": None})
            continue

        cipo.sort(key=lambda p: (p[1] + p[3]) / 2, reverse=True)
        x1, y1, x2, y2, conf, cls = cipo[0]
        u = (x1 + x2) / 2

        h_angle_deg = pixel_to_h_angle_deg_50(u, crop_info)
        az_radar = cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext)
        az_radar_deg = float(np.rad2deg(az_radar))

        clusters = get_radar_clusters(radar_data, rec["radar_timestamp_ns"], lat_buffer=_LAT_BUFFER_M)
        cluster = find_nearest_cluster_lateral(clusters, az_radar, lat_buffer_m=_LAT_BUFFER_M)

        if cluster is not None:
            D, V = cluster["range"], cluster["range_rate"]
            results.append({
                "image": rec["image"],
                "cipo_detected": True,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "azimuth_radar_deg": az_radar_deg,
                "distance_m": round(D, 2),
                "speed_ms": round(V, 2),
                "_ts_ns": ts_ns,
            })
        else:
            results.append({
                "image": rec["image"],
                "cipo_detected": True,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "azimuth_radar_deg": az_radar_deg,
                "distance_m": None,
                "speed_ms": None,
                "_ts_ns": ts_ns,
            })

    # Pass 2: backfill - look ahead AND behind for cluster, estimate if same object
    LOOKAHEAD = 4
    LOOKBEHIND = 4
    AZ_TOL_DEG = 4.0  # same object: azimuth within 4 deg
    MAX_GAP_S = 1.0   # max time gap (seconds)

    for i in range(len(results)):
        r = results[i]
        if not r.get("cipo_detected") or r.get("distance_m") is not None:
            continue
        ts_i = r.get("_ts_ns")
        az_i = r.get("azimuth_radar_deg")
        if ts_i is None or az_i is None:
            continue

        best_D, best_V, best_gap = None, None, float("inf")

        def check_match(rj, ts_j, az_j, dt_s, D_ref, V_ref, is_forward):
            """is_forward: ref is from past -> D_est = D_ref + V_ref*dt"""
            if ts_j is None or D_ref is None or V_ref is None:
                return None, None
            if dt_s <= 0 or dt_s > MAX_GAP_S:
                return None, None
            daz = abs(np.angle(np.exp(1j * np.deg2rad(az_i - az_j))))
            if np.rad2deg(daz) > AZ_TOL_DEG:
                return None, None
            if is_forward:
                D_est = D_ref + V_ref * dt_s
            else:
                D_est = D_ref - V_ref * dt_s
            if D_est <= 0:
                return None, None
            return D_est, V_ref

        # Look ahead (future frames): D(t) = D_future - V*dt
        for k in range(1, LOOKAHEAD + 1):
            j = i + k
            if j >= len(results):
                break
            rj = results[j]
            if not rj.get("cipo_detected") or rj.get("distance_m") is None:
                continue
            ts_j = rj.get("_ts_ns")
            az_j = rj.get("azimuth_radar_deg")
            dt_s = (ts_j - ts_i) / 1e9
            D_est, V_est = check_match(rj, ts_j, az_j, dt_s, rj["distance_m"], rj["speed_ms"], is_forward=False)
            if D_est is not None and k < best_gap:
                best_gap = k
                best_D = D_est
                best_V = V_est

        # Look behind (past frames): D(t) = D_past + V*dt
        for k in range(1, LOOKBEHIND + 1):
            j = i - k
            if j < 0:
                break
            rj = results[j]
            if not rj.get("cipo_detected") or rj.get("distance_m") is None:
                continue
            ts_j = rj.get("_ts_ns")
            az_j = rj.get("azimuth_radar_deg")
            dt_s = (ts_i - ts_j) / 1e9
            D_est, V_est = check_match(rj, ts_j, az_j, dt_s, rj["distance_m"], rj["speed_ms"], is_forward=True)
            if D_est is not None and k < best_gap:
                best_gap = k
                best_D = D_est
                best_V = V_est

        if best_D is not None:
            r["distance_m"] = round(best_D, 2)
            r["speed_ms"] = round(best_V, 2)

    # Remove internal _ts_ns from all results before output
    for r in results:
        r.pop("_ts_ns", None)

    with open(out_path, "w") as f:
        json.dump({"sequence": seq, "results": results}, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
