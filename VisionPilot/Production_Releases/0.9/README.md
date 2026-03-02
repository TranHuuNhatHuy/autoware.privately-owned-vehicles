# VisionPilot 0.9 – Lateral + Longitudinal Release

This release runs **lateral control** (EgoLanes + AutoSteer + PID) and **longitudinal tracking** (AutoSpeed + ObjectFinder + SpeedPlanner + longitudinal PID) in parallel, and publishes all outputs via POSIX shared memory for external consumers.

## 1. Build

From `Production_Releases/0.9`:

```bash
mkdir -p build
cd build
cmake ..      # ONNX Runtime + TensorRT (uses $ONNXRUNTIME_ROOT)
make -j$(nproc)
cd ..
```

Ensure:
- `ONNXRUNTIME_ROOT` points to your ONNX Runtime GPU install.
- TensorRT/CUDA are installed.

## 2. Configure (`visionpilot.conf`)

Edit `visionpilot.conf` in this directory:

- **Mode & source**
  - `mode=video` or `mode=camera`
  - `source.video.path=/path/to/video.mp4`
- **Models**
  - `models.egolanes.path=.../Egolanes_fp32.onnx`
  - `models.autosteer.path=.../AutoSteer_FP32.onnx`
  - `models.autospeed.path=.../AutoSpeed_n.onnx`
  - `models.homography_yaml.path=.../homography_2.yaml`
- **Timing**
  - `pipeline.target_fps=10.0`
- **Lateral PID**
  - `steering_control.Kp/Ki/Kd/Ks`
- **Longitudinal**
  - `longitudinal.autospeed.conf_thresh`
  - `longitudinal.autospeed.iou_thresh`
  - `longitudinal.ego_speed_default_ms` (used when CAN is disabled/invalid)
  - `longitudinal.pid.Kp/Ki/Kd`
- **CAN**
  - `can_interface.enabled=true/false`
  - `can_interface.interface_name=can0`

## 3. Run

```bash
./run_final.sh           # uses /usr/share/visionpilot/visionpilot.conf if present
./run_final.sh ./visionpilot.conf   # explicit config path
```

You should see:
- EgoLanes + AutoSteer lateral pipeline initialization
- AutoSpeed + ObjectFinder longitudinal initialization
- “Lateral and Longitudinal pipelines running in PARALLEL…”

## 4. Shared Memory Outputs

The process publishes a single shared-memory segment with all outputs:

- Name: `/visionpilot_state`
- Struct: `VisionPilotState` (see `include/publisher/visionpilot_shared_state.hpp`)
  - Lateral: steering angles, PathFinder CTE/yaw/curvature, lane departure flag
  - Longitudinal: CIPO distance/velocity, RSS safe distance, ideal speed, FCW/AEB flags, longitudinal control effort
  - CAN/ego: speed, steering angle, validity

### Quick test reader

From `0.9`:

```bash
./tools/shm_reader          # live view while visionpilot is running
./tools/shm_reader --once   # single snapshot
```

If VisionPilot is running correctly you will see frame IDs increasing and CIPO / steering values updating. When VisionPilot stops, `shm_reader` will show the last published frame until the segment is unlinked.
