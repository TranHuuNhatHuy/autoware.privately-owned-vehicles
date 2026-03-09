# VisionPilot - Open AD Kit Demos

VisionPilot demos with Open AD Kit containers.

## Prerequisites

**Docker** - You will find an example video in the Test folder called ['traffic-driving.mp4'](https://drive.google.com/file/d/1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3/view?usp=drive_link) which is the default video that will be used as input for the demos. You can add other test videos to the Test folder if you wish, however, you would need to update the mount path in each AI model's launch script to point to any additional videos.

### Building the Docker image from scratch

The **visionpilot** container image is automatically pulled from [GHCR](https://github.com/orgs/autowarefoundation/packages/container/package/visionpilot) when running demos. To build it locally instead, **run from the project root**:

```bash
# Build for x64 with ONNX Runtime 1.22.0
docker build -t visionpilot -f VisionPilot/SDV/OpenADKit/Docker/Dockerfile . --build-arg ARCH=x64 --build-arg ONNXRUNTIME_VERSION=1.22.0

# Build for ARM64 with ONNX Runtime 1.22.0
docker build -t visionpilot -f VisionPilot/SDV/OpenADKit/Docker/Dockerfile . --build-arg ARCH=arm64 --build-arg ONNXRUNTIME_VERSION=1.22.0
```
