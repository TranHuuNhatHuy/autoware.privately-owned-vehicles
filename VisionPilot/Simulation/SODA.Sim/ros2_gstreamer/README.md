## SodaSim ROS 2 GStreamer Bridge

Standalone ROS 2 package that rebroadcasts the Soda.Sim ROS 2 camera topic (default: `/vehicle/camera`) into a GStreamer pipeline. Replace the example topic if your setup differs.

### Dependencies (Ubuntu)

Confirmed on Ubuntu 22.04 (Jammy).

```bash
sudo apt install \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  ros-humble-cv-bridge ros-humble-image-transport
```

### Build

```bash
source /opt/ros/humble/setup.bash
cd <SODASIM_REPO_ROOT>/ros2_gstreamer
colcon build --symlink-install --base-paths .
source install/setup.bash
```

### Run

Before running, source your ROS 2 environment (example for Humble):

```bash
source /opt/ros/humble/setup.bash
source <SODASIM_REPO_ROOT>/ros2_gstreamer/install/setup.bash
```

```bash
ros2 run sodasim_gstreamer image_to_gstreamer_node_exe
```

### Send to a specific viewer IP (UDP)

Replace the `udpsink host` with the viewer’s IP address:

```bash
ros2 run sodasim_gstreamer image_to_gstreamer_node_exe --ros-args \
  -p input_topic:=/vehicle/camera \
  -p pipeline:="appsrc name=ros_appsrc is-live=true format=time do-timestamp=true ! \
                videoconvert ! x264enc tune=zerolatency key-int-max=30 bitrate=4000 \
                speed-preset=veryfast ! rtph264pay pt=96 config-interval=1 ! \
                udpsink host=<VIEWER_IP> port=5600 sync=false"
```

### VLC validation with SDP

1. Create a simple SDP file (example: `test.sdp`):

```sdp
v=0
o=- 0 0 IN IP4 0.0.0.0
s=ROS Camera
c=IN IP4 0.0.0.0
t=0 0
m=video 5600 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
a=recvonly
```

2. Open it in VLC on the viewer machine:

```bash
vlc test.sdp
```

If you don’t see video, verify the network path and firewall allow UDP traffic on the chosen port.
