## EgoLanes
EgoLanes is a neural network that processes raw image frames and performs real-time semantic segmentation of driving lanes in the image. It produces a three class segmentation output for the ego-left lane, the ego-right lane and all other lanes. It outputs lanes at 1/4 resolution of the input image size allowing for quick inference on low power embedded hardware. EgoLanes was trained with data from a variety of real-world datasets including TuSimple, OpenLane, CurveLanes, Jiqing, and ONCE3D Lane.

## Watch the explainer video
Please click the video link to play - [***Video link***](https://drive.google.com/file/d/1UxzTlYf8ief8PKG65k-X9hN2WbcUoi9q/view?usp=drive_link)

<img src="../../../Media/EgoLanes_GIF_2.gif" width="100%">

### Loss Function:

- Lane-level binary cross-entropy loss: This loss penalized the model in its individual class level predictions
- Edge preservation loss: This loss ensured the model was able to predict lane boundaries accurately

## EgoLanes model weights
### [Link to Download Pytorch Model Weights *.pth](https://drive.google.com/file/d/1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX/view?usp=sharing)
### [Link to Download ONNX FP32 Weights *.onnx](https://drive.google.com/file/d/1b4jAoH6363ggTgVU0b6URbFfcOL3-r1Q/view?usp=sharing)
