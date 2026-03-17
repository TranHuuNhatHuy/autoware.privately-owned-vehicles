## AutoSteer
AutoSteer is a neural network which aims to predict the steering angle of a vehicle in order to follow the overall shape of the road the vehicle is travelling along. This is a type of road-curvature estimation, and can be used to help autonomous vehicles navigate highly curved roads and roads with high bank angles, where traditional road curvature estimation methods fail.

AutoSteer takes as input the lane mask output from the EgoLanes network for the current and previous image. It outputs a probability vector which encodes fixed steering angles, and the argmax of this probability vector informs us of the steering angle predicted by the model. In practice, we add a moving average filter to the model output to ensure that the predicted steering values are smooth over time.

## Watch the explainer video
Please click the video link to play - [***Video link***](https://drive.google.com/file/d/1hYJss-xWskAktQg0qU722b8YyQSokq_d/view?usp=drive_link)


## AutoSteer model weights
### [Link to Download Pytorch Model Weights *.pth](https://drive.google.com/file/d/17yu0H81sFE6ZHuviT7SXH3iMjMmyyS0t/view?usp=sharing)
### [Link to Download ONNX FP32 Weights *.onnx](https://drive.google.com/file/d/1gxH6EM4HJ0rfnqt90cT1w7hgizW49jQe/view?usp=sharing)
