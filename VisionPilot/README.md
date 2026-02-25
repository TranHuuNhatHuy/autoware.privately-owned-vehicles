# Vision Pilot

Vision Pilot is a productionizable and safety certifiable fully open-source Level 2 autonomous driving system designed for integration with automotive OEMs and Tier-1 suppliers in series production vehicles. It utilizes a single front-facing RGB camera with a 50 horizontal-degree FoV to enable ADAS features and in-lane autopilot on highways. Vision Pilot is designed to run in real-time on embedded edge hardware which can support between 3 to 5 INT8 TOPS. 

![VisionPilot Diagram](../Media/VisionPilot.png)

### Production Releases

The Production Releases folder contains specific releases of Vision Pilot that enable production-level self-driving features.

### Middleware Recipes

Vision Pilot is released as a standalone C++ application without any middleware requirement, additionally, we provide multiple middleware recipes which show how Vision Pilot could be integrated with popular middlewares including IceOryx, ZENOH, and ROS2. Vision Pilot is deployable on both QNX as well as Linux Ubuntu (tested with 22.04). The Middleware recipes folder contians example implementations of Vision Pilot for different middlewares.


## Architecture

Vision Pilot utilizes a component-based End-to-End autonomous driving architecture, wherein the entire autonomous driving stack is learnable. Vision Pilot also supports safety-perception neural networks which act as safety guardrails for the primary End-to-End stack. The overall End-to-End autonomous driving model which powers Vision Pilot is called AutoDrive, which is comprised of two End-to-End networks called AutoSteer for autonomous steering, AutoSpeed for autonomous acceleration/braking and Safety Perception models to address long-tail edge-case scenarios.

![VisionPilot Diagram](../Media/AutoDrive_E2E_Model.png)

## AutoDrive Model

The AutoDrive model aims to utilize a shared backbone architecture with multiple stems and heads relying upon the same network weights. This helps improve model efficiency and forces the model to learn generalizable image features.

![VisionPilot Diagram](../Media/AutoDrive_E2E_Architecture.png)

**More information about the specific models as well as examples showing how to try out and train the individual models in AutoDrive can be found in the [Models](../Models/) folder.**