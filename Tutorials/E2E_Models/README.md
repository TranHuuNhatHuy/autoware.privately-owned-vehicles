# Tutorials for Autoware VisionPilot and its End-to-End Models

Autoware **VisionPilot** is a production-grade, open-source highway autonomy system designed to power SAE 
Level-4+ autonomous driving. This project provides a safety-certifiable software stack optimized for 
integration by automotive OEMs and Tier-1 suppliers into passenger vehicles.

Our key philosophies:
- **End-to-End (E2E) AI Architecture**: unlike traditional modular stacks that rely on hand-coded rules, our 
VisionPilot utilizes neural networks for core component technologies. The roadmap includes a transition from
the modular E2E to full monolithic and eventually Hybrid E2E AI for maximum safety and explainability.
- **Mapless Operation**: the system does not require high-definition (HD) 3D maps. It mimics human driving by 
relying on real-time scene perception and standard 2D navigational (sat-nav) maps.
- **Democratization**: all models, training pipelines, and weights are released under the Apache 2.0 license
to ensure the technology is accessible to both researchers and commercial developers.

Technology roadmap:

| Version | Autonomy Level | Target Domain | AI Approach |
|---|---|---|---|
| Vision Pilot | Level 2+ | Highways | Component-based E2E AI |
| Vision Pilot PRO | Level 2++ | Highway & Urban | Monolithic E2E AI |
| Vision Drive | Level 4+ | All Roads | Hybrid E2E AI |