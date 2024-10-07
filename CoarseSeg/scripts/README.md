
## create_masks

Contains scripts to process open datasets and create semantic masks in a unified labelling scheme according to the CoarseSeg neural task specification. 

Open semantic segmentation datasets contain various labelling methodologies and semantic classes. The scripts in create_masks parse data and create semantic colormaps in a single unified semantic format.

Colormap values for unified semantic classes created from training data are as follows:

| Semantic Class             | RGB                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Sky | ![#3DB8FF](https://via.placeholder.com/10/3DB8FF?text=+) rgb(61, 184, 255)|
| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+)rgb(61, 93, 255)|
| Foreground Objects | ![#FF1C91](https://via.placeholder.com/10/FF1C91?text=+) rgb(255, 28, 145) |
| Vulnerable Living | ![#FF3D3D](https://via.placeholder.com/10/FF3D3D?text=+) rgb(255, 61, 61)|
| Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
| Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
| Road | ![#00FFDC](https://via.placeholder.com/10/00FFDC?text=+) rgb(0, 255, 220) |

#### The open datasets used in CoarseSeg include:
- [ACDC](https://acdc.vision.ee.ethz.ch/)
- [MUSES](https://muses.vision.ee.ethz.ch/)
- [IDDAW](https://iddaw.github.io/)
- [BDD100K](https://www.vis.xyz/bdd100k/)
- [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)
- [Comma10K](https://github.com/commaai/comma10k)

#### Please note: 
All of the open datasests besides Comma10K include semantic labels for specific foreground objects:
- `Vulnerable Living`
- `Small Mobile Vehicle`
- `Large Mobile Vehicle`

The Comma10K dataset however, only contains a single class label for `Foreground Objects` which includes all movable foreground elements such as pedestrians, vehicles, animals etc. 

Furthermore, the Comma10K dataset does not include labels for `Road Edge Delimeter`. 

Lastly, the Comma10K dataset does not natively provide `sky` class semantic pixel labels and a separate pre-trained neural network was used to create pixel level sky masks for the Comma10K images.

## pytorch

Contains scripts related to model training, inference and evaluation



