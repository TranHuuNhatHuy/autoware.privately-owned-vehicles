import os
import cv2
from PIL import Image
from argparse import ArgumentParser

from Models.inference.auto_speed_infer import AutoSpeedNetworkInfer

color_map = {           # BGR
    1: (0, 0, 255),     # Red
    2: (0, 255, 255),   # Yellow
    3: (255, 255, 0)    # Cyan
}


def make_visualization(prediction, input_image_filepath):

    img_cv = cv2.imread(input_image_filepath)
    for pred in prediction:
        x1, y1, x2, y2, conf, cls = pred

        # Pick color, fallback to white if unknown class
        color = color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # Uncomment this if wanna show classes
        # label = f"Class: {int(cls)} | Score: {conf:.2f}"
        # cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tran: let's not show imgs, instead saving em in batch.
    # cv2.imshow("Prediction Objects", img_cv)
    # cv2.waitKey(0)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def main():

    parser = ArgumentParser()

    parser.add_argument(
        "-p", 
        "--model_checkpoint_path", 
        dest = "model_checkpoint_path",
        help = "Path to Pytorch checkpoint file to load model dict"
    )
    parser.add_argument(
        "-i",
        "--input_image_dirpath",
        dest = "input_image_dirpath",
        help = "Path to input image directory which will be processed by AutoSpeed"
    )
    parser.add_argument(
        "-o",
        "--output_image_dirpath",
        dest = "output_image_dirpath",
        help = "Path to output image directory where visualizations will be saved",
        required = True
    )
    
    args = parser.parse_args()

    # Arranging I/O dirs
    input_image_dirpath = args.input_image_dirpath
    output_image_dirpath = args.output_image_dirpath
    if (not os.path.exists(output_image_dirpath)):
        os.makedirs(output_image_dirpath)

    # Model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = AutoSpeedNetworkInfer(model_checkpoint_path)

    # Process through input image dir
    for filename in sorted(os.listdir(input_image_dirpath)):
        if (filename.endswith((".png", ".jpg", ".jpeg"))):

            # Fetch image
            input_image_filepath = os.path.join(
                input_image_dirpath, filename
            )
            img_id = filename.split(".")[0].zfill(3)
            print(f"Reading Image: {input_image_filepath}")

            # Inference
            img = Image.open(input_image_filepath).convert("RGB")
            prediction = model.inference(img)

            # Visualization
            vis_image = make_visualization(
                prediction, 
                input_image_filepath
            )
            
            output_image_filepath = os.path.join(
                output_image_dirpath,
                f"{img_id}_data.png"
            )
            vis_image.save(output_image_filepath)

        else:
            print(f"Skipping non-image file: {filename}")
            continue


if __name__ == "__main__":
    main()