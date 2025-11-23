import cv2
import numpy as np

from BEVHomography import BEVHomography

"""

VisionPilot 0.5 first release

This script processes a video file frame by frame to perform lane detection,
as well as curve parameters estimation (lane offset, yaw angle, curvature) using
a bird's-eye view (BEV) transformation.

"""


def crop_and_rescale(
        frame: np.ndarray,
        crop_margin: tuple[int, int, int, int],
        rescale_size: tuple[int, int],
):
    """
    Crop and rescale the input frame.

    Args:
        frame (np.ndarray): Input video frame.
        crop_margin (tuple[int, int, int, int]): Margins to crop (top, right, bottom, left).
        rescale_size (tuple[int, int]): Desired output size (width, height).
    Returns:
        np.ndarray: Cropped and rescaled frame.
    """

    top, right, bottom, left = crop_margin
    cropped = frame[
        top  : frame.shape[0] - bottom, 
        left : frame.shape[1] - right
    ]
    rescaled = cv2.resize(cropped, rescale_size)

    return rescaled


def main():

    # Read homography (should be computed once with findHomography)
    BEVHomography = BEVHomography()
    homomatrix    = BEVHomography.homography_matrix
    
    # Prep video
    video_filepath = "/mnt/Storage/Daihatsu/video_frames.avi"
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    try:
        while cap.isOpened():
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crop + rescale: 2880 × 1860 ---crop---> 2880 x 1440 ---rescale---> 640 x 320
            frame = crop_and_rescale(
                frame,
                crop_margin  = (420, 0, 0, 0),
                rescale_size = (640, 320)
            )

            # Run inference

            # Show raw binary mask (must be normalized so we can use homography)

            # Convert raw binary mask to BEV using homography

            # Process BEV to extract lane points

            # Show BEV masks (debugging purpose)

            # Process lane points to get curve parameters of the road (lane offset, yaw angle, curvature)

            # Show BEV vis with the curve parameters and sliding windows and basically everything that helps us debug

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF

            # Quit on Q or ESC
            if ((key == ord("q")) or (key == 27)):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()