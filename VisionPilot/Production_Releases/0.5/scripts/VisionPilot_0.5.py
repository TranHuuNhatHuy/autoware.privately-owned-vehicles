import os
import cv2
import numpy as np

from BEVHomography import BEVHomography
from model_components.inference import AutoSteerNetworkInfer

"""

VisionPilot 0.5 first release

This script processes a video file frame by frame to perform lane detection,
as well as curve parameters estimation (lane offset, yaw angle, curvature) using
a bird's-eye view (BEV) transformation.

"""

FRAME_W     = 2880
FRAME_H     = 1440
BEV_W       = 640
BEV_H       = 640


# ========================== PREPROCESSING HELPER FUNCS ========================== #

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


def upscale_prediction(
        prediction: np.ndarray,
        target_size: tuple[int, int]
):
    """
    Upscale the prediction to the target size.

    Args:
        prediction (np.ndarray): Input prediction array of shape (C, H, W).
        target_size (tuple[int, int]): Desired output size (width, height).

    Returns:
        np.ndarray: Upscaled prediction array of shape (C, target_height, target_width).
    """

    # Prep "canvas"
    C, H, W = prediction.shape
    target_w, target_h = target_size
    upscaled = np.zeros(
        (C, target_h, target_w), 
        dtype = prediction.dtype
    )

    # Fetch pred coords
    pred_coords = [
        np.where(prediction[c] > 0)
        for c in range(C)
    ]

    # Compute scale
    scale_x = target_w / W
    scale_y = target_h / H

    # Mark positive preds on upscaled
    for c, (ys, xs) in enumerate(pred_coords):
        if (ys.size == 0):
            continue

        xs_scaled = (xs.astype(np.float32) * scale_x).astype(np.int32)
        ys_scaled = (ys.astype(np.float32) * scale_y).astype(np.int32)
        xs_scaled = np.clip(xs_scaled, 0, target_w - 1)
        ys_scaled = np.clip(ys_scaled, 0, target_h - 1)

        upscaled[c, ys_scaled, xs_scaled] = 1

    return upscaled


# ========================== BEV PROCESSING FUNCS ========================== #


def fit_poly(
        x: np.ndarray, 
        y: np.ndarray
):
    """
    Fits a 2nd order polynomial x = Ay^2 + By + C

    Args:
        x (np.ndarray): x coordinates of lane points.
        y (np.ndarray): y coordinates of lane points.

    Returns:
        np.ndarray | None:  Polynomial coefficients [A, B, C] 
                            or None if insufficient points.
    """

    if (len(x) < 5): 
        return None
    
    return np.polyfit(y, x, 2)


def ego_line_match(binary_channel: np.ndarray):
    """
    Extract egoline points from binary channel and fit em.

    Args:
        binary_channel (np.ndarray): Binary mask of a single lane channel.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            - Polynomial coefficients [A, B, C] or None if fitting failed.
            - x coordinates of lane points or None if no points found.
            - y coordinates of lane points or None if no points found.
    """

    nonzero = binary_channel.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    if (len(nonzero_x) == 0): 
        return None, None, None

    fit = fit_poly(nonzero_x, nonzero_y)
    
    return fit, nonzero_x, nonzero_y


def sliding_window_multi(binary_channel: np.ndarray):
    """
    Histogram + sliding window, for other lines (3rd channel)

    Args:
        binary_channel (np.ndarray): Binary mask of a single lane channel.

    Returns:
        tuple[list[np.ndarray], np.ndarray]:
            - List of polynomial coefficients [A, B, C] for each detected lane line.
            - Visualization image with sliding windows and detected points.
    """

    out_img = np.dstack((
        binary_channel, 
        binary_channel, 
        binary_channel
    )) * 0
    histogram = np.sum(
        binary_channel[binary_channel.shape[0]//2 : , : ], 
        axis = 0
    )
    
    peaks = []
    threshold = 20
    # Find starting peaks
    for i in range(len(histogram)):
        if (histogram[i] > threshold):
            if (
                (len(peaks) == 0) or 
                (abs(i - peaks[-1]) > 50)
            ): 
                peaks.append(i)

    nwindows = 9
    window_height = int(binary_channel.shape[0] // nwindows)
    margin = 40
    min_pix = 10
    
    nonzero = binary_channel.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    all_fits = []

    for start_x in peaks:
        current_x = start_x
        lane_inds = []
        for window in range(nwindows):
            win_y_low = binary_channel.shape[0] - (window+1) * window_height
            win_y_high = binary_channel.shape[0] - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin
            
            # Visualizing windows
            cv2.rectangle(
                out_img,
                (win_x_low,win_y_low),
                (win_x_high,win_y_high),
                (0, 255, 0), 
                1
            ) 
            
            good_inds = (
                (nonzero_y >= win_y_low) & 
                (nonzero_y < win_y_high) & 
                (nonzero_x >= win_x_low) & 
                (nonzero_x < win_x_high)
            ).nonzero()[0]
            lane_inds.append(good_inds)
            
            if (len(good_inds) > min_pix):
                current_x = int(np.mean(nonzero_x[good_inds]))
        
        lane_inds = np.concatenate(lane_inds)
        if (len(lane_inds) > 50): 
            x = nonzero_x[lane_inds]
            y = nonzero_y[lane_inds]
            fit = fit_poly(x, y)

            if fit is not None:
                all_fits.append(fit)
                out_img[y, x] = [0, 0, 255]

    return all_fits, out_img


def get_pixel_params(
        left_fit, 
        right_fit
):
    """
    Calculates curvature and offset in PIXELS.

    Args:
        left_fit (tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]): 
            Polynomial coefficients [A, B, C] for left lane or None.
        right_fit (tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]): 
            Polynomial coefficients [A, B, C] for right lane or None.

    Returns:
        tuple[tuple[float, float], float]:
            - Tuple of left and right curvature in pixels.
            - Offset from lane center in pixels (positive = car is to the right of center).
    """

    y_eval = BEV_H      # Evaluate at the bottom of the BEV image
    
    left_curverad = 0
    right_curverad = 0
    offset_px = 0
    
    # Radius of Curvature: R = ((1 + (2Ay + B)^2)^1.5) / |2A|
    if (left_fit is not None):
        A = left_fit[0]
        B = left_fit[1]
        left_curve_rad = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)

    if (right_fit is not None):
        A = right_fit[0]
        B = right_fit[1]
        right_curve_rad = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)
        
    # Offset
    if (
        (left_fit is not None) and 
        (right_fit is not None)
    ):
        left_x  = left_fit[0]*y_eval**2  + left_fit[1]*y_eval  + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        
        lane_center = (left_x + right_x) / 2
        img_center = BEV_W / 2
        
        # Positive = Car is to the right of center (Need to steer Left)
        offset_px = lane_center - img_center
        
    return (left_curverad, right_curverad), offset_px


# ========================== MAIN PROCESSING LOOP ========================== #


def main():

    # # Japanese highway tends to have lane width 3.5 meters
    # # In BEV of standard frame, this lane is around 130 pixels
    # XM_PER_PIX  = 3.5 / 130     # Meters per pixel in x dimension

    # # Need confirmation from the customer here
    # YM_PER_PIX  = 30  / 640     # Meters per pixel in y dimension

    # Read homography (should be computed once with findHomography)
    bev_homography = BEVHomography()
    
    # Prep video
    video_filepath = "/mnt/Storage/Daihatsu/video_frames_trimmed.avi"
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # Prep inference
    model_checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "assets/Best_Egolanes.pth"
    )
    model = AutoSteerNetworkInfer(
        checkpoint_path = model_checkpoint_path
    )

    try:
        while cap.isOpened():
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Crop + rescale: 2880 × 1860 ---crop---> 2880 x 1440 ---rescale---> 640 x 320
            frame = crop_and_rescale(
                frame,
                crop_margin  = (420, 0, 0, 0),
                rescale_size = (640, 320)
            )

            # Run inference
            prediction = model.inference(frame)

            # Get raw binary mask (must be [0 - 1] normalized so we can use homography)
            # Should be same size as cropped frame (2880 x 1440)
            binary_mask = np.moveaxis(
                upscale_prediction(
                    prediction,
                    target_size = (FRAME_W, FRAME_H)
                ),
                0, -1
            )

            # Convert raw binary mask to BEV using homography, all 3 channels
            bev_mask = bev_homography.warp_to_bev(
                binary_mask
            ) * 255.0

            # LANE DETECTION IN BEV SPACE

            # Split channels
            c_egoleft       = bev_mask[:, :, 0]
            c_egoright      = bev_mask[:, :, 1]
            c_otherlanes    = bev_mask[:, :, 2]

            # Fit 2 egolines
            fit_left, lx, ly    = ego_line_match(c_egoleft)
            fit_right, rx, ry   = ego_line_match(c_egoright)

            # Fit other lines with sliding window
            fit_others, viz     = sliding_window_multi(c_otherlanes)

            # Show BEV masks (debugging purpose)

            # Process lane points to get curve parameters of the road (lane offset, yaw angle, curvature)

            # Show BEV vis with the curve parameters and sliding windows and basically everything that helps us debug
            
            cv2.imshow("Frame", bev_mask)
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