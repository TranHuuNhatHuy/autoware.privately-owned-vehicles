import cv2
import numpy as np

"""

This script finds the homography matrix to transform a cropped image of a road scene
into a bird eye view (BEV) grid. The homography is computed using manually defined source
and destination points.

Standard frame:
- Location: VisionPilot/Production_Releases/0.5/scripts/assets/standard_frame.jpg
- Resolution: 2880 x 1860
- Cropped resolution: 2880 x 1440
- BEV grid resolution: 640 x 640

This output BEV grid that can be used for lane detection and other applications.

"""


class BEVHomography:
    def __init__(self):
        """
        Initialize the BEVHomography class by loading the standard frame and defining
        source and destination points.
        """

        # Load standard frame image
        self.standard_frame = cv2.imread(
            "./assets/standard_frame.jpg"
        )

        # Crop upper part of the image to get 2880 x 1440
        CROPPED_H = 1860 - 1440
        self.standard_frame = self.standard_frame[CROPPED_H : , :]

        # MANUALLY DEFINED SOURCE POINTS
        # Define manually 4 source points by picking the best frame where car 
        # is perfectly straight (so egoleft/right make perfect trapezoid)
        # Four source points will be in the order: 
        # - Left bottom
        # - Right bottom
        # - Left top
        # - Right top
        # Supported by GIMP to get pixel coordinates
        RAW_SRC_POINTS = [
            (448  , 1859),  # Left bottom
            (2108 , 1859),  # Right bottom
            (1380 , 1049),  # Left top
            (1488 , 1049)   # Right top
        ]

        # Source points (normalized)
        self.raw_h, self.raw_w = self.standard_frame.shape[:2]
        self.src_points = [
            (
                x / self.raw_w, 
                y / self.raw_h
            ) 
            for x, y in RAW_SRC_POINTS
        ]

        # Destination points, for a BEV grid of 640 x 640
        # They are NOT normalized
        self.dst_points = [
            (159, 639),  # left bottom
            (479, 639),  # right bottom
            (159, 0),    # left top
            (479, 0)     # right top
        ]

    def compute_homography(self, cropped_image):
        """
        Compute the homography matrix from the cropped image to the BEV grid.
        """

        h, w = cropped_image.shape[:2]

        # Convert normalized source points to pixel coordinates
        src_pts_pixel = [(int(x * w), int(y * h)) for x, y in self.src_points]

        # Convert to numpy arrays
        src_pts_np = cv2.convertPointsToHomogeneous(np.array(src_pts_pixel)).reshape(-1, 2)
        dst_pts_np = cv2.convertPointsToHomogeneous(np.array(self.dst_points)).reshape(-1, 2)

        # Compute homography matrix
        homography_matrix, _ = cv2.findHomography(src_pts_np, dst_pts_np)

        return homography_matrix

    def warp_to_bev(self, cropped_image):
        homography_matrix = self.compute_homography(cropped_image)
        bev_image = cv2.warpPerspective(cropped_image, homography_matrix, (640, 640))
        return bev_image


# INPUT : cropped-only image (2880 x 1440)

# 1. Read raw image (2880 x 1860)

# 2. Crop raw image (2880 x 1440)

# 3. Find source points and destination points. Basically I can put a single perfect frame here and manually select the points.

# 4. Define manually 4 source points by picking the best frame where car is perfectly straight (so egoleft/right make perfect trapezoid)
# Four source points will be in the order: left bottom, right bottom, left top, right top
# Must be normalized coords (0, 1)

# 5. Set destination points
# From a BEV grid of 640 x 640
# Four destination points will be in the order: left bottom (159, 639), right bottom (479, 639), left top (159, 0), right top (479, 0)
# They are NOT normalized

# OUTPUT : BEV grid with straight lane markings (not picture, just the mask itself)