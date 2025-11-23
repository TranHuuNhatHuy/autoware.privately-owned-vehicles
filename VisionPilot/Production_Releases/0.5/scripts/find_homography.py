import cv2


"""

This script finds the homography matrix to transform a cropped image of a road scene
into a bird eye view (BEV) grid. The homography is computed using manually defined source
and destination points.

Standard frame:
- Location: VisionPilot/Production_Releases/0.5/scripts/assets/standard_frame.jpg
- Resolution: 2880 x 1860
- Cropped resolution: 2880 x 1440
- BEV grid resolution: 640 x 640

"""


class BEVHomography:
    def __init__(self):

        # Load standard frame image
        self.standard_frame = cv2.imread(
            "./assets/standard_frame.jpg"
        )

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
        h, w = self.standard_frame.shape[:2]
        self.src_points = [
            (x / w, y / h) 
            for x, y in RAW_SRC_POINTS
        ]

        