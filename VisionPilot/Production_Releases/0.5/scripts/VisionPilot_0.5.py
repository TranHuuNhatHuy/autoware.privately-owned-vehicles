import cv2

def main():

    video_filepath = "/mnt/Storage/Daihatsu/video_frames.avi"

    # Read homography (should be computed once with findHomography)
    
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    try:
        while cap.isOpened():
            
            # Read data
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF

            # Quit on Q or ESC
            if ((key == ord("q")) or (key == 27)):
                break

            # Crop: 2880 × 1860 ---> 2880 x 1440

            # Rescale---> 640 x 320

            # Run inference

            # Show raw binary mask (must be normalized so we can use homography)

            # Convert raw binary mask to BEV using homography

            # Process BEV to extract lane points

            # Show BEV masks (debugging purpose)

            # Process lane points to get curve parameters of the road (lane offset, yaw angle, curvature)

            # Show BEV vis with the curve parameters and sliding windows and basically everything that helps us debug

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()