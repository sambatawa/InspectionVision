import os
import time
from datetime import datetime
import cv2
#from anomalyBlue import processBlueAnomaly
from anomalyBlue2 import processBlueAnomaly

WINDOW = "Stasiun 1"
WINDOW_RESULT = "Hasil Analisis"
baseDir = os.path.dirname(os.path.abspath(__file__))
outputDir = os.path.join(baseDir, "hasil")
INTERVAL_CAPTURE = 20
CAPTURE_DELAY = 3
ROI = (0.2, 0.15, 0.8, 0.8)


def saveCapture(annotated_frame, status, detections, capture_count):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = "stasiun1-"+ "-" + timestamp + ".jpg"
    outputPath = os.path.join(outputDir, filename)
    cv2.imwrite(outputPath, annotated_frame)

    print("Capture #", capture_count)
    print("Status:", status)
    print("Objek:", len(detections))
    print("Path:", outputPath)
    if capture_count == 1:
        cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_RESULT,
            annotated_frame.shape[1],
            annotated_frame.shape[0],
        )
    cv2.imshow(WINDOW_RESULT, annotated_frame)


def main():
    os.makedirs(outputDir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    last_capture_time = -INTERVAL_CAPTURE
    capture_count = 0
    fail_start_time = None
    print("Kamera akti")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        status, detections, _, annotated_frame = processBlueAnomaly(frame, roi=ROI)
        now = time.monotonic()
        if status == "FAIL":
            if fail_start_time is None:
                fail_start_time = now
            if (now - fail_start_time) >= CAPTURE_DELAY and (now - last_capture_time) >= INTERVAL_CAPTURE:
                capture_count += 1
                saveCapture(annotated_frame, status, detections, capture_count)
                last_capture_time = now
        else:
            fail_start_time = None

        cv2.imshow(WINDOW, annotated_frame)
        cv2.waitKey(1)

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
