# main.py
import cv2

from defense.liveness import LivenessDetector

def main():
    defense = LivenessDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = defense.process_frame(frame)
        cv2.imshow("Liveness Detector", output)

        if cv2.waitKey(1) & 0xFF == 27: #esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
