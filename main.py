# main.py

import cv2
from defense.liveness_detection import LivenessDetector

# Creates detector instance
detector = LivenessDetector(no_blink_threshold=10)

# Initializes webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run liveliness detection on frame
    processed = detector.process_frame(frame)
    cv2.imshow("Liveliness Detection", processed)

    if cv2.waitKey(1) & 0xFF == 27:  # Symbol being the ESC key
        break

cap.release()
cv2.destroyAllWindows()
