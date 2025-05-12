# defense/mouth.py

import cv2
import time
from scipy.spatial import distance as dist

class MouthTracker:
    def __init__(self, log_path):
        self.mar_alert_triggered = False
        self.log_path = log_path

    def compute_mar(self, mouth):
        A = dist.euclidean(mouth[2], mouth[3])  # top lip to bottom lip
        B = dist.euclidean(mouth[4], mouth[5])  # upper inner lip to lower inner lip
        C = dist.euclidean(mouth[0], mouth[1])  # horizontal width
        return (A + B) / (2.0 * C)

    def detect_mouth_spoof(self, frame, mouth_landmarks, mar_thresholds=(0.10, 0.35)):
        mar = self.compute_mar(mouth_landmarks)
        cv2.putText(frame, f"MAR: {mar:.3f}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        if mar < mar_thresholds[0] or mar > mar_thresholds[1]:
            cv2.putText(frame, "[!] Mouth Abnormality Detected", (50, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            if not self.mar_alert_triggered:
                with open(self.log_path, "a") as f:
                    f.write(f"[MOUTH SPOOF] MAR: {mar:.3f} at {time.ctime()}\n")
                self.mar_alert_triggered = True
        else:
            self.mar_alert_triggered = False

        return frame, mar