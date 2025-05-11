import cv2
import mediapipe as mp
import time
import os

class LivenessDetector:
    def __init__(self, no_blink_threshold=10, log_dir="logs"):
        self.no_blink_threshold = no_blink_threshold
        self.blink_timer = time.time()
        self.alert_triggered = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "events.log")

    def log_spoof_alert(self):
        with open(self.log_path, "a") as f:
            f.write(f"[!] Spoof alert: No blink detected at {time.ctime()}\n")

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Just track if a face is detected for now (can expand to EAR blink tracking later)
            elapsed = time.time() - self.blink_timer

            if elapsed > self.no_blink_threshold and not self.alert_triggered:
                self.alert_triggered = True
                self.log_spoof_alert()
                cv2.putText(frame, "No Blink Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Liveness OK ({int(self.no_blink_threshold - elapsed)}s left)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Reset timer if no face
            self.blink_timer = time.time()
            self.alert_triggered = False

        return frame