import cv2
import mediapipe as mp
import time
import os

from scipy.spatial import distance as dist
import numpy as np

class LivenessDetector:
    def __init__(self, no_blink_threshold=10, log_dir="logs"):
        self.no_blink_threshold = no_blink_threshold
        self.blink_timestamps = []
        self.blink_timer = time.time()
        self.alert_triggered = False
        self.blink_count = 0
        self.blink_frame_flag = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "events.log")

    def log_spoof_alert(self):
        with open(self.log_path, "a") as f:
            f.write(f"[!] Spoof alert: No blink detected at {time.ctime()}\n")

    def compute_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Just tracks if a face is detected for now (can expand to EAR blink tracking later)
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # Get eye landmark coordinates
            left_eye_idxs = [33, 160, 158, 133, 153, 144]
            right_eye_idxs = [362, 385, 387, 263, 373, 380]

            left_eye = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in left_eye_idxs]
            right_eye = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in right_eye_idxs]

            # Calculates EAR
            left_ear = self.compute_ear(left_eye)
            right_ear = self.compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (50, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # EAR threshold for blink detection
            EAR_THRESHOLD = 0.22

            if avg_ear < EAR_THRESHOLD:
                # Blink detected — reset the spoof timer
                self.blink_timer = time.time()
                self.alert_triggered = False

                if not self.blink_frame_flag:
                    self.blink_count += 1
                    self.blink_frame_flag = True
                    self.blink_timestamps.append(time.time())


                    with open(self.log_path, "a") as f:
                        f.write(f"[BLINK] Detected at {time.ctime()}\n")
                
            else:
                self.blink_frame_flag = False

            # BPM Calculations
            current_time = time.time()
            self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= 60]
            bpm = len(self.blink_timestamps)
            cv2.putText(frame, f"BPM: {bpm}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)        

            elapsed = time.time() - self.blink_timer
            remaining = int(self.no_blink_threshold - elapsed)

            if remaining <= 0:
                if not self.alert_triggered:
                    self.alert_triggered = True
                    self.log_spoof_alert()
                cv2.putText(frame, "No Blink Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif remaining > 0:
                self.alert_triggered = False
                cv2.putText(frame, f"Liveness OK {(remaining)}s left)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Resets timer if no face on source feed
            self.blink_timer = time.time()
            self.alert_triggered = False

        return frame