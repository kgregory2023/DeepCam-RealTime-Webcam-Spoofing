import cv2
import mediapipe as mp
import time
import os

from scipy.spatial import distance as dist
import numpy as np

class LivenessDetector:
    def __init__(self, no_blink_threshold= 10, log_dir="logs"):
        # Blink tracking state
        self.no_blink_threshold = no_blink_threshold
        self.model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        self.blink_timestamps = []
        self.blink_timer = time.time()
        self.alert_triggered = False
        self.blink_count = 0
        self.blink_frame_flag = False

        # Nose tracking state
        self.prev_nose = None
        self.nose_still_frame_count = 0
        self.movement_threshold = 2.5  # Pixel movement sensitivity
        self.still_frame_limit = 30    # Frames of stillness before alert (~1 sec at 30 FPS)
        self.head_alert_triggered = False

        # Head pose tracking state
        self.pose_camera_matrix = None
        self.pose_dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "events.log")
        self.confidence_log_path = os.path.join(log_dir, "confidence.log")
        self.last_confidence_log_time = time.time()

    def log_spoof_alert(self):
        with open(self.log_path, "a") as f:
            f.write(f"[!] Spoof alert: No blink detected at {time.ctime()}\n")
    
    def log_confidence_data(self, bpm, pattern_variance, confidence):
        current_time = time.time()
        if current_time - self.last_confidence_log_time >= 5:
            with open(self.confidence_log_path, "a") as f:
                f.write(f"[CONFIDENCE] Time: {time.ctime()} | BPM: {bpm} | Blink Variance: {pattern_variance:.3f} | Score: {confidence}\n")
            self.last_confidence_log_time = current_time

    def compute_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def track_nose_movement(self, face_landmarks, frame, w, h): #method for tracking nose tracking
        nose_landmark = face_landmarks[1]
        nose_x = int(nose_landmark.x * w)
        nose_y = int(nose_landmark.y * h)
        cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 0), -1)

        if self.prev_nose is not None:
            dx = abs(nose_x - self.prev_nose[0])
            dy = abs(nose_y - self.prev_nose[1])
            movement = (dx**2 + dy**2) ** 0.5

            if movement < self.movement_threshold:
                self.nose_still_frame_count += 1
            else:
                self.nose_still_frame_count = 0

            if self.nose_still_frame_count >= self.still_frame_limit:
                cv2.putText(frame, "[!] Low Head Movement", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if not self.head_alert_triggered:
                    with open(self.log_path, "a") as f:
                        f.write(f"[HEAD STAGNANT] Low movement detected at {time.ctime()}\n")
                    self.head_alert_triggered = True
            else:
         
                self.head_alert_triggered = False  # Resets if movement resumes
                
        self.prev_nose = (nose_x, nose_y)
        return frame
    
    def estimate_head_pose(self, face_landmarks, w, h): #method for tracking head pose
        image_points = np.array([
            (face_landmarks[1].x * w, face_landmarks[1].y * h),     # Nose tip
            (face_landmarks[152].x * w, face_landmarks[152].y * h), # Chin
            (face_landmarks[33].x * w, face_landmarks[33].y * h),   # Left eye left corner
            (face_landmarks[263].x * w, face_landmarks[263].y * h), # Right eye right corner
            (face_landmarks[61].x * w, face_landmarks[61].y * h),   # Left mouth
            (face_landmarks[291].x * w, face_landmarks[291].y * h)  # Right mouth
        ], dtype=np.float64)

        if self.pose_camera_matrix is None:
            focal_length = w
            center = (w / 2, h / 2)
            self.pose_camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.pose_camera_matrix,
            self.pose_dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Converts the rotation vector to angle
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = [angle[0] for angle in euler_angles]
        return pitch, yaw, roll
        

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Tracks face and EAR blink tracking 
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            frame = self.track_nose_movement(face_landmarks, frame, w, h)

            # Tracks Head Pose
            pitch, yaw, roll = self.estimate_head_pose(face_landmarks, w, h)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
            EAR_THRESHOLD = 0.24

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

            # BPM(Blinks Per Minute) Calculations
            current_time = time.time()
            self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= 60]
            bpm = len(self.blink_timestamps)

            # Blink Variance
            if len(self.blink_timestamps) >= 6:
                intervals = np.diff(self.blink_timestamps[-6:])
                pattern_variance = np.var(intervals)
            else:
                pattern_variance = 1.0        

            elapsed = time.time() - self.blink_timer
            remaining = int(self.no_blink_threshold - elapsed)

            score = 100

            # Blink Rates
            if bpm < 3:
                score -= 40
            elif bpm < 5:
                score -= 20
            
            # Since Last Blink
            if elapsed > self.no_blink_threshold:
                score -= 40

            # Blink Randomness   
            if pattern_variance < 0.2:
                score -= 30
            elif pattern_variance < 0.5:
                score -= 15
                
            score = max(0, min(100, score))

            cv2.putText(frame, f"BPM: {bpm}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {score}%", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            self.log_confidence_data(bpm, pattern_variance, score)

            if remaining <= 0:
                if not self.alert_triggered:
                    self.alert_triggered = True
                    self.log_spoof_alert()
                cv2.putText(frame, "No Blink Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                self.alert_triggered = False
                cv2.putText(frame, f"Liveness OK ({remaining}s left)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Resets timer if no face on source feed
            cv2.putText(frame, f"No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.alert_triggered = False

        return frame