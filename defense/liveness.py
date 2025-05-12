from defense.blink import BlinkDetector
from defense.mouth import MouthTracker
from defense.headpose import estimate_head_pose, track_nose_movement
from defense.utils import log_spoof_alert, log_confidence_data, ensure_log_dirs
from defense.mouth import draw_mouth_landmarks

import time
import cv2
import mediapipe as mp

class LivenessDetector:
    def __init__(self, no_blink_threshold=10, log_dir="logs"):
        # Setup logging
        self.log_path, self.confidence_log_path = ensure_log_dirs(log_dir)
        self.last_confidence_log_time = time.time()
        self.conf_log_interval = 5  # seconds
        self.last_confidence_log_time = 0

        self.no_blink_threshold = no_blink_threshold
        self.blink = BlinkDetector()
        self.last_score = 100

        # Head movement tracking
        self.prev_nose = None
        self.movement_threshold = 2.5
        self.still_frame_limit = 30
        self.nose_still_frame_count = 0

        self.mar_alert_triggered = False
        self.pose_alert_triggered = False
        self.head_alert_triggered = False
        self.alert_triggered = False
        self.mouth_tracker = MouthTracker(self.log_path)

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame

        face_landmarks = results.multi_face_landmarks[0].landmark
        frame = draw_mouth_landmarks(frame, face_landmarks, w, h)

        # -- EAR/Blink --
        left_eye_idxs = [33, 160, 158, 133, 153, 144]
        right_eye_idxs = [362, 385, 387, 263, 373, 380]
        left_eye = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in left_eye_idxs]
        right_eye = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in right_eye_idxs]
        left_ear = self.blink.compute_ear(left_eye)
        right_ear = self.blink.compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        blinked = self.blink.update(avg_ear)

        if blinked:
            log_spoof_alert(self.log_path, "[BLINK] Detected")

        # -- MAR/Mouth --
        mouth_idxs = [78, 308, 13, 14, 82, 312]
        mouth = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in mouth_idxs]
        mar = self.mouth_tracker.compute_mar(mouth)
        frame, mar = self.mouth_tracker.detect_mouth_spoof(frame, mouth)
        if mar < 0.10 or mar > 0.35:
            cv2.putText(frame, "[!] Mouth Abnormality", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            if not self.mar_alert_triggered:
                log_spoof_alert(self.log_path, f"[MOUTH SPOOF] MAR: {mar:.3f}")
                self.mar_alert_triggered = True
        else:
            self.mar_alert_triggered = False

        # -- Head Pose & Nose Tracking --
        pitch, yaw, roll, rvec, tvec, cam_matrix, dist_coeffs = estimate_head_pose(frame, face_landmarks, w, h)
        frame, self.prev_nose, self.nose_still_frame_count, self.head_alert_triggered = track_nose_movement(
            face_landmarks, frame, w, h, 
            self.prev_nose, 
            self.movement_threshold, 
            self.nose_still_frame_count, 
            self.still_frame_limit, 
            self.log_path,
            self.head_alert_triggered,
            rvec, tvec, cam_matrix, dist_coeffs
        )

        pitch_threshold = 30
        yaw_threshold = 40
        if abs(yaw) > yaw_threshold or abs(pitch) > pitch_threshold:
            cv2.putText(frame, "[!] Head Angle Alert", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if not self.pose_alert_triggered:
                log_spoof_alert(self.log_path, f"[HEAD ANGLE] Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
                self.pose_alert_triggered = True
        else:
            self.pose_alert_triggered = False

        # -- Draw info
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {self.blink.blink_count}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elapsed = time.time() - self.blink.blink_timer
        bpm, variance = self.blink.get_bpm_and_variance()
        score = 100

        # Blink Rate Bonus
        if bpm >= 8:
            score += 10  # high activity bonus
        elif bpm < 3:
            score -= 40
        elif bpm < 5:
            score -= 20

        # Last Blink Timing
        if elapsed > self.no_blink_threshold:
            score -= 40
        
        # MAR
        if mar < 0.10 or mar > 0.35:
            score -= 15

        # Nose stillness
        if self.nose_still_frame_count >= self.still_frame_limit:
            score -= 20

        # Head pose
        if abs(yaw) > 50 or abs(pitch) > 45:
            score -= 10
        if abs(yaw) > 70 or abs(pitch) > 60:
            score -= 20

        # Blink Pattern Variance
        if variance < 0.2:
            score -= 30
        elif variance < 0.5:
            score -= 15
        elif variance > 1.5:
            score += 5  # irregular but likely human

        # Clamp the final score
        score = max(0, min(100, score))
        smoothed_score = int(0.6 * self.last_score + 0.4 * score)
        self.last_score = smoothed_score
        score = smoothed_score

        if elapsed > self.no_blink_threshold:
            score -= 40
            cv2.putText(frame, "No Blink Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if not self.alert_triggered:
                log_spoof_alert(self.log_path)
                self.alert_triggered = True
        else:
            cv2.putText(frame, f"Liveness OK ({int(self.no_blink_threshold - elapsed)}s left)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.alert_triggered = False

        cv2.putText(frame, f"BPM: {bpm}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {score}%", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        if time.time() - self.last_confidence_log_time >= self.conf_log_interval:
            log_confidence_data(self.confidence_log_path, bpm, variance, score)
            self.last_confidence_log_time = time.time()
        return frame
