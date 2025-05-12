# defense/headpose.py

import cv2
import numpy as np

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

pose_camera_matrix = None
pose_dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

def estimate_head_pose(frame, face_landmarks, w, h):
    global pose_camera_matrix
    image_points = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),     # Nose tip
        (face_landmarks[152].x * w, face_landmarks[152].y * h), # Chin
        (face_landmarks[33].x * w, face_landmarks[33].y * h),   # Left eye
        (face_landmarks[263].x * w, face_landmarks[263].y * h), # Right eye
        (face_landmarks[61].x * w, face_landmarks[61].y * h),   # Left mouth
        (face_landmarks[291].x * w, face_landmarks[291].y * h)  # Right mouth
    ], dtype=np.float64)

    # Draw pose landmark dots
    for idx in [1, 152, 33, 263, 61, 291]:
        x = int(face_landmarks[idx].x * w)
        y = int(face_landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # Green dots

    if pose_camera_matrix is None:
        focal_length = w
        center = (w / 2, h / 2)
        pose_camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, pose_camera_matrix, pose_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    def normalize_angle(angle):
        return (angle + 180) % 360 - 180

    roll, pitch, yaw = [normalize_angle(angle[0]) for angle in euler_angles]
    return pitch, yaw, roll, rotation_vector, translation_vector, pose_camera_matrix, pose_dist_coeffs

def track_nose_movement(face_landmarks, frame, w, h, prev_nose, movement_threshold, still_frame_count, still_frame_limit, log_path, head_alert_triggered, rvec, tvec, camera_matrix, dist_coeffs):
    nose_landmark = face_landmarks[1]
    nose_x = int(nose_landmark.x * w)
    nose_y = int(nose_landmark.y * h)
    cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 0), -1)

    # Draw nose line for head orientation
    nose_start = np.array([[face_landmarks[1].x * w, face_landmarks[1].y * h]], dtype=np.float64)
    nose_end_3D = np.array([[0, 0, 300.0]], dtype=np.float64)
    nose_end_2D, _ = cv2.projectPoints(nose_end_3D, rvec, tvec, camera_matrix, dist_coeffs)

    x, y = nose_end_2D[0].ravel()
    if 0 <= x < w and 0 <= y < h:
        p2 = (int(x), int(y))
    p1 = tuple(nose_start.ravel().astype(int))
    p2 = tuple(nose_end_2D[0].ravel().astype(int))
    cv2.line(frame, p1, p2, (0, 0, 255), 2)  # Red line from nose

    if prev_nose is not None:
        dx = abs(nose_x - prev_nose[0])
        dy = abs(nose_y - prev_nose[1])
        movement = (dx**2 + dy**2) ** 0.5

        if movement < movement_threshold:
            still_frame_count += 1
        else:
            still_frame_count = 0

    return frame, (nose_x, nose_y), still_frame_count, head_alert_triggered
