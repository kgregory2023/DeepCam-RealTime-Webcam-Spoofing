# defense/blink.py
import time
from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, threshold=0.24):
        self.threshold = threshold
        self.blink_count = 0
        self.blink_frame_flag = False
        self.blink_timestamps = []
        self.blink_timer = time.time()

    def compute_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def update(self, avg_ear):
        blink_detected = False
        if avg_ear < self.threshold:
            self.blink_timer = time.time()
            if not self.blink_frame_flag:
                self.blink_count += 1
                self.blink_frame_flag = True
                self.blink_timestamps.append(time.time())
                blink_detected = True
        else:
            self.blink_frame_flag = False

        return blink_detected

    def get_bpm_and_variance(self):
        current_time = time.time()
        self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= 60]
        bpm = len(self.blink_timestamps)
        if len(self.blink_timestamps) >= 6:
            intervals = [self.blink_timestamps[i+1] - self.blink_timestamps[i] for i in range(len(self.blink_timestamps)-1)]
            pattern_variance = sum([(x - sum(intervals)/len(intervals))**2 for x in intervals]) / len(intervals)
        else:
            pattern_variance = 1.0
        return bpm, pattern_variance
