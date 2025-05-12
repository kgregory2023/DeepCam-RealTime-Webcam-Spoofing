import time
import os

def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def ensure_log_dirs(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    return (
        os.path.join(log_dir, "events.log"),
        os.path.join(log_dir, "confidence.log")
    )

def log_spoof_alert(log_path, message="[!] Spoof alert: No blink detected"):
    with open(log_path, "a") as f:
        f.write(f"{message} at {time.ctime()}\n")

def log_confidence_data(log_path, bpm, variance, score, jitter):
    with open(log_path, "a") as f:
        f.write(f"[CONFIDENCE] Time: {time.ctime()} | BPM: {bpm} | Blink Variance: {variance:.3f} | Jitter: {jitter:.5f} | Score: {score}\n")
