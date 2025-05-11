## DeepCam: Real-Time Deepfake Spoof Detection System

1. Overview
DeepCam is a real-time biometric spoof detection system built to identify deepfake-based webcam attacks using behavioral biometrics. It leverages facial landmark tracking and blink detection to determine whether a video feed is genuine or spoofed.
This project is developed to explore the intersection of computer vision, biometric security, and adversarial deepfake defense.

2. System Architecture
Webcam Input
→ Face Detection & Landmark Tracking (MediaPipe)
→ EAR Calculation (Blink Detection)
→ Time Threshold Logic
→ Spoof Alert Trigger + Logging

Components:
· Video stream processor (OpenCV)
· Face mesh landmark detector (MediaPipe)
· EAR logic to track blinks
· Spoof logger that writes to a timestamped log file

3. Tech Stack
· Python
· OpenCV: Real-time video processing and drawing overlays
· MediaPipe: Facial mesh and eye landmark detection
· DeepFaceLive: To simulate spoofed webcam feeds for testing
· Time/OS modules: For spoof timer and log management

4. Implemented Features
· EAR-based blink detection logic
· Spoof alert triggered if no blink is detected for a configurable time (e.g., 10 seconds)
· Real-time console logging of spoof events

5. Detection Logic
· EAR (Eye Aspect Ratio) Calculation:
   Uses vertical and horizontal eye landmark distances
   Threshold for blink: EAR < 0.25
· Spoof Condition:
  If no blink is detected in 10+ seconds, spoof alert is triggered
  Logged with timestamp: [!] Spoof alert: No blink detected at TIME

6. Testing Process
· Simulated real faces with live webcam
· Spoofed input generated using DeepFaceLive (celebrity face overlay or alternate static face)
· Testing conducted using side-by-side comparison
· Manual verification of alert accuracy during both real and spoofed feeds

7. Limitations & Known Issues
· Only EAR-based detection currently implemented
· False positives possible (e.g., tired user or natural low blink rate)
· No head pose or mouth detection yet
· Scoring system is still basic (binary logic only)

9. Ethical Use Statement
All spoofed face inputs were created using DeepFaceLive. The use of the DeepFaceLive system is intended solely for research and educational purposes. 


