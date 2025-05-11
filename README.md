# DeepCam: Real-Time Deepfake Webcam Spoofing & Defense Monitoring

DeepCam simulates a real-time AI deepfake attack using a spoofed webcam feed and provides defense mechanisms such as liveness detection and facial embedding verification.

---

## Project Goals

- Simulate real-time webcam spoofing using DeepFaceLive and virtual webcam tools
- Explore vulnerabilities in video-based verification systems (Zoom, Discord, etc.)
- Build real-time defenses: liveness detection, face embedding comparison, and anti-spoofing alerts
- Educate on emerging threats in biometric security and demonstrate ethical hacking principles

---

## Features

- 🔴 **Spoofing Pipeline** using DeepFaceLive and OBS VirtualCam
- 🔵 **Defense Toolkit** including:
  - Eye-blink and head movement detection
  - DeepFace-based face verification
  - Optional ML model for spoof classification
- Demo Video: Coming Soon
- Technical writeup: Spoofing vs. Detection Results

---

## Project Structure
DeepCam/
├── /spoofing/
│ ├── setup_deepfacelive.md
│ └── demo_video.mp4
├── /defense/
│ ├── liveness_detector.py
│ ├── face_verification.py
│ └── model/
│ └── anti_spoofing_model.pt
├── deepcam_demo.ipynb
├── README.md
├── requirements.txt

---

## Ethical Disclaimer

This project is for educational and ethical cybersecurity research only. Do **not** use this tool to deceive, impersonate, or cause harm. Always disclose your intent when testing on any system or service.

---

## Status

Repository initialized  
DeepFaceLive setup complete  
Defense modules in development  
Demo and writeup coming soon
