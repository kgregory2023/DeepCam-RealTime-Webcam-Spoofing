# DeepFake Webcam Spoofing Setup (DeepFaceLive + OBS)

## Tools Required

- [DeepFaceLive](https://github.com/iperov/DeepFaceLive)
- OBS Studio (for virtual webcam output)
- Zoom, Discord, or similar app for testing spoofed webcam

---

## Setup Steps

1. **Install DeepFaceLive**
   - Clone the repo:
     ```bash
     git clone https://github.com/iperov/DeepFaceLive.git
     ```
   - Follow their install guide for your OS (Windows is most supported).

2. **Download a Face Model**
   - Use a pretrained model or create one from real footage.
   - Load model into DeepFaceLive GUI.

3. **Install OBS Studio**
   - https://obsproject.com/
   - Enable Virtual Camera from the settings.

4. **Route DeepFaceLive Output**
   - Set DeepFaceLive to output to OBS’s virtual webcam.
   - Start OBS Virtual Camera.
   - Use it in Zoom or any video call app.

5. **Record Demo**
   - Record both real and spoofed sessions for comparison.

---

## Tips

- Dim lighting improves success rate.
- Run comparisons side-by-side for better effect.
- Do not use for deception or social engineering — educational purpose only.
