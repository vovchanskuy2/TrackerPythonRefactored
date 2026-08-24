TrackerPythonRefactored — a real-time motion capture system

This project captures human movements from a webcam and transmits them as quaternions (bone rotations) and blendshape values (facial expressions) via UDP. It uses MediaPipe to recognize posture, hands, and face in a single frame.

Features
- 🧍 **Posture** (33 skeleton points)
- ✋ **Hands** (21 points each) 
- 😊 **Face** (~30 key points for facial expressions) 
- 📦 **UDP transmission** in JSON format (port 5005) 
- ⚡ **Real-time** (~30 FPS) 
- 🔄 **Automatic recalculation** to the Unreal Engine coordinate system 
- 🧪 **Debug** – saving the last packet to `debug_udp.json`

Requirements
- Python 3.8+

Installation
- git clone https://github.com/vovchanskuy2/TrackerPythonRefactored
- cd TrackerPythonRefactored
- pip install -r requirements.txt

Launch

python main.py    
A window with the video will open, and data capture and transmission will begin.

Using    
The sending is done to 127.0.0.1:5005 (you can change this in the code).

Each frame sends a JSON of the following form:

json
{
 "timestamp": 123456789.123,
 "skeleton": {
 "Hips": { "x": 0.1, "y": 0.2, "z": 0.3, "w": 1.0 },
 ...
 },
 "face": [0.5, 0.0, ...]
}    

Customization    
Change the IP/port in the UDP_IP and UDP_PORT variables in main.py.
