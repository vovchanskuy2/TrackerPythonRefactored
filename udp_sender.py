# udp_sender.py
import socket
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class UDPSender:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port

    def send(self, data):
        """
        data:
        {
            "t": float,
            "bones": {
                name: [x,y,z,w]  # quaternion (MediaPipe space)
            },
            "curves": {
                name: float  # blendshape values
            }
        }
        """

        ue_bones = {}
        for name, quat in data.get("bones", {}).items():
            ue_bones[name] = [float(v) for v in self._convert_quat_to_ue(quat)]

        packet = {
            "t": data.get("t", 0.0),
            "bones": ue_bones,
            "curves": {k: float(v) for k, v in data.get("curves", {}).items()}
        }

        try:
            msg = json.dumps(packet, separators=(',', ':')).encode('utf-8')
            self.sock.sendto(msg, (self.ip, self.port))

            # DEBUG (отключишь потом)
            with open("debug_udp.json", "w") as f:
                f.write(json.dumps(packet))

        except Exception as e:
            logging.error(f"UDP send error: {e}")

    # ---------- CORE PART ----------

    def _convert_quat_to_ue(self, q):
        """
        Конвертация quaternion из MediaPipe в Unreal систему координат
        q = [x, y, z, w]
        """

        x, y, z, w = q

        # Перестановка осей
        ue_x = z
        ue_y = x
        ue_z = -y
        ue_w = w

        return [ue_x, ue_y, ue_z, ue_w]

    # --------------------------------

    def close(self):
        self.sock.close()