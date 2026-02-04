import socket
import json
import logging

logging.basicConfig(level=logging.INFO)

class UDPSender:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port
        logging.info(f"UDP sender готов: {ip}:{port}")

    def send_pose_data(self, pose, hands, face, timestamp):
        """
        pose: list of 33 dicts или list of tuples (x,y,z,vis)
        hands: list of 2 lists (left/right), каждая по 21
        face: list of 478 tuples
        """
        data = {
            "timestamp": timestamp,
            "pose": self._to_list_of_dicts(pose) if pose else [],
            "hands": [self._to_list_of_dicts(hand) for hand in hands if hand] if hands else [],
            "face": self._to_list_of_dicts(face) if face else []
        }
        try:
            message = json.dumps(data).encode('utf-8')
            self.sock.sendto(message, (self.ip, self.port))
        except Exception as e:
            logging.error(f"Ошибка отправки UDP: {e}")

    def _to_list_of_dicts(self, landmarks):
        if not landmarks:
            return []
        return [{"x": lm[0], "y": lm[1], "z": lm[2], "vis": lm[3]} for lm in landmarks]

    def close(self):
        self.sock.close()