import socket
import json
import logging
import time

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
            "left_hand": self._to_list_of_dicts(hands[0]) if hands and len(hands) > 0 and hands[0] else [],
            "right_hand": self._to_list_of_dicts(hands[1]) if hands and len(hands) > 1 and hands[1] else [],
            "face": self._to_list_of_dicts(face) if face else []
        }
        try:
            message = json.dumps(data).encode('utf-8')
            self.sock.sendto(message, (self.ip, self.port))
        except Exception as e:
            logging.error(f"Ошибка отправки UDP: {e}")
        time.sleep(0.1)
        #Check prints
        with open('sendedjson.json', 'w') as f:
            f.write(json.dumps(data))

    def _to_list_of_dicts(self, landmarks):
        if not landmarks:
            return []

        result = []

        # если landmarks = [ [ (x,y,z,v), ... ] ]
        # убираем лишний уровень
        if len(landmarks) == 1 and isinstance(landmarks[0], list):
            landmarks = landmarks[0]

        for lm in landmarks:
            # lm = (x,y,z,vis)
            result.append({
                "x": float(lm[0]),
                "y": float(lm[1]),
                "z": float(lm[2]),
                "vis": float(lm[3])
            })

        return result

        
    def close(self):
        self.sock.close()