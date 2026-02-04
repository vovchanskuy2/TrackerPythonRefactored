import cv2
import logging

logging.basicConfig(level=logging.INFO)

def get_capture(source=0, width=1280, height=720):
    """Подключает камеру."""
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error(f"Ошибка открытия камеры {source}. Попытка альтернативы.")
            cap = cv2.VideoCapture(1 if source == 0 else 0)
            if not cap.isOpened():
                raise ValueError("Камера не найдена.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logging.info(f"Камера {source} подключена с разрешением {width}x{height}.")
        return cap
    except Exception as e:
        logging.error(f"Ошибка при подключении камеры: {e}")
        raise