from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

model = YOLO('runs/detect/helmet_detection5/weights/best.pt')

image_path = 'pers.jpg'

# Загрузка изображения с помощью PIL (для проверки и преобразования)
pil_image = Image.open(image_path)

img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Прогнозирование на изображении
results = model(img_cv2)

# Инициализация счетчика найденных людей
people_count = 0

# Обработка результатов и рисование bounding box для людей
for result in results:
    for box in result.boxes.data.tolist():
        class_id = int(box[5])
        if class_id == 1:
            people_count += 1
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)


text = f"Found {people_count} people"
cv2.putText(img_cv2, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv2.imshow("Detected People", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
