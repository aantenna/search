from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

model = YOLO('runs/detect/helmet_detection5/weights/best.pt')

image_path = 'pers.jpg'

pil_image = Image.open(image_path)

img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Прогнозирование на изображении
results = model(img_cv2)

# Инициализация счетчиков найденных объектов
people_count = 0
helmet_count = 0

# Обработка результатов и подсчет объектов
for result in results:
    for box in result.boxes.data.tolist():
        class_id = int(box[5])

        if class_id == 1:
            people_count += 1
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)

        elif class_id == 0:
            helmet_count += 1
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Расчет количества людей без каски
people_without_helmet = people_count - helmet_count

text_people_without_helmet = f"People without helmet: {people_without_helmet}"
text_people_with_helmet = f"People with helmet: {helmet_count}"
cv2.putText(img_cv2, text_people_without_helmet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(img_cv2, text_people_with_helmet, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Detected Objects", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
