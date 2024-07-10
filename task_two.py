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

# Инициализация счетчика найденных людей
people_count = 0

# Список для хранения центров описывающих рамок людей
centers = []

# Обработка результатов и рисование bounding box для людей
for result in results:
    for box in result.boxes.data.tolist():
        class_id = int(box[5])
        if class_id == 1:
            people_count += 1
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Вычисление центра описывающей рамки и добавление в список центров
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            centers.append(center)

# Определение групп людей
groups = []
visited = set()

for i in range(len(centers)):
    if i in visited:
        continue

    group = [centers[i]]
    visited.add(i)

    for j in range(i + 1, len(centers)):
        if j not in visited:
            dist = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
            if dist < 100:  # Порог расстояния для определения группы (менее 100 пикселей)
                group.append(centers[j])
                visited.add(j)

    if len(group) > 1:
        groups.append(group)

# Отрисовка описывающих прямоугольников для групп
for group in groups:
    if len(group) > 1:
        # Находим ограничивающий прямоугольник для группы
        x_min = min(pt[0] for pt in group)
        y_min = min(pt[1] for pt in group)
        x_max = max(pt[0] for pt in group)
        y_max = max(pt[1] for pt in group)

        cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        cv2.putText(img_cv2, f"Group of {len(group)} people", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.putText(img_cv2, f"Found {people_count} people", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Detected People and Groups", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
