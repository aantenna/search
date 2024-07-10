from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN

model = YOLO('runs/detect/helmet_detection5/weights/best.pt')

image_path = 'pers.jpg'

pil_image = Image.open(image_path)
img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Прогнозирование на изображении
results = model(img_cv2)

# Инициализация списка центров и рамок обнаруженных людей
centers = []
bboxes = []

for result in results:
    for box in result.boxes.data.tolist():
        class_id = int(box[5])
        if class_id == 1:
            x1, y1, x2, y2 = map(int, box[:4])
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
            bboxes.append((x1, y1, x2, y2))
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)


if centers:
    db = DBSCAN(eps=100, min_samples=2).fit(centers)
    labels = db.labels_

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        group_points = [bboxes[idx] for idx, lbl in enumerate(labels) if lbl == label]
        group_count = len(group_points)

        # Определение координат описывающего прямоугольника
        x1_group = min([x1 for (x1, y1, x2, y2) in group_points])
        y1_group = min([y1 for (x1, y1, x2, y2) in group_points])
        x2_group = max([x2 for (x1, y1, x2, y2) in group_points])
        y2_group = max([y2 for (x1, y1, x2, y2) in group_points])

        cv2.rectangle(img_cv2, (x1_group, y1_group), (x2_group, y2_group), (0, 255, 0), 2)
        cv2.putText(img_cv2, f"Group: {group_count}", (x1_group, y1_group - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

text = f"Found {len(centers)} people"
cv2.putText(img_cv2, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Detected People and Groups", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
