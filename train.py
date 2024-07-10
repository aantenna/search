from ultralytics import YOLO

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Можно заменить на yolov8n.pt, yolov8m.pt, yolov8l.pt в зависимости от необходимой модели

# Обучение модели на вашем наборе данных
model.train(
    data='data.yaml',  # Путь к вашему конфигурационному файлу данных
    epochs=50,         # Количество эпох
    batch=16,          # Размер батча
    imgsz=640,         # Размер изображений
    name='helmet_detection'  # Название проекта/модели
)