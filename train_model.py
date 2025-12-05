from ultralytics import YOLO
import time

model = YOLO("yolo11n.pt")

start = time.time()

model.train(
    data="dataset_yolo/dataset.yaml",
    epochs=100,           
    patience=20,          
    imgsz=640,            
)

end = time.time()

# Calcula tempo total
elapsed = end - start

# Converte para horas/minutos/segundos
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print(f"\n\nTempo total de treino: {hours}h {minutes}m {seconds}s")
