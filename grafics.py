import pandas as pd
import matplotlib.pyplot as plt

# Caminho para o results.csv do YOLO
csv_path = "runs/detect/train2/results.csv"   # ALTERE se necessário

# Lendo CSV
df = pd.read_csv(csv_path)

# Lista de métricas comuns nos treinos YOLO
metricas = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)"
]

# Filtrar apenas métricas existentes no CSV
metricas = [m for m in metricas if m in df.columns]

# Criar gráficos
for metrica in metricas:
    plt.figure()
    plt.plot(df["epoch"], df[metrica])
    plt.xlabel("Época")
    plt.ylabel(metrica)
    plt.title(f"YOLO Treinamento - {metrica}")
    plt.grid(True)
    plt.show()
