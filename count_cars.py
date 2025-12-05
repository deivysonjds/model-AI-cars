from ultralytics import YOLO
from time import sleep
import cv2
import serial

model = YOLO("runs/detect/train2/weights/best.pt")
cap = cv2.VideoCapture(0)

arduino = serial.Serial('COM5', 9600, timeout=1)
sleep(2)
time_class: int = 1
CONF_THRESHOLD = 0.60  # nível mínimo de confiança

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a detecção
    results = model(frame)[0]

    count = 0
    annotated_frame = frame.copy()

    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Filtra só "car" com confiança alta
        if label == "car" and conf >= CONF_THRESHOLD:

            count += 1 

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Desenha a caixa
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                annotated_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    # Mostra a contagem
    cv2.putText(
        annotated_frame,
        f"Carros: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Define o tempo pelo número de carros
    if count == 2:
        time_class = 1
    elif count == 4:
        time_class = 2
    elif count == 6:
        time_class = 3
    elif count > 8:
        time_class = 4
    else:
        time_class = 1

    arduino.write(str(time_class).encode())

    cv2.imshow("Deteccao de Carros", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
