from ultralytics import YOLO
import cv2
#import serial

model = YOLO("runs/detect/train2/weights/best.pt")
cap = cv2.VideoCapture(1)

#arduino = serial.Serial('COM5', 9600, timeout=1)
#arduino.reset_input_buffer()

CONF_THRESHOLD = 0.70  # nível mínimo de confiança

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a detecção
    results = model(frame)[0]

    count = 0  # contador de carros confiáveis
    annotated_frame = frame.copy()

    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Filtra: só "car" com confiança alta
        if label == "car" and conf >= CONF_THRESHOLD:

            count += 1  # conta apenas carros confiáveis

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
        f"Carros confiaveis: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Define o tempo pelo número de carros confiáveis
    if count == 1:
        time = 5
    elif count == 2:
        time = 10
    elif count == 3:
        time = 15
    elif count > 3:
        time = 20
    else:
        time = 4  # sem carros

    #arduino.write(f"{time}\n".encode())

    cv2.imshow("Deteccao de Carros Confiaveis", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
