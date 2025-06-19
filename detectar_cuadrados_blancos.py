import cv2
import numpy as np
import time  # Para agregar retrasos entre iteraciones

# Rango para color rojo (más amplio)
rojo_bajo = np.array([0, 100, 50])  # Ajuste más bajo
rojo_alto = np.array([10, 255, 255])  # Ajuste más alto

# Rango para color azul (más amplio)
azul_bajo = np.array([100, 100, 50])  # Ajuste más bajo
azul_alto = np.array([130, 255, 255])  # Ajuste más alto

# Inicializar captura de video
cap = cv2.VideoCapture(2)  # Cambia el índice si tienes más de una cámara

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Dimensión física conocida del QR (en cm)
qr_size = 5.0  # Tamaño físico del QR (lado)

# Inicializar QRCodeDetector
qr_detector = cv2.QRCodeDetector()

while True:
    # Leer fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el fotograma.")
        break

    # Redimensionar para mejor rendimiento
    frame = cv2.resize(frame, (640, 480))

    # Detectar y decodificar el QR
    data, bbox, _ = qr_detector.detectAndDecode(frame)
    qr_width = 0  # Inicializar la dimensión en píxeles
    if bbox is not None and data:
        # Dibujar el borde del QR en la imagen
        bbox = np.int32(bbox)  # Convertir los puntos a enteros
        for i in range(len(bbox[0])):
            pt1 = tuple(bbox[0][i])
            pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Calcular la dimensión del QR en píxeles
        qr_width = np.linalg.norm(bbox[0][0] - bbox[0][1])
        
        # Mostrar la información del QR
        cv2.putText(frame, f"QR Size: {qr_size} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"QR Data: {data}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convertir a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscaras para colores
    mask_red = cv2.inRange(hsv, rojo_bajo, rojo_alto)
    mask_blue = cv2.inRange(hsv, azul_bajo, azul_alto)

    # Detectar contornos para rojo y azul
    for mask, color_name, color in [(mask_red, "Rojo", (0, 0, 255)), (mask_blue, "Azul", (255, 0, 0))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filtrar objetos pequeños
                # Aproximar el contorno para detectar cuadrados
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4:  # Verificar si tiene 4 vértices
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.9 <= aspect_ratio <= 1.1:  # Verificar proporción (cercano a cuadrado)
                        if qr_width > 0:
                            # Calcular dimensiones físicas del cuadrado
                            square_width_cm = (w / qr_width) * qr_size
                            square_height_cm = (h / qr_width) * qr_size
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, f"{square_width_cm:.2f}x{square_height_cm:.2f} cm", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"{color_name} Cuadrado", (x, y - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar resultados
    cv2.imshow("Frame Original", frame)
    cv2.imshow("Rojo", mask_red)
    cv2.imshow("Azul", mask_blue)

    # Agregar un retraso de 0.5 segundos (ajustable) entre iteraciones
    time.sleep(0.5)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
