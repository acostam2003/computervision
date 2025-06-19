import cv2
import numpy as np

def apply_cartoon_effect(image):

    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicamos un desenfoque para reducir el ruido 
    gray_blur = cv2.medianBlur(gray,5)

    # Detectamos bordes utilizando un filtro de Laplaciano
    edges = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize = 5)

    # Invertimos los bordes detectados 
    edges_inv = cv2.bitwise_not(edges)

    # Umbralizamos para crear una máscara binaria
    _, mask = cv2.threshold(edges_inv, 90, 250, cv2.THRESH_BINARY)

    # Reducimos el número de colores en la imagen original
    color = cv2.bilateralFilter(image, 0, 250, 250)
    # Aplicamos un efecto bilateral para suavizar colores pero mantener bordes
    color = cv2.bitwise_and(color, color, mask=mask)
    return apply_cartoon_effect

# Captura de video de la cámara web
cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
     
    if not ret: 
        break

    # Aplicar el filtro de caricatura
    cartoon_Frame = apply_cartoon_effect(frame)

    # Mostrar la imagen original y la imagen con el filtro 
    cv2.imshow('Original', frame)
    cv2.imshow('Cartoon Effect', cartoon_Frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows

