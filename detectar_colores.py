import cv2
import numpy as np

def get_hsv_values(image_path):
    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = hsv[y, x]
            print(f"HSV Value at ({x}, {y}): {pixel}")
    
    # Cargar la imagen
    image = cv2.imread('foto_bien.jpg')
    if image is None:
        print("No se pudo cargar la imagen.")
        return

    # Convertir a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mostrar la imagen para seleccionar puntos
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', pick_color)

    print("Haz clic en las áreas rojas y azules de la imagen para obtener los valores HSV.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen proporcionada
image_path = 'foto_bien.jpg'
get_hsv_values(image_path)
