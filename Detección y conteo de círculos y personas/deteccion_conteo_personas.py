import cv2
import numpy as np

# Cargamos la imagen
image = cv2.imread('7.jpeg',cv2.IMREAD_COLOR)
image = cv2.resize(image,None,fx=0.4,fy=0.4)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicamos desenfoque gaussiano para reducir el ruido y realizar una mejor detección
gray_blurred = cv2.GaussianBlur(gray,(9,9),2)

# Detectamos los círculos utilizando la transformada de Hough
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp =1.2, minDist=30,param1=100, param2=30,
                           minRadius=5,maxRadius=50)

# Si se detectan circulos, dibujarlo
if circles is not None: 
    circles = np.round(circles[0,:]).astype("int")

    for(x,y,r) in circles: 
        # Dibujamos el círculo
        cv2.circle(image, (x,y),r,(0,255,0),4)

        # Dibujamos el centro del círculo 
        cv2.circle(image, (x,y),2,(0,128,255),3)

# Obtener la dimensión del arreglo
print(circles)
dimensiones = circles.shape[0]
cv2.putText(image, f"La cantidad de circulos detectados es: {dimensiones}", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Mostramos la imagen con los circulos detectados
cv2.imshow("Círculos detectados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

