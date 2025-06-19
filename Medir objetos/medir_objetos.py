import cv2
import numpy as np

# Cargamos una imagen 
image = cv2.imread('7.jpeg')
image = cv2.resize(image,None,fx=0.5,fy=0.5)

# Convertimos la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Se aplica Gauss para desenfocar la imagen y reducir el ruido
blurred = cv2.GaussianBlur(gray, (7,7),0)

# Detecar los bordes en la imagen 
edges = cv2.Canny(blurred,50,150)

# Encontramos los contornos
contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Gereamos el parámetro de referencia (por ejemplo, un obketo de 1 cm conocido en la imagen
ref_length_pixels = None # Establecemos después de identificar el objeto de referencia
real_length = 5.0 # Longitud real en cm del objeto de referencia

for contour in contours: 
    # Ignorar contornos muy pequeños
    if cv2.contourArea(contour) < 100:
        continue

    # Calcular el rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(contour)

    # Dibujar el rectángulo en la imagen original
    cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)

    # Calcular la dimensión en píxeles del objeto
    object_length_pixels = max(x,h)

    # Configurar la longitud de referencia si aún no se ha hecho
    if ref_length_pixels is None:
        ref_length_pixels= object_length_pixels 
        scale = real_length / ref_length_pixels
        cv2.putText(image, "Referencia", (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
    else:
        # Convertir las dimensiones en pixeles a dimensiones reales
        object_length_real = object_length_pixels * scale
        cv2.putText(image, f"{object_length_real:.2f} cm", (x-40,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

# Mostrar la imagen con las medidas
cv2.imshow("Medición de Objeto", image)
cv2.waitKey(0)
cv2.destroyAllWindows