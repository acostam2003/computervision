# ORB ------------------------------------------------------------
import cv2  # Importa OpenCV

# Lee la imagen y luego la redimensiona al 50% de su tamaño original
image = cv2.imread('orb_museo.jpg')
image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Crea el detector ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

# Detecta puntos clave y calcula descriptores usando ORB
keypoints, descriptors = orb.detectAndCompute(image, None)

# Dibuja los puntos clave en la imagen redimensionada
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# Muestra la imagen con los puntos clave detectados
cv2.imshow('ORB Keypoints', image_with_keypoints)

# Espera a que se presione una tecla y luego cierra las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
