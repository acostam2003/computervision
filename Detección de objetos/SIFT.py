# SIFT ------------------------------------------------------------
import cv2  # Importa OpenCV

# Lee la imagen 'imagen.jpg'
image = cv2.imread('orb_museo.jpg')
image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Crea el detector SIFT (Scale-Invariant Feature Transform)
sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

# Detecta puntos clave y calcula descriptores usando SIFT
keypoints, descriptors = sift.detectAndCompute(image, None)

# Dibuja los puntos clave en la imagen
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# Muestra la imagen con los puntos clave detectados
cv2.imshow('SIFT Keypoints', image_with_keypoints)

# Espera a que se presione una tecla y luego cierra las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
