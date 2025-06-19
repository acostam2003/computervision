import cv2  # Importa OpenCV
sift = cv2.xfeatures2d.SIFT_create()

# Verifica si la versión de OpenCV tiene el módulo xfeatures2d disponible
if hasattr(cv2, 'xfeatures2d'):
    # Lee la imagen desde un archivo (en este caso, 'imagen.jpg')
    image = cv2.imread('orb_museo.jpg')
    
    # Verifica si la imagen se cargó correctamente
    if image is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    else:
        # Crea el detector SURF (Speeded-Up Robust Features) con un umbral de Hessian de 400
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

        # Detecta puntos clave y calcula descriptores en la imagen
        keypoints, descriptors = surf.detectAndCompute(image, None)

        # Dibuja los puntos clave en la imagen
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

        # Muestra la imagen con los puntos clave detectados
        cv2.imshow('SURF Keypoints', image_with_keypoints)

        # Espera a que se presione una tecla y luego cierra las ventanas
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Error: La versión de OpenCV instalada no incluye el módulo xfeatures2d. "
          "Instala 'opencv-contrib-python' para habilitar SURF.")
