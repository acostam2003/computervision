# Importa la biblioteca OpenCV para procesamiento de imágenes y visión por computadora
import cv2  

# Carga el clasificador de detección de rostros preentrenado (cascada Haar)
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lee una imagen desde un archivo (en este caso, 'imagen.jpg')
image = cv2.imread('personas_4.jpg')

# Convierte la imagen a escala de grises, ya que la detección de rostros funciona mejor en imágenes en blanco y negro
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecta los rostros en la imagen gris usando el clasificador cargado anteriormente
# scaleFactor ajusta el tamaño de la imagen en cada escala y minNeighbors define el número mínimo de vecinos para ser considerado rostro
faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Itera sobre cada coordenada de los rostros detectados
for (x, y, w, h) in faces:
    # Dibuja un rectángulo alrededor de cada rostro detectado en la imagen original en color
    # (x, y) es la esquina superior izquierda, (x+w, y+h) es la esquina inferior derecha, el color es azul y el grosor es 2 píxeles
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)

# Muestra la imagen con los rostros detectados en una ventana titulada 'Detected Faces'
cv2.imshow('Detected Faces', image)

# Espera a que se presione cualquier tecla para cerrar la ventana
cv2.waitKey(0)

# Cierra todas las ventanas abiertas de OpenCV
cv2.destroyAllWindows()
