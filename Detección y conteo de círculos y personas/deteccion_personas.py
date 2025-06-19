import cv2

# Inicializamos el detector HOG con un clasificador preentrenado para personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detectar_y_contar_personas(imagen):
    # Redimensionamos la imagen para acelerar el proceso de detección (opcional)
    imagen_redimensionada = cv2.resize(imagen, (min(800, imagen.shape[1]), min(600, imagen.shape[0])))

    # Detectamos personas en la imagen
    (rectangulos, _) = hog.detectMultiScale(imagen_redimensionada, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Dibujamos los rectángulos alrededor de las personas detectadas
    for (x, y, w, h) in rectangulos:
        cv2.rectangle(imagen_redimensionada, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Contamos la cantidad de personas detectadas
    cantidad_personas = len(rectangulos)

    # Escribimos la cantidad de personas detectadas en la imagen
    cv2.putText(imagen_redimensionada, f"Personas detectadas: {cantidad_personas}", (10, imagen_redimensionada.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return imagen_redimensionada, cantidad_personas

# Cargar la imagen donde se desea detectar personas
imagen = cv2.imread('12.jpeg')

# Detectar y contar personas en la imagen
imagen_resultante, cantidad_personas = detectar_y_contar_personas(imagen)

# Mostrar la imagen con las personas detectadas y el conteo
cv2.imshow("Deteccion de Personas", imagen_resultante)
print(f"Cantidad de personas detectadas: {cantidad_personas}")
cv2.waitKey(0)
cv2.destroyAllWindows()
