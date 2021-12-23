import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import img_to_array
from keras.models import load_model

modelo = 'C:/Users/gonma/PycharmProjects/Manos/ModeloManos.h5'
pesos = 'C:/Users/gonma/PycharmProjects/Manos/PesosManos.h5'

cnn = load_model(modelo)  # Carga del modelo
cnn.load_weights(pesos)  # Carga de los pesos

direccion = 'C:/Users/gonma/PycharmProjects/Manos/Fotos/Validacion'
dire_img = os.listdir(direccion)
num_items = len(dire_img)
print(f'Items: {num_items}-> {dire_img}')

# Leo la cámara
cap = cv2.VideoCapture(0)

# --------------- Creo un objeto que va a almacenar la detección y el seguimiento de las manos ------------- #
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()  # Primer parámetro: FALSE para que no haga la detección 24/7
# Solo hará detección cuando hay una confianza alta.
# Segundo parámetro: número máximo de manos.
# Tercer parámetro: confianza mínima de detección.
# Cuarto parámetro: confianza mínima de seguimiento.

# -------------------------------------- Método para dibujar las manos ----------------------------------------#
dibujo = mp.solutions.drawing_utils  # Con este método dibujo 21 puntos críticos de la mano.

while 1:
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta linea voy a almacenar las coordenadas de los puntos.
    # print(resultado.multi_hand_landmarks) # Si quiero ver si existe la detección.

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                # print(id, lm)
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                pto_i1 = posiciones[4]
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1] - 150), (pto_i5[2] - 150)
                ancho, alto = (x1 + 150), (y1 + 250)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg)  # Conversion de la imagen a una matriz.
                x = np.expand_dims(x, axis=0)  # Agrego nuevo eje.
                vector = cnn.predict(x)  # Va a ser un array de 2 dimensiones, donde se pondrá 1 en la clase correcta.
                resultado = vector[0]  # [1,0] | [0,1]
                respuesta = np.argmax(resultado)  # Entrega el índice de valor más alto:
                if respuesta == 1:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Se dibuja el rectángulo en pantalla.
                    # Se pone el texto correcto:
                    cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                elif respuesta == 0:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Se dibuja el rectángulo en pantalla.
                    # Se pone el texto correcto:
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:  # 27 es la tecla ESC. (Para cerrar el programa)
        break

cap.release()
cv2.destroyAllWindows()
