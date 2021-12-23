import cv2
import SeguimientoManos as sm
import numpy as np

# ----------------- Librerias para controlar el volumen ------------------ #
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ----------------- Parámetros de la cámara ---------------------- #
anchoCam, altoCam = 640, 480

# ----------------- Lectura de la cámara ------------------------- #
cap = cv2.VideoCapture(0)
cap.set(3, anchoCam)
cap.set(4, altoCam)

# ----------- Objeto que almacena nuestra clase ------------------ #
detector = sm.DetectorManos(maxManos=1, confDeteccion=0.7)

# ----------------- Control de audio del PC ---------------------- #
dispositivos = AudioUtilities.GetSpeakers()
interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
RangoVol = volumen.GetVolumeRange()
#print(f" RANGO VOLUMEN EQUIPO ----->> {RangoVol}")

volMin = RangoVol[0]
volMax = RangoVol[1]

while True:
    ret, frame = cap.read()  # Lectura de la captura de video


    frame = detector.encontrar_manos(frame)  # LLamo al objeto que contiene la clase para la detección
    lista, bbox = detector.encontrar_posicion(frame, dibujar=False)  # Llamo al método que me devuelve la posición
    if len(lista) != 0:  # Si hay algo en la lista lo voy a imprimir en este caso selecciono el punto 4 y 8.
        #print(lista[4], lista[8])  # Estos puntos pertenecen al extremo del pulgar y el extremo del índice.
        x1, y1 = lista[4][1], lista[4][2]  # Extracción de coordenadas x e y del pulgar
        x2, y2 = lista[8][1], lista[8][2]  # Extracción de coordenadas x e y del índice

        # ---------------- Voy a comprobar que el dedo índice y el pulgar estén levantados ----------------- #
        dedos = detector.dedos_arriba()
        #print(dedos)

        if dedos[0] == 1 and dedos[1] == 1:
            longitud, frame, linea = detector.distancia(4, 8, frame, r=8, t=2)
            #print(f"LONGITUD ---->>> {longitud}")

            #  Mi rango de manos es 25 hasta 200.
            #  Mi rango de volumen es de -96 a 0.
            vol = np.interp(longitud, [0, 150], [volMin, volMax])  # 25 es a volMin lo que 200 es a volMax
            #print(f"VOLUMEN ----->> {vol}")
            volumen.SetMasterVolumeLevel(vol, None)

            if longitud < 0:
                cv2.circle(frame, (linea[4], linea[5]), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Video", frame)
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()








