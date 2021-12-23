import math
import cv2
import mediapipe as mp
import time


class DetectorManos:
    # ------------- Inicio los parámetros de detección -------------- #
    def __init__(self, mode=False, maxManos=2, modelComplexity=1, confDeteccion=0.5, confSeguimiento=0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.modelComplex = modelComplexity
        self.confDeteccion = confDeteccion
        self.confSeguimiento = confSeguimiento

        # ------------- creo los objetos que detectan las manos y las dibujan ------------ #
        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(self.mode, self.maxManos,self.modelComplex, self.confDeteccion, self.confSeguimiento)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    # --------------- Función para encontrar las manos ------------------ #
    def encontrar_manos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS)
        return frame

    # ---------------- Funcion para encontrar la posición ---------------- #
    def encontrar_posicion(self, frame, manoNum=0, dibujar=True):
        xlista = []
        ylista = []
        bbox = []
        self.lista = []

        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[manoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape  # Extraigo las dimensiones de los fotogramas
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # Se convierte la información a pixeles.
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)  # Dibujo un círculo

            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax
            if dibujar:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lista, bbox

    # ----------------- Función para detectar y dibujar los dedos arriba ---------------------- #
    def dedos_arriba(self):
        dedos = []
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos

    # ----------------- Función para detectar la distancia entre dedos ---------------------- #
    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        # Hay que indicar los dos puntos de interes que quiero: extremo del índice (8) y extremo el pulgar (4):
        x1, y1 = self.lista[p1][1:]
        x2, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # La c de cx y cy significa CENTRO

        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)  # Linea entre extremo índice y extremo pulgar
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)  # Círculo en el extremo del índice
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)  # Círculo en el extremo del pulgar
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)  # Círculo entre ambos extremos

        distancia = math.hypot(x2 - x1, y2 - y1)

        return distancia, frame, [x1, y1, x2, y2, cx, cy]


# ----------------- Función para detectar la distancia entre dedos ---------------------- #
def main():
    ptiempo = 0
    ctiempo = 0

    # Lectura de la webcam:
    cap = cv2.VideoCapture(0)

    # Creación del objeto:
    detector = DetectorManos()
    # -------- Detección de las manos ----------- #
    while True:
        ret, frame = cap.read()

        # Una vez se tenga la imagen, se procesa:
        frame = detector.encontrar_manos(frame)
        lista, bbox = detector.encontrar_posicion(frame)

        # ----------- Mostramos los FPS ------------ #
        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo)

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Manos", frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
