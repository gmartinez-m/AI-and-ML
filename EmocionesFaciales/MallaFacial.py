import cv2
import mediapipe as mp
import math

# ------- Captura de video por webcam -------- #
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Ancho de ventana
cap.set(4, 720)  # Largo de ventana

# --------- Funcion dibujo ---------------- #
mpDibujo = mp.solutions.drawing_utils
confDibujo = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)  # Configuración del dibujo

# --------- Objeto donde se almacena la malla facial ------------ #
mpMallaFacial = mp.solutions.face_mesh  # Llamada al objeto e instancia
mallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)  # Creación del objeto

# --------- While ---------- #
while True:
    ret, frame = cap.read()  # Lectura de cámara
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Correción de color

    resultados = mallaFacial.process(frameRGB)

    # Listas para almacenar resultados:
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:  # Si se detecta algún rostro
        for rostro in resultados.multi_face_landmarks:  # Se muestra el rostro detectado
          mpDibujo.draw_landmarks(frame, rostro, mpMallaFacial.FACEMESH_CONTOURS, confDibujo, confDibujo)

          # Ahora se extraen los puntos del rostro detectado:
          for id, puntos in enumerate(rostro.landmark):
              alto, ancho, c = frame.shape
              x, y = int(puntos.x*ancho), int(puntos.y*alto)
              px.append(x)
              py.append(y)
              lista.append([id, x, y])

              # -------- ver png "PUNTOS CARA INFO" para saber los puntos faciales clave ------------ #
              if len(lista) == 468:
                  # CEJA DERECHA
                  x1, y1 = lista[65][1:]
                  x2, y2 = lista[158][1:]
                  cx, cy = (x1+x2) // 2, (y1+y2) // 2
                  #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                  #cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED)
                  #cv2.circle(frame, (x2, y2), r, (0, 0, 0), cv2.FILLED)
                  #cv2.circle(frame, (cx, cy), r, (0, 0, 0), cv2.FILLED)
                  longitud1 = math.hypot(x2 - x1, y2 - y1)


                  # CEJA IZQUIERDA
                  x3, y3 = lista[295][1:]
                  x4, y4 = lista[385][1:]
                  cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                  longitud2 = math.hypot(x4 - x3, y4 - y3)

                  # BOCA EXTREMOS
                  x5, y5 = lista[78][1:]
                  x6, y6 = lista[308][1:]
                  cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                  longitud3 = math.hypot(x6 - x5, y6 - y5)

                  # BOCA APERTURA
                  x7, y7 = lista[13][1:]
                  x8, y8 = lista[14][1:]
                  cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                  longitud4 = math.hypot(x8 - x7, y8 - y7)

                  # ---- Condicionales ----- #
                  # Enfadado:
                  if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                      cv2.putText(frame, 'Persona ENFADADA', (480,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)

                  # Feliz:
                  elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20:
                      cv2.putText(frame, 'Persona FELIZ', (480,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),3)

                  # Asombrado:
                  elif longitud1 > 35 and longitud2 > 35 and longitud3 > 85 and longitud3 < 90 and longitud4 > 20:
                      cv2.putText(frame, 'Persona ASOMBRADA', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                  # Triste:
                  elif longitud1 > 25 and longitud1 < 35 and longitud2 > 25 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5:
                      cv2.putText(frame, 'Persona TRISTE', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)






    cv2.imshow("Emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()