import cv2 
import numpy as np
import os
import sys
from time import sleep

algorithm_types = ["KNN", "GMG", "CNT", "MOG", "MOG2"]
alt_type = algorithm_types[1]

def kernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dIlation":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    elif KERNEL_TYPE == "Erosion":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    elif KERNEL_TYPE == "Opening":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    elif KERNEL_TYPE == "Closing":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    elif KERNEL_TYPE == "Dilate":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    else:
        raise ValueError("Unknown kernel type: {}".format(KERNEL_TYPE))

##print das matrizes de kernel
# print("DILATION")
# print(kernel("dIlation"))
# print("EROSION")
# print(kernel("Erosion"))
# print("OPENING")
# print(kernel("Opening"))
# print("CLOSING")
# print(kernel("Closing"))
# print("DILATE")
# print(kernel("Dilate"))

def filter(frame, filter):
    if filter == "closing":
        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel("Closing"),iterations=2)
    elif filter == "opening":
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel("Opening"), iterations=2)
    elif filter == "dilation":
        return cv2.dilate(frame, kernel("dIlation"), iterations=2)
    elif filter == "erosion":
        return cv2.erode(frame, kernel("Erosion"), iterations=2)
    if filter == "combine": ##combina os filtros
        closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel("Closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel("Opening"), iterations=2)
        dilation = cv2.dilate(opening, kernel("dIlation"), iterations=2)
        return dilation

def get_algorithm(algorithm_type):
    if algorithm_type == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    elif algorithm_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    elif algorithm_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    elif algorithm_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    elif algorithm_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    else:
        raise ValueError("Unknown algorithm type: {}".format(algorithm_type))


w_min = 30 
h_min = 30
offset = 10
linha_ROI = 200  ##linha de referencia para contagem de carros
objects = 0

def centroide(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)

detected_objects = []

def set_info(frame,detected_objects):
    global objects
    for(x,y) in detected_objects:
        if y < linha_ROI + offset and y > linha_ROI - offset:
            objects += 1
            cv2.line(frame, (0, linha_ROI), (640, linha_ROI), (255, 0, 0), 2)
            cv2.putText(frame, "Carros: {}".format(objects), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            detected_objects.remove((x,y))

def show_info(frame, mask):
    text = "Carros detectados: {}".format(objects)
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Mascara", mask)


cap = cv2.VideoCapture("path_to_video.mp4")
background_subtractor = get_algorithm(alt_type) ##define alg para subtracao de fundo

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame = cv2.resize(frame, (640, 480))  # redimensiona o frame

    fg_mask = background_subtractor.apply(frame)  # aplica a subtração de fundo

    fg_mask_filter = filter(fg_mask, "combine")  # aplica filtro morfológico

    contorno, _ = cv2.findContours(fg_mask_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # encontra os contornos

    cv2.line(frame, (0, linha_ROI), (640, linha_ROI), (255, 127, 0), 3)  # desenha a linha de referência

    for i, c in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= w_min) and (h >= h_min)  # valida o contorno
        if not validar_contorno:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # desenha o retângulo ao redor do contorno
        centro = centroide(x, y, w, h)  # calcula o centroide
        detected_objects.append(centro)  # adiciona à lista de objetos detectados
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)  # desenha o centroide

    set_info(detected_objects)  # atualiza a contagem de objetos
    show_info(frame, fg_mask_filter)  # mostra as informações na tela

    if cv2.waitKey(10) & 0xFF == 27:  # tecla Esc
        break

cap.release()
cv2.destroyAllWindows()
