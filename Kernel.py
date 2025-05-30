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
    

cap = cv2.VideoCapture("path_to_video.mp4")
background_subtractor = get_algorithm(alt_type) ##define alg para subtracao de fundo

def main():
    if not cap.isOpened():
        sys.exit()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame = cv2.resize(frame, (640, 480))  ##redimensiona o frame

        fg_mask = background_subtractor.apply(frame) ##aplica o algoritmo de subtracao de fundo

        fg_mask_filter = filter(fg_mask, "combine")  ##aplica o filtro de morfologia

        ##valido para o exemplo de video da rodovia
        cars_after_mask = cv2.bitwise_and(frame, frame, mask=fg_mask_filter)  ##aplica a mascara filtrada no frame original
        cv2.imshow("Cars after mask", cars_after_mask)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mascara e filter", fg_mask_filter)

        if cv2.waitKey(10) & 0xFF == ord('c'): 
            break

    cap.release()
    cv2.destroyAllWindows()

##falta a aula 5: contar carros e salvar o video