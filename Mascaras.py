import cv2 
import numpy as np
import os
import sys
from time import sleep

algorithm_types = ["KNN", "GMG", "CNT", "MOG", "MOG2"]
alt_type = algorithm_types[3]

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
        cv2.imshow("Frame", frame)
        cv2.imshow("Mascara", fg_mask)

        if cv2.waitKey(10) & 0xFF == ord('c'): 
            break

    cap.release()
    cv2.destroyAllWindows()
