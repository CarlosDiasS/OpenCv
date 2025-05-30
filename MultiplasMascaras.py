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
background_subtractor = []
for algo,i in enumerate(algorithm_types):
    background_subtractor.append(background_subtractor(i))  ##define alg para subtracao de fundo

def main():
    if not cap.isOpened():
        sys.exit()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame = cv2.resize(frame, fx=0.35, fy=0.35)  ##redimensiona o frame

        KNN = background_subtractor[0].apply(frame) ##aplica o algoritmo de subtracao de fundo
        GMG = background_subtractor[1].apply(frame) ##aplica o algoritmo de subtracao de fundo
        CNT = background_subtractor[2].apply(frame) ##aplica o algoritmo de subtracao de fundo
        MOG = background_subtractor[3].apply(frame) ##aplica o algoritmo de subtracao de fundo
        MOG2 = background_subtractor[4].apply(frame) ##aplica o algoritmo de subtracao de fundo

        cv2.imshow("Frame", frame)
        cv2.imshow("KNN", KNN)
        cv2.imshow("GMG", GMG)
        cv2.imshow("CNT", CNT)
        cv2.imshow("MOG", MOG)
        cv2.imshow("MOG2", MOG2)


        if cv2.waitKey(10) & 0xFF == ord('c'): 
            break

    cap.release()
    cv2.destroyAllWindows()
