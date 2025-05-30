import numpy as np
import cv2
from time import sleep

video = "path"
delay = 10 ##const

cap = cv2.VideoCapture(video)
hasFrame, frame = cap.read()

framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT)*np.random.uniform(size=72)

frames=[]

for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    if not hasFrame:
        continue
    frames.append(frame)

medianFrame = np.median(frames,axis=0).astype(dtype=np.uint8)
cv2.imshow("Frames_medios.jpg", medianFrame)

cap.set(cv2.CAP_PROP_POS_FRAMES,0)

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
##cv2.imshow("Cinza", grayMedianFrame)
##cv2.waitKey(0)
cv2.imwrite("median_frame_jpg", grayMedianFrame)

##percorrer todos os frames

while(True):

    temp = float(1/delay)
    sleep(temp) ##delay para o video

    hasFrame, frame = cap.read()
    if not hasFrame:
        break
        
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayMedianFrame, frameCinza)
    threshold, diff = cv2.threshold(diff,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) ##diferentes contrastes de acordo com o threshold  cv2.THRESH_OTSU AUTO-AJUSTA O THRESHOLD

    cv2.imshow("Frame Cinza", frameCinza)
    if(cv2.waitKey(1) & 0xFF == ord('c')): ##Press 'c' to continue to the next frame
        break




cap.release()







