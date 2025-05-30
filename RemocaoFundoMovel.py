import numpy as np
import cv2

video = "path"

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
print(medianFrame)

cv2.imshow("Frames_medios.jpg", medianFrame)