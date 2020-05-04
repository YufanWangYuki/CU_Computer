import cv2
import numpy as np


cap = cv2.VideoCapture(0)
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10, detectShadows=True)

while True:
    _, frame = cap.read()

    mask = subtractor.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
