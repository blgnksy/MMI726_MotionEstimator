import cv2
import numpy as np

print(cv2.__version__)
print(np.__version__)

#BLOCK_SIZE
BLOCK_SIZE=16
#SEARCH_WINDOWS_SIZE
SEARCH_WINDOW=16
#Global variables
F_WIDTH=0
F_HEIGTH=0

def motionEstimator(frame1, frame2):
    block=np.zeros(((16,16,3)),np.uint8)
    for i in range(1):
        for j in range(1):
            block=frame1[i*BLOCK_SIZE:i*BLOCK_SIZE+BLOCK_SIZE-1,j*BLOCK_SIZE:j*BLOCK_SIZE+BLOCK_SIZE-1,:]
    cv2.imshow("b",block)
    cv2.moveWindow("b", 0, 350)
#Reading from file
cap = cv2.VideoCapture('./Input/foreman_cif.y4m')
F_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
F_HEIGTH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);

print("Width: %d - Height: %d" %(int(F_WIDTH), int(F_HEIGTH)))
while(cap.isOpened()):
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    #print(frame1.dtype)
    motionEstimator(frame1=frame1,frame2=frame2)
    #cv2.imshow('frame',frame1)
    if cv2.waitKey(50) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()