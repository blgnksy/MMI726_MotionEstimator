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
    frame1_padded=np.zeros((int(F_HEIGTH)+(BLOCK_SIZE*2),int(F_WIDTH)+(BLOCK_SIZE*2),3), dtype=np.uint8)
    # print(frame1_padded.shape)
    # print(frame1.shape)
    frame1_padded[BLOCK_SIZE-1:(BLOCK_SIZE-1+int(F_HEIGTH)),BLOCK_SIZE-1:(BLOCK_SIZE-1+int(F_WIDTH)),:]=frame1
    frame2_blocks=np.zeros((int(F_HEIGTH)/BLOCK_SIZE,int(F_WIDTH)/BLOCK_SIZE,3), dtype=np.uint8)
    for i in range(frame2_blocks.shape[0]):
        for j in range(frame2_blocks.shape[1]):
            cv2.imshow("Chunks", frame2[i*BLOCK_SIZE:i*BLOCK_SIZE-1+BLOCK_SIZE,j*BLOCK_SIZE:j*BLOCK_SIZE-1+BLOCK_SIZE,:])
            cv2.moveWindow("Chunks", 200, 700)
    #cv2.imshow("padded",frame1_padded)
    #cv2.moveWindow("padded", 0, 350)
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
    cv2.imshow('frame',frame1)
    if cv2.waitKey(50) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()