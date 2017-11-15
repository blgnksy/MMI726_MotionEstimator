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
F_COUNT=0


def motionEstimator(frame1, frame2,f_count):
    refBlock=np.zeros(((16,16,3)),np.uint8)
    searchBlock = np.zeros(((16, 16, 3)), np.uint8)
    for i in range(F_HEIGTH/BLOCK_SIZE):
        for j in range(F_WIDTH/BLOCK_SIZE):
            refBlock=frame1[i*BLOCK_SIZE:i*BLOCK_SIZE+BLOCK_SIZE,j*BLOCK_SIZE:j*BLOCK_SIZE+BLOCK_SIZE,:]
            start_x=i*BLOCK_SIZE-SEARCH_WINDOW
            start_y = j * BLOCK_SIZE - SEARCH_WINDOW
            stop_x=i*BLOCK_SIZE+SEARCH_WINDOW
            stop_y = j * BLOCK_SIZE + SEARCH_WINDOW
            min_MAD=1000000
            if  start_x>=0 & start_y>=0 & stop_x<=F_HEIGTH & stop_y<=F_WIDTH:
                search_block=frame2[start_x:start_x+15,start_y:start_y+15,:]
                s_MAD=np.sum(np.absolute(refBlock-searchBlock))
                if s_MAD<min_MAD:
                    min_MAD=s_MAD
                    min_i=start_x
                    min_j=start_y
            print ("Frame No :%d - I:%d - J:%d - Minimum MAD:%d - Min I:%d - Min J :%d" %(f_count,i,j,min_MAD,start_x,start_y))
            # cv2.imshow('block', ref_block)
            # cv2.moveWindow("block",100,400)


#Reading from file
cap = cv2.VideoCapture('./Input/foreman_cif.y4m')

F_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
F_HEIGTH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print("Width: %d - Height: %d" %(F_WIDTH, F_HEIGTH))
while(cap.isOpened()):
    ret1, frame1 = cap.read()
    F_COUNT+=1
    ret2, frame2 = cap.read()
    #print(frame1.dtype)
    if ret1 == True & ret2==True:
        motionEstimator(frame1=frame1,frame2=frame2,f_count=F_COUNT)
    cv2.imshow('frame',frame1)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()