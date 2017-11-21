import cv2
import numpy as np
import time
from enum import Enum

class SearchMode(Enum):
    Exhaustive=1
    Diamond=2

# Global variables
# BLOCK_SIZE
BLOCK_SIZE = 16
# SEARCH_WINDOWS_SIZE
SEARCH_WINDOW = 16
# Frame Width
F_WIDTH = 0
# Frame Heigth
F_HEIGTH = 0
#Frame Count
F_COUNT=0


def MeanSquaredError(frameO, frameC):
    err = np.sum((frameO.astype("float") - frameC.astype("float")) ** 2)
    err /= float(F_WIDTH * F_HEIGTH)
    return err

def PeakSignolNoiseRatio(frameO, frameC):
    mse=MeanSquaredError(frameO=frameO,frameC=frameC)
    return 10*np.log10(255*255/mse)

def MotionCompansation(frame1, motionVector):
    motionCompansatedFrame = np.zeros_like(frame1, dtype=frame1.dtype)
    try:
        for i in range(F_HEIGTH / BLOCK_SIZE):
            for j in range(F_WIDTH / BLOCK_SIZE):
                mVecX = motionVector[i, j, 0]
                mVecY = motionVector[i, j, 1]
                motionCompansatedFrame[mVecX:mVecX+BLOCK_SIZE, mVecY:mVecY+BLOCK_SIZE, :] = frame1[
                                                                                              i * BLOCK_SIZE:i * BLOCK_SIZE + BLOCK_SIZE,
                                                                                              j * BLOCK_SIZE:j * BLOCK_SIZE + BLOCK_SIZE,
                                                                                              :]
    except:
        print "Error" #I faced a an error but I was not able to recover it.
    return motionCompansatedFrame


def MotionExhaustiveSearcher(frame2, refBlock, start_x, start_y):
    searchBlock = np.zeros(((BLOCK_SIZE, BLOCK_SIZE, 1)), np.uint8)
    startTimeExhPerBlock = time.time()
    if (start_x - SEARCH_WINDOW >= 0) and (start_y - SEARCH_WINDOW >= 0):
        searchBlock = frame2[start_x - SEARCH_WINDOW:start_x + (3 * SEARCH_WINDOW),
                      start_y - SEARCH_WINDOW:start_y + (3 * SEARCH_WINDOW), 0]
        sx=start_x - SEARCH_WINDOW
        sy=start_y - SEARCH_WINDOW
    else:
        searchBlock = frame2[start_x:start_x + (3 * SEARCH_WINDOW),
                      start_y:start_y + (3 * SEARCH_WINDOW), 0]
        sx = start_x
        sy = start_y

    res = cv2.matchTemplate(searchBlock, refBlock, 0)
    _, _, min_loc, _ = cv2.minMaxLoc(res)
    print("--- %s seconds per Block (Exhaustive Search) ---" % (time.time() - startTimeExhPerBlock))
    return sx+min_loc[0], sy+min_loc[1]

def MotionDiamondSearcher(frame2,refBlock):

    return 0

def MotionEstimator(frame1, frame2, searchMode):
    startTimeExhFrame = time.time()
    # I used only one channel for the sake simplicity.
    refBlock = np.zeros(((BLOCK_SIZE, BLOCK_SIZE, 1)), np.uint8)

    # To keep motion vector
    motionVector = np.zeros((F_HEIGTH / BLOCK_SIZE, F_WIDTH / BLOCK_SIZE, 2))

    # Dividing Reference Image into Blocks
    for i in range(F_HEIGTH / BLOCK_SIZE):
        for j in range(F_WIDTH / BLOCK_SIZE):
            refBlock = frame1[i * BLOCK_SIZE:i * BLOCK_SIZE + BLOCK_SIZE, j * BLOCK_SIZE:j * BLOCK_SIZE + BLOCK_SIZE, 0]

            start_x = i * BLOCK_SIZE
            start_y = j * BLOCK_SIZE
            stop_x = i * BLOCK_SIZE + SEARCH_WINDOW + BLOCK_SIZE
            stop_y = j * BLOCK_SIZE + SEARCH_WINDOW + BLOCK_SIZE

            if start_x >= 0 & start_y >= 0 & stop_x <= F_HEIGTH & stop_y <= F_WIDTH:
                if  searchMode==SearchMode.Exhaustive:
                    mVecX, mVecY = MotionExhaustiveSearcher(frame2=frame2, refBlock=refBlock, start_x=start_x, start_y=start_y)
                else:
                    mVecX, mVecY = MotionDiamondSearcher(frame2=frame2, refBlock=refBlock, start_x=start_x,
                                                            start_y=start_y)
                motionVector[i, j, :] = (mVecX, mVecY)
    print("--- %s seconds per Frame (Exhaustive Search)---" % (time.time() - startTimeExhFrame))
    return motionVector


if __name__ == "__main__":
    # Reading from file
    cap = cv2.VideoCapture('./Input/foreman_cif.y4m')

    F_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    F_HEIGTH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    startTimeVideo = time.time()
    while (cap.isOpened()):
        F_COUNT+=1
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        if ret1 == True & ret2 == True:
            motionVector = MotionEstimator(frame1=frame1, frame2=frame2,searchMode=SearchMode.Exhaustive)
            mCompFrame = MotionCompansation(frame1=frame1, motionVector=motionVector)
            # print motionVec
        cv2.imshow('Reference Frame of Sequence', frame1)
        cv2.moveWindow('Reference Frame of Sequence', 0, 0)

        cv2.imshow('Motion Compensated Frame', mCompFrame)
        cv2.moveWindow('Motion Compensated Frame', 10 + F_HEIGTH, 10 + F_WIDTH)

        resErr=cv2.absdiff(frame2,mCompFrame)
        cv2.imshow('Residual Error', resErr)
        cv2.moveWindow('Residual Error', 200, 50)

        resErrFilename="./Output/resErrFrame_"+str(F_COUNT+1)+".png"
        cv2.imwrite(resErrFilename,resErr)

        mCompFilename="./Output/mCompFrame_"+str(F_COUNT+1)+".png"
        cv2.imwrite(mCompFilename,mCompFrame)

        psnr=PeakSignolNoiseRatio(frameO=frame2,frameC=mCompFrame)

        print ("The PSNR of Original Frame-%d and Motion Companseted Image is equal to %d" %(F_COUNT,psnr))
        # For the sake simplicity, I just used  first 6 frame.
        #if F_COUNT==6:
        #    break
        if cv2.waitKey(7500) & 0xFF == ord('q'):
            break
    print("---> %s seconds per Video <---" % (time.time() - startTimeVideo))
    cap.release()
    cv2.destroyAllWindows()