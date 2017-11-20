import cv2
import numpy as np
import time

# BLOCK_SIZE
BLOCK_SIZE = 16
# SEARCH_WINDOWS_SIZE
SEARCH_WINDOW = 16
# Global variables
F_WIDTH = 0
F_HEIGTH = 0
F_COUNT = 0


def motionCompansation(frame1, motionVector):
    motionCompansatedFrame = np.zeros_like(frame1, dtype=frame1.dtype)
    for i in range(F_HEIGTH / BLOCK_SIZE):
        for j in range(F_WIDTH / BLOCK_SIZE):
            mVecX=motionVec[i,j,0]
            mVecY=motionVec[i,j,1]
            motionCompansatedFrame[i * BLOCK_SIZE:i * BLOCK_SIZE + BLOCK_SIZE,
            j * BLOCK_SIZE:j * BLOCK_SIZE + BLOCK_SIZE, :] = frame1[i * BLOCK_SIZE:i * BLOCK_SIZE + BLOCK_SIZE,
                                                             j * BLOCK_SIZE:j * BLOCK_SIZE + BLOCK_SIZE, :]
    return motionCompansatedFrame


def MotionExhaustiveSearcher(frame2, refBlock, start_x, start_y):
    searchBlock = np.zeros(((BLOCK_SIZE, BLOCK_SIZE, 1)), np.uint8)
    startTimeExhPerBlock = time.time()
    if (start_x - SEARCH_WINDOW >= 0) and (start_y - SEARCH_WINDOW >= 0):
        searchBlock = frame2[start_x - SEARCH_WINDOW:start_x + (3 * SEARCH_WINDOW),
                      start_y - SEARCH_WINDOW:start_y + (3 * SEARCH_WINDOW), 0]
    else:
        searchBlock = frame2[start_x:start_x + (3 * SEARCH_WINDOW),
                      start_y:start_y + (3 * SEARCH_WINDOW), 0]

    res = cv2.matchTemplate(searchBlock, refBlock, 0)
    _, _, min_loc, _ = cv2.minMaxLoc(res)
    print("--- %s seconds per Block ---" % (time.time() - startTimeExhPerBlock))
    return min_loc[0], min_loc[1]


def MotionEstimator(frame1, frame2):
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
                mVecX, mVecY = MotionExhaustiveSearcher(frame2=frame2, refBlock=refBlock, start_x=0, start_y=0)
                motionVector[i, j, :] = (mVecX, mVecY)
    print("--- %s seconds per Frame ---" % (time.time() - startTimeExhFrame))
    return motionVector


# Reading from file
cap = cv2.VideoCapture('./Input/foreman_cif.y4m')

F_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
F_HEIGTH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while (cap.isOpened()):
    startTimeVideo = time.time()
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    if ret1 == True & ret2 == True:
        motionVec = MotionEstimator(frame1=frame1, frame2=frame2)
        mCompFrame=motionCompansation(frame1=frame1,motionVector=motionVec)
        # print motionVec
    cv2.imshow('frame', frame1)
    cv2.imshow('motion compansation', mCompFrame)
    if cv2.waitKey(7500) & 0xFF == ord('q'):
        break
print("---> %s seconds per Video <---" % (time.time() - startTimeVideo))
cap.release()
cv2.destroyAllWindows()
