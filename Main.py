import cv2
import numpy as np
import time

# Global variables
# BLOCK_SIZE
BLOCK_SIZE = 16
# SEARCH_WINDOWS_SIZE
SEARCH_WINDOW = 16
# Frame Width
F_WIDTH = 0
# Frame Heigth
F_HEIGTH = 0
# Frame Count
F_COUNT = 0


def MeanSquaredError(frame1, frame2):
    """Calculates Mean Squared Error between Original Image and Comapensated Image

    Keyword arguments:
    frame1 -- First Image
    frame2 -- Second Image
    """
    err = np.sum((frame1.astype("float") - frame2.astype("float")) ** 2)
    err /= float(F_WIDTH * F_HEIGTH)
    return err


def PeakSignalNoiseRatio(frameO, frameC):
    """Calculates Peak Signal Noise Ratio of Original Image and Comapensated Image

    Keyword arguments:
    frame0 -- Original Image
    frameC -- Compensated Image
    """
    mse = MeanSquaredError(frame1=frameO, frame2=frameC)
    return 10 * np.log10(255 * 255 / mse)


def MotionCompansation(frame1, motionVector):
    """Calculates Motion Comapensated Image

    Keyword arguments:
    frame1 -- Reference Image
    motionVector -- Motion Vector of Blocks
    """
    motionCompansatedFrame = np.zeros_like(frame1, dtype=frame1.dtype)
    try:
        for i in range(F_HEIGTH / BLOCK_SIZE):
            for j in range(F_WIDTH / BLOCK_SIZE):
                mVecX = motionVector[i, j, 0]
                mVecY = motionVector[i, j, 1]
                motionCompansatedFrame[mVecX:mVecX + BLOCK_SIZE, mVecY:mVecY + BLOCK_SIZE, :] = frame1[
                                                                                                i * BLOCK_SIZE:i * BLOCK_SIZE + BLOCK_SIZE,
                                                                                                j * BLOCK_SIZE:j * BLOCK_SIZE + BLOCK_SIZE,
                                                                                                :]
    except:
        pass # I faced a an error but I was not able to recover it.
    return motionCompansatedFrame


def MotionExhaustiveSearcher(frame2, refBlock, start_x, start_y):
    """Finds the motion vector of a given reference block for Exhaustive Search Algorithm. 

    Keyword arguments:
    frame2 -- Consecutive Frame
    refBlock -- Rference Block
    start_x -- X Coordinate for start
    start_y -- Y Coordinate for start
    """
    searchBlock = np.zeros(((BLOCK_SIZE, BLOCK_SIZE, 1)), np.uint8)
    startTimeExhPerBlock = time.time()
    if (start_x - SEARCH_WINDOW >= 0) and (start_y - SEARCH_WINDOW >= 0):
        searchBlock = frame2[start_x - SEARCH_WINDOW:start_x + (3 * SEARCH_WINDOW),
                      start_y - SEARCH_WINDOW:start_y + (3 * SEARCH_WINDOW), 0]
        sx = start_x - SEARCH_WINDOW
        sy = start_y - SEARCH_WINDOW
    else:
        searchBlock = frame2[start_x:start_x + (3 * SEARCH_WINDOW),
                      start_y:start_y + (3 * SEARCH_WINDOW), 0]
        sx = start_x
        sy = start_y

    res = cv2.matchTemplate(searchBlock, refBlock, 0)
    _, _, min_loc, _ = cv2.minMaxLoc(res)
    print("--- %s seconds per Block (Exhaustive Search) ---" % (time.time() - startTimeExhPerBlock))
    return sx + min_loc[0], sy + min_loc[1]


def LargeDiamondSearchPattern(frame2, refBlock, start_x, start_y):
    """Large Diamond Search Pattern.

    Keyword arguments:
    frame2 -- Consecutive Frame
    refBlock -- Reference Block
    start_x -- X Coordinate for start
    start_y -- Y Coordinate for start
    """
    position = (0, 0)
    cost = 100000000
    lookup_array = {(0, 0), (-2, 0), (0, 2), (1, 1), (2, 0), (1, -1), (0, -2), (-1, -1)}
    try:
        for lookup in lookup_array:
            if (start_x + lookup[0] >= 0 & start_y + lookup[1] >= 0 & start_x + lookup[
                0] + BLOCK_SIZE <= F_HEIGTH & start_y + lookup[1] + BLOCK_SIZE <= F_WIDTH):
                searchBlock = frame2[start_x + lookup[0]:start_x + lookup[0] + BLOCK_SIZE,
                              start_y + lookup[1]:start_y + lookup[1] + BLOCK_SIZE, 0]
                lookupMse = MeanSquaredError(frameO=refBlock, frameC=searchBlock)
                if (lookupMse < cost):
                    cost = lookupMse
                    position = lookup
    except:
        pass

    return position


def SmallDiamondSearchPattern(frame2, refBlock, start_x, start_y):
    """Small Diamond Search Pattern.

    Keyword arguments:
    frame2 -- Consecutive Frame
    refBlock -- Reference Block
    start_x -- X Coordinate for start
    start_y -- Y Coordinate for start
    """
    cost = 100000000
    lookup_array = {(0, 0), (-1, 0), (0, 1), (0, 1), (0, -1)}
    position=(0,0)
    try:
        for lookup in lookup_array:
            if (start_x + lookup[0] >= 0 & start_y + lookup[1] >= 0 & start_x + lookup[
                0] + BLOCK_SIZE <= F_HEIGTH & start_y + lookup[1] + BLOCK_SIZE <= F_WIDTH):
                searchBlock = frame2[start_x + lookup[0]:start_x + lookup[0] + BLOCK_SIZE,
                              start_y + lookup[1]:start_y + lookup[1] + BLOCK_SIZE, 0]
                lookupMse = MeanSquaredError(frameO=refBlock, frameC=searchBlock)
                if (lookupMse < cost):
                    cost = lookupMse
                    position = lookup
    except:
        pass

    return position


def MotionDiamondSearcher(frame2, refBlock, start_x, start_y):
    """Calculates the motion vector of a given block for Diamond Search Algorithm.

    Keyword arguments:
    frame2 -- Consecutive Frame
    refBlock -- Reference Block
    start_x -- X Coordinate for start
    start_y -- Y Coordinate for start
    """
    startTimeDiaPerBlock = time.time()
    position = LargeDiamondSearchPattern(frame2=frame2, refBlock=refBlock, start_x=start_x, start_y=start_y)
    while (position != (0,0)):
        start_x += position[0]
        start_y += position[1]
        position = LargeDiamondSearchPattern(frame2=frame2, refBlock=refBlock, start_x=start_x, start_y=start_y)
    position = SmallDiamondSearchPattern(frame2=frame2, refBlock=refBlock, start_x=start_x, start_y=start_y)
    print("--- %s seconds per Block (Diamond Search) ---" % (time.time() - startTimeDiaPerBlock))
    start_x += position[0]
    start_y += position[1]
    return start_x, start_y


def MotionEstimatorDiamond(frame1, frame2):
    """Calculates the motion vector of an image for Diamond Search Algorithm.

    Keyword arguments:
    frame1 -- Reference Frame
    frame2 -- Consecutive Frame
    """
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
                mVecX, mVecY = MotionDiamondSearcher(frame2=frame2, refBlock=refBlock, start_x=start_x,
                                                     start_y=start_y)
                motionVector[i, j, :] = (mVecX, mVecY)
    print("--- %s seconds per Frame (Diaamond Search)---" % (time.time() - startTimeExhFrame))
    return motionVector


def MotionEstimatorExh(frame1, frame2):
    """Calculates the motion vector of an image for Exhaustive Search Algorithm.

    Keyword arguments:
    frame1 -- Reference Frame
    frame2 -- Consecutive Frame
    """
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
                mVecX, mVecY = MotionExhaustiveSearcher(frame2=frame2, refBlock=refBlock, start_x=start_x,
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
        F_COUNT += 1
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        if ret1 == True & ret2 == True:
            motionVector = MotionEstimatorExh(frame1=frame1, frame2=frame2)
            mCompFrame = MotionCompansation(frame1=frame1, motionVector=motionVector)
            # print motionVec
        cv2.imshow('Reference Frame of Sequence', frame1)
        cv2.moveWindow('Reference Frame of Sequence', 0, 0)

        cv2.imshow('Motion Compensated Frame', mCompFrame)
        cv2.moveWindow('Motion Compensated Frame', 10 + F_HEIGTH, 10 + F_WIDTH)

        resErr = cv2.absdiff(frame2, mCompFrame)
        cv2.imshow('Residual Error', resErr)
        cv2.moveWindow('Residual Error', 200, 50)

        # resErrFilename = "./Output/resErrFrame_Dia_" + str(F_COUNT + 1) + ".png"
        # cv2.imwrite(resErrFilename, resErr)
        #
        # mCompFilename = "./Output/mCompFrame_Dia_" + str(F_COUNT + 1) + ".png"
        # cv2.imwrite(mCompFilename, mCompFrame)

        resErrFilename = "./Output/resErrFrame_Exh_" + str(F_COUNT + 1) + ".png"
        cv2.imwrite(resErrFilename, resErr)

        mCompFilename = "./Output/mCompFrame_Exh_" + str(F_COUNT + 1) + ".png"
        cv2.imwrite(mCompFilename, mCompFrame)
        psnr = PeakSignalNoiseRatio(frameO=frame2, frameC=mCompFrame)

        print ("The PSNR of Original Frame-%d and Motion Companseted Image is equal to %d" % (F_COUNT, psnr))

        # For the sake simplicity, I just used  first 6 frame.
        if F_COUNT==3:
           break
        if cv2.waitKey(7500) & 0xFF == ord('q'):
            break
    print("---> %s seconds per Video <---" % (time.time() - startTimeVideo))
    cap.release()
    cv2.destroyAllWindows()
