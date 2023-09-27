import numpy as np
import cv2
import sys
import imutils
import numba as nb


@nb.jit
def setsize(input, resultheight, resultwidth):
    inimage = cv2.imread(input).astype(np.uint8)
    prevheight, prevwidth = inimage.shape[:2]
    col = resultwidth - prevwidth 
    row = resultheight - prevheight
    if np.absolute(row) > 0:
        inimage = imutils.rotate_bound(inimage, 90)
        col = row
    resultimg = np.copy(inimage)
    energy = getmap(resultimg)
    if col > 0:
        scaledimage = np.copy(inimage)
        cumulativeseam = []
        for temp in range(col):
            seam = getseam(resultimg, energy)
            cumulativeseam.append(seam)
            resultimg = delseam(resultimg, seam)
            energy = getmap(resultimg)
        seamnum = len(cumulativeseam)
        scaledimage = np.copy(inimage)
        for temp in range(seamnum):
            seam = cumulativeseam.pop(0)
            scaledimage = addseam(seam, scaledimage)
            cumulativeseam = newseam(cumulativeseam, seam)
        if np.absolute(row) > 0:
            scaledimage = imutils.rotate_bound(scaledimage, -90)
        cv2.imwrite('resized_output.png', scaledimage.astype(np.uint8))
    else:
        col = np.absolute(col)
        for temp in range(col):
            seam = getseam(resultimg, energy)
            resultimg = delseam(resultimg, seam)
            energy = getmap(resultimg)
        if np.absolute(row) > 0:
            resultimg = imutils.rotate_bound(resultimg, 270)
        cv2.imwrite('resized_output.png', resultimg.astype(np.uint8))
    return -1

def getseam(image, energy):
    vert, horiz = image.shape[:2]
    seam = np.zeros(image.shape[0])
    diff = np.zeros(image.shape[:2]) + sys.maxsize
    diff[0,:] = np.zeros(image.shape[1])
    edge = np.zeros(image.shape[:2])
    for width in range(vert-1):
        for height in range(horiz):
            if height != 0:
                one = diff[width+1, height-1]
                behind = diff[width, height] + energy[width+1, height-1]
                diff[width+1, height-1] = min(one, behind)
                if one > behind:
                    edge[width+1, height-1] = 1
            two = diff[width+1, height]
            present = diff[width, height] + energy[width+1, height]
            diff[width+1, height] = min(two, present)
            if two>present:
                edge[width+1, height] = 0
            if height != horiz-1:
                three = diff[width+1, height+1]
                ahead = diff[width, height] + energy[width+1, height+1]
                diff[width+1, height+1] = min(three, ahead)
                if three>ahead:
                    edge[width+1, height+1] = -1

    seam[vert-1] = np.argmin(diff[vert-1,:])
    for i in (x for x in reversed(range(vert)) if  x>0):
        seam[i-1] = seam[i] + edge[i, int(seam[i])]
    return seam

@nb.jit
def addseam(seam, image):
    vert, horiz = image.shape[:2]
    empty = np.zeros((vert, 1, 3), dtype=np.uint8)
    scaledimg = np.hstack((image, empty))
    for width in range(vert):
        for height in range(horiz, int(seam[width]), -1):
            scaledimg[width, height] = image[width, height-1]
        for i in range(3):
            temp1 = scaledimg[width, int(seam[width])-1, i]
            temp2 = scaledimg[width, int(seam[width])+1, i]
            scaledimg[width, int(seam[width]), i] = (int(temp1)+(temp2))/2
    return scaledimg

@nb.jit
def delseam(image, seam):
    vert, horiz = image.shape[:2]
    for width in range(vert):
        for height in range(int(seam[width]), horiz-1):
            image[width, height] = image[width, height+1]
    shrunkimg = image[:, 0:horiz-1]
    return shrunkimg

@nb.jit
def newseam(left, present):
    updated= []
    for temp in left:
        temp[np.where(temp >= present)] += 2
        updated.append(temp)
    return updated  

            
def getmap(image):
    image = cv2.GaussianBlur(image, (3,3), 0).astype(np.int8)
    newimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(newimg, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(newimg, cv2.CV_64F, 0, 1, ksize=3)
    defx = cv2.convertScaleAbs(x)
    defy = cv2.convertScaleAbs(y)
    energy = cv2.addWeighted(defx, 0.5, defy, 0.5, 0)
    return energy

if __name__ == '__main__':
    inimage = input("Enter the name of the image: ")
    image = cv2.imread(inimage)
    print("Current dimensions (HxW):", str(image.shape[:2]))  
    resulth = input("Resized height: ")
    resultw = input("Resized width: ")
    setsize(inimage, int(resulth), int(resultw))
    print("Completed!")

    
