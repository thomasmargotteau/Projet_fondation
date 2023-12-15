import cv2 as cv
from pyautogui import size

# Getting window dimensions with pyautogui lib
ScreenWidth,ScreenHeight=size()
ScreenWidth=ScreenWidth/2
ScreenHeight=ScreenHeight/2

# Loading the image
cap=cv.imread("ArucoBoard.png")

# Rescaling the image to the window dimensions
#cap = cv.resize(cap,(ScreenWidth.__floor__(),ScreenHeight.__floor__()),interpolation=cv.INTER_LINEAR)
#cap = cv.resize(cap,(0,0),fx=1,fy=1)

def findThatAruco(img,markersize=5,totalmarkers=50):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    k=getattr(cv.aruco,f'DICT_{markersize}X{markersize}_{totalmarkers}')
    arucoDict=cv.aruco.Dictionary_get(k)
    arucoParam=cv.aruco.DetectorParameters_create()
    bbox,ids,_=cv.aruco.detectMarkers(gray,arucoDict,parameters=arucoParam)
    cv.aruco.drawDetectedMarkers(img,bbox)
    return bbox,ids

key = 0

while (key!=27):

    cap = cv.imread("ArucoBoard.png")
    cap = cv.resize(cap, (0, 0), fx=2, fy=2)

    bbox,ids=findThatAruco(cap)
    cv.imshow('cap',cap)

    key=cv.waitKey(1)
