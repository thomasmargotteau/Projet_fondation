import cv2 as cv
from pyautogui import size

# Getting window dimensions with pyautogui lib
ScreenWidth,ScreenHeight=size()

# Loading the image
cap=cv.imread("CarteVideDessus.jpg")

# Rescaling the image to the window dimensions
cap = cv.resize(cap,(ScreenWidth,ScreenHeight),interpolation=cv.INTER_LINEAR)
key = 0

while (key!=27):
    cv.imshow('cap',cap)
    key=cv.waitKey(1)
