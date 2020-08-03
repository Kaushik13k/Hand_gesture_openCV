import cv2
import numpy as np

hand = cv2.imread('hand.png', 0)

ret, thres = cv2.threshold(hand, 70, 255, cv2.THRESH_BINARY) #hand= grey scale image, 70= threshhold value, 255= max threshvalue
                                                                    #cv2.THRESH_BINARY= backgroud black fore is while

contours,_ = cv2.findContours(thres.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #copy of image, kind of countors ,approx funcn
#returns image , contours, heirchy....hence we dont need any image or heirchy its given as "_"

hull = [cv2.convexHull(c) for c in contours]#single line for loop, c= countors
#Hull means the exterior or the shape of the object.... for more info google :)
final = cv2.drawContours(hand, hull, -1, (255,0,0))

cv2.imshow('Originals', hand)
cv2.imshow('Thresh',thres)
cv2.imshow('Convex hull',final)

cv2.waitKey(0)
cv2.destroyAllWindows()