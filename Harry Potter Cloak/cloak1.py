import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)

bg=0

for i in range(30):
    ret ,bg=cap.read()


while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break
   
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    l_black = np.array([0,0,0])
    u_black = np.array([360,255,50])
    mask1 = cv2.inRange(hsv, l_black, u_black)

    l_black = np.array([0,0,0])
    u_black = np.array([360,255,50])
    mask2 = cv2.inRange(hsv, l_black, u_black)

    mask1 = mask1+mask2

    mask1=cv2.morphologyEx(mask1 ,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2)
    mask1=cv2.morphologyEx(mask1 ,cv2.MORPH_DILATE,np.ones((3,3),np.uint8),iterations=1)
       
    mask2 = cv2.bitwise_not(mask1) 

    res1=cv2.bitwise_and(bg,bg,mask=mask1)
    res2=cv2.bitwise_and(frame,frame,mask=mask2)
    op=cv2.addWeighted(res1,1,res2,1,0)


    cv2.imshow("mask", op)
    if cv2.waitKey(5) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()