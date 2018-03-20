
import numpy as np
import cv2
image1=cv2.imread('straw7.jpg')

image2=image1
image2=cv2.resize(image2,(700,700))
image=image1
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image=cv2.resize(image,(700,700))
image1=image
_,th=cv2.threshold(image,127,255,cv2.THRESH_OTSU,0)
cv2.imshow('imaeg',image)






gray_lap_gray_inv=255-image
height,width=image.shape


cv2.imshow('gray_lap',gray_lap_gray_inv)
gray_lap=cv2.Laplacian(gray_lap_gray_inv,cv2.CV_8UC3,ksize=1)
dilate_lap = cv2.convertScaleAbs(gray_lap)

se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.morphologyEx(gray_lap, cv2.MORPH_CLOSE, se1)
cv2.imshow('mask',mask)

dilate_lap=cv2.dilate(gray_lap,(3,3),iterations=0)

dilate_lap[dilate_lap>10]=255
counting=[]
_,gray_lap=cv2.threshold(dilate_lap,126,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
dilate_lap=cv2.dilate(gray_lap,None,iterations=1)

cv2.imshow('di_lap',dilate_lap)


_, cnts, h = cv2.findContours(dilate_lap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            print('c and len',c,len(c))

            x, y, w, h = cv2.boundingRect(c)
            cc=cv2.drawContours(image1,c,-1,(0,0,255),2)
            ccc=image1-cc
            cv2.imshow('ccc',ccc)
            #cv2.imshow('image', image1)
            cv2.waitKey(0)
            cropped_image = dilate_lap[x+85:x+5 + h+90, y-50:y+5+ w-50]
            cropped_RGB=image2[x+85:x+5 + h+90, y-50:y+5+ w-50]
            cv2.imshow('cropped_rgb',cropped_RGB)
            cv2.waitKey(0)
            se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            cropped_image = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, se1)
            dil_cropped=cv2.erode(cropped_image,None,iterations=1)

            se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            dil_cropped = cv2.morphologyEx(dil_cropped, cv2.MORPH_CLOSE, se1)
            dil_cropped=cv2.dilate(dil_cropped,(3,3),iterations=4)

            cv2.imshow('cropped_image', cropped_image)
            cv2.imshow('dil_cropped',dil_cropped)

            se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            mask = cv2.morphologyEx(dil_cropped, cv2.MORPH_CLOSE, se1)
            mask=cv2.erode(mask,(7,7),iterations=2)
            mask=cv2.dilate(mask,(5,5),iterations=4)
            se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)

            cv2.imshow('Output_mask', mask)
            cv2.waitKey(0)
            mask=~mask
            mask = cv2.copyMakeBorder(mask, top=25, bottom=25, left=25,
                                               right=25,
                                               borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            mask = cv2.copyMakeBorder(mask, top=25, bottom=25, left=25,
                                      right=25,
                                      borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
            _, cnts, h = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                for (i,c) in enumerate(cnts):
                    area=cv2.contourArea(c)
                    print('area',area)
                    x1, y1, w1, h1 = cv2.boundingRect(c)
                    if area>100 and area<400 :
                        counting.append(i)
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(cropped_RGB,(x-55,y-44),(x+w-55,y+h-44),(255,0,5),1)
                        print('number of Straw Counted',len(counting))
                        cv2.putText(cropped_RGB,str(i),(x-55,y-44),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
                cv2.putText(cropped_RGB,'total-'+str(len(counting)),(270,222),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)

                cv2.imshow('im_mask', image1)
                cv2.waitKey(0)
                cv2.imshow('i_mask', cropped_RGB)
                cv2.waitKey(0)
cv2.destroyAllWindows()
if __name__=='__main__':
    print('Executed successfully')





