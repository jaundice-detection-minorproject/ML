import cv2 as cv
import pickle
import numpy as np
from feature_extraction import getMostDominantColor
import keyboard
import matplotlib.pyplot as plt

eye=cv.CascadeClassifier("./eye.xml")
model=pickle.load(open("model.pkl","rb"))

def detect(img,eye:cv.CascadeClassifier):
    img1=img.copy()
    face_rect=eye.detectMultiScale(img1,scaleFactor = 1.2,minNeighbors = 5)
    target=[]
    for x,y,l,w in face_rect:
        imge=img[y:y+l,x:x+w]
        imge=cv.resize(imge,(250,250))
        target.append(imge)
        img1=cv.rectangle(img1,(x,y),(x+w,y+l),(255,0,255))
    return img1,target

def loadCamera():
    video=cv.VideoCapture(0)
    while(True):
        _,img=video.read()
        img,target=detect(img,eye)
        cv.imshow("Video",img)
        if(keyboard.is_pressed("s")):
            break
        cv.waitKey(1)
    return target

def findJaundice(target):
    not_have_prob=0
    have_prob=0
    for item in target:
        color,_=getMostDominantColor(item)
        color=np.array(color)
        color=np.reshape(color,(1,-1))
        y=model.predict_proba(color)
        not_have_prob+=y[0][0]
        have_prob+=y[0][1]
    if(len(target)!=0):
        not_have_prob/=len(target)
        have_prob/=len(target)
        if(not_have_prob>=have_prob):
            print("You Don't Have Jaundice with Accuracy : %.2f%%"%(not_have_prob*100))
            return 0
        else:
            print("You Have Jaundice with Accuracy : %.2f%%"%(have_prob*100))
            return 1
    else:
        print("Eye Not Found")
        return -1

def imageUpload(path):
    img=cv.imread(path)
    img,target=detect(img,eye)
    return target
if __name__=="__main__":
    while(True):
        use=int(input("Enter Type 1) Video Camera 2) Upload Image: "))
        if(use==1):
            target=loadCamera()
        elif(use==2):
            img_path=input("Enter Image Path : ")
            target=imageUpload(img_path)
        else:
            print("Invalid Key Enter")
            continue
        findJaundice(target)
        break
        