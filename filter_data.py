import cv2 as cv
import requests
import os
from PIL import Image

def loadFile(url,filename):
    res=requests.get(url)
    with open(filename,"wb") as f:
        f.write(res.text.encode())

def detect(img,eye:cv.CascadeClassifier,name):
    img1=img.copy()
    imr=img.copy()
    face_rect=eye.detectMultiScale(img1,scaleFactor = 1.2,minNeighbors = 5)
    r=0
    for x,y,l,w in face_rect:
        img1=img[y:y+l,x:x+w]
        img1=cv.resize(img1,(250,250))
        save(img1,f"{name}")
        imr=cv.rectangle(imr,(x,y),(x+w,y+l),(255,0,255))
        r+=1
    return imr

def save(img,name):
    r=Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    r.save(os.path.join(target_path,name))
    
if __name__=="__main__":
    # eye_url="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    # loadFile(eye_url,"eye.xml")
    eye=cv.CascadeClassifier("./eye.xml")
    target_path="../Dataset/Dataset/Positive"
    source_path="../Dataset/Unprocess"
    dirs=os.listdir(source_path)
    for val in dirs:
        img=cv.imread(os.path.join(source_path,val))
        img=cv.resize(img,(250,250))
        img=detect(img,eye,val)

