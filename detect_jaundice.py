import cv2 as cv
import pickle
import numpy as np
from feature_extraction import getMostDominantColor
import keyboard

# Load the eye cascade classifier and trained model
eye = cv.CascadeClassifier("./eye.xml")
model = pickle.load(open("./model.pkl","rb"))

def detect(img, eye:cv.CascadeClassifier=eye):
    """
    Detects eyes in an image using the specified cascade classifier.
    Returns the image with rectangles drawn around the detected eyes, 
    and a list of the detected eyes.
    """
    img1 = img.copy()
    face_rect = eye.detectMultiScale(img1, scaleFactor=1.2, minNeighbors=5)
    target = []
    for x, y, l, w in face_rect:
        # Extract eye region
        imge = img[y:y+l,x:x+w]
        # Resize eye region to 250x250
        imge = cv.resize(imge,(250,250))
        target.append(imge)
        # Draw rectangle around eye region
        img1 = cv.rectangle(img1, (x,y), (x+w,y+l), (255,0,255))
    return img1, target

def loadCamera():
    """
    Opens the default camera and captures frames until the 's' key is pressed.
    Returns a list of detected eyes.
    """
    video = cv.VideoCapture(0)
    while True:
        _, img = video.read()
        img, target = detect(img, eye)
        cv.imshow("Video",img)
        if keyboard.is_pressed("s"):
            break
        cv.waitKey(1)
    return target

def findJaundice(target):
    """
    Determines if the eyes in the list have jaundice or not, 
    and returns a flag indicating if jaundice was detected and the probability.
    """
    not_have_prob = 0
    have_prob = 0
    for item in target:
        # Get the most dominant color in the eye region
        color, _ = getMostDominantColor(item)
        color = np.array(color)
        color = np.reshape(color, (1,-1))
        # Predict the probability of jaundice using the trained model
        y = model.predict_proba(color)
        not_have_prob += y[0][0]
        have_prob += y[0][1]
    if len(target) != 0:
        not_have_prob /= len(target)
        have_prob /= len(target)
        if not_have_prob >= have_prob:
            return 0, not_have_prob
        else:
            return 1, have_prob
    else:
        return -1, -1

def imageUpload(path, isload=True):
    """
    Reads an image from the specified path and returns a list of detected eyes.
    """
    img = cv.imread(path)
    if isload:
        img, target = detect(img, eye)
        return target
    else:
        return [img]

if __name__=="__main__":
    try:
        while True:
            # Prompt user to choose between camera or uploaded image
            use = int(input("Enter Type 1) Video Camera 2) Upload Image: "))
            if use == 1:
                target = loadCamera()
            elif use == 2:
                img_path = input("Enter Image Path : ")
                target = imageUpload(img_path)

            else:
                print("Invalid Key Enter")
                continue
            
            # Determine if jaundice is present and output the result
            output, probability = findJaundice(target)
            if output == 0:
                print("You Don't Have Jaundice with Accuracy %.2")
            elif(output==1):
                print("You Have Jaundice with Accuracy %.2f%%"%(probability*100))
            else:
                print("Eye Not Found")
            break
    except:
        print("Image Not Found")
        