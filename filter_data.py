import cv2 as cv
import requests
import os
from PIL import Image

# A function to download a file from a given URL and save it with a given filename
def loadFile(url, filename):
    res = requests.get(url)
    with open(filename, "wb") as f:
        f.write(res.content)

# A function to detect eyes in an image using a given Haar cascade classifier and draw rectangles around the detected eyes
def detect(img, eye_cascade, name):
    img_copy = img.copy()
    img_rect = img.copy()
    # Detect faces in the image using the Haar cascade classifier
    face_rect = eye_cascade.detectMultiScale(img_copy, scaleFactor=1.2, minNeighbors=5)
    r = 0
    for x, y, l, w in face_rect:
        # Crop the image to the detected face region
        img_copy = img[y:y+l, x:x+w]
        # Resize the cropped image to a fixed size of 250x250 pixels
        img_copy = cv.resize(img_copy, (250, 250))
        # Save the cropped and resized image to a target directory with the given filename
        save(img_copy, f"{name}")
        # Draw a rectangle around the detected face region
        img_rect = cv.rectangle(img_rect, (x, y), (x+w, y+l), (255, 0, 255))
        r += 1
    # Return the image with rectangles drawn around the detected face regions
    return img_rect

# A function to save an image to a given directory with a given filename
def save(img, name):
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    img.save(os.path.join(target_path, name))

if __name__ == "__main__":
    # Download the Haar cascade classifier for eye detection
    eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    loadFile(eye_url, "eye.xml")
    eye_cascade = cv.CascadeClassifier("./eye.xml")
    # Set the target directory where the cropped and resized images will be saved
    target_path = "../Dataset/PreProcess Dataset/Positive"
    # Set the source directory where the original images are located
    source_path = "../Dataset/Raw Dataset"
    dirs = os.listdir(source_path)
    for val in dirs:
        # Load the original image from the source directory
        img = cv.imread(os.path.join(source_path, val))
        # Resize the image to a fixed size of 250x250 pixels
        img = cv.resize(img, (250, 250))
        # Detect eyes in the image using the Haar cascade classifier and save the cropped and resized images to the target directory
        img = detect(img, eye_cascade, val)
