'''
Steps To Run Then Code 

1) In current Directory First git clone https://github.com/jaundice-detection-minorproject/Raw-Dataset.git

2) Then create a python file and put this code.

3) Install the Required Libraries
    - scikit-learn==1.0.2
    - numpy==1.21.6
    - opencv-python==4.7.0.72
    - pillow==9.4.0
    - keyboard==0.13.5

4) Then Run This File Using Command prompt 
    - py './file_name.py'
'''

# Import Required Libraries
import os
import numpy as np
import keyboard
import cv2 as cv
import requests
from PIL import Image
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Define a class JaundiceDetectionSystem
class JaundiceDetectionSystem:

    def __init__(self):
        # URL for the trained classifier to detect eyes
        self.eye_url="https://onlinewindow.glitch.me/access/other/1/tkarpjag7m"
        # File name for the trained eye classifier
        self.eye_filename="./eyeDetection.xml"
        # Directory to save the pre-processed images
        self.target_dir="./PreProcess Dataset"
        # Name of the final dataset
        self.final_dataset_name="dataset.pkl"
        # Directory for the raw dataset
        self.raw_dataset="./Raw-Dataset"
        # Path to save the trained model
        self.model_path="./model.pkl"

        # Check if the raw dataset directory exists
        if(not os.path.exists(self.raw_dataset)):
            # Raise an exception if the directory does not exist
            raise Exception(f"Load Raw Dataset First using command 'git clone https://github.com/jaundice-detection-minorproject/Raw-Dataset.git'")  
        print("Raw Dataset Loaded")

        # Check if the trained eye classifier file exists
        if(not os.path.exists(f"./{self.eye_filename}")):
            # If it does not exist, try to load it from the given URL
            if(not self.loadFile(url=self.eye_url,file_path=self.eye_filename)):
                # If the file cannot be loaded, raise an exception
                raise Exception("Failed To Load Eye Classifier")
        print("Eye Classifier Loaded")

        # Load the eye classifier model from the trained file
        self.eye_cascade = cv.CascadeClassifier(f"./{self.eye_filename}")

        # Check if the pre-processed images directory exists
        if(not os.path.exists(self.target_dir)):
            # If it does not exist, extract the eyes from raw images
            if(not self.extractEyeFromFaceRaw()):
                # If the eyes cannot be extracted, raise an exception
                raise Exception("Failed To Load Eye Detection")
        print("Eye Detection Complete")

        # Check if the final dataset exists
        if(not os.path.exists(self.final_dataset_name)):
            # If it does not exist, extract features from the pre-processed images
            self.feature_extraction()
        print("Feature Extraction Complete")

    # Method to load a file from a given URL and save it to the specified path
    def loadFile(self,url,file_path):
        try:
            res = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(res.content)
            return True
        except Exception as e:
            print(e)
            return False

    # Method to detect eyes from a given image
    def detectEyeFromImage(self,image:np.ndarray):
        try:
            target=[]
            # Detect the eyes in the image using the trained eye classifier
            face_rect = self.eye_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
            # Crop the eye region and add it to the list of target images
            for x, y, l, w in face_rect:
                target.append(cv.resize(image[y:y+l, x:x+w], (250, 250)))
            return target
        except Exception as e:
            print(e)
            return []

    def extractEyeFromFaceRaw(self):
        """
        Extracts eyes from raw images and saves them to a folder.
        """
        try:
            # Get a list of directories in the raw dataset path
            dirs = os.listdir(self.raw_dataset)
            for className in dirs:
                # Skip hidden files and directories
                if(className==".git"):continue
                counter=1
                # Iterate over the images in the current directory
                for imageName in os.listdir(os.path.join(self.raw_dataset,className)):
                    # Skip hidden files and directories
                    if(imageName==".git"):continue
                    # Load the image from file
                    path=os.path.join(self.raw_dataset, className,imageName)
                    img = cv.imread(path)
                    # Resize the image to 250x250 for consistency
                    img = cv.resize(img, (250, 250))
                    # Detect eyes in the image
                    target = self.detectEyeFromImage(img)
                    # Iterate over the detected eyes
                    for img in target:
                        # Save each detected eye to a file
                        self.saveImage(img,f"{counter}.jpg")
                        counter+=1
            return True
        except Exception as e:
            # If an error occurs, print the error message and return False
            print(e)
            return False
    
    def getMostDominantColor(self,image:np.ndarray):
        # resize the image to 50x50
        image=cv.resize(image,(50,50))
        # convert color space from BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # reshape the image to a 1D array of pixels
        pixel_vals = image.reshape((-1,3)) 
        # convert pixel values to float32 data type
        pixel_vals = np.float32(pixel_vals)
        # define termination criteria for k-means clustering
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.75)
        # perform k-means clustering on pixel values
        retval, labels, centers = cv.kmeans(pixel_vals, 5, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # convert cluster centers to 8-bit unsigned integers
        centers = np.uint8(centers)
        # create segmented data using cluster labels and centers
        segmented_data = centers[labels.flatten()]
        # reshape the segmented data to the original image shape
        segmented_image = segmented_data.reshape((image.shape))
        # set the image to the segmented image
        image=segmented_image
        # convert color space from BGR to RGB to HSV
        hsv=cv.cvtColor(cv.cvtColor(image,cv.COLOR_BGR2RGB),cv.COLOR_RGB2HSV)
        # normalize pixel values to be between 0 and 1
        hsv=cv.cvtColor(hsv,cv.COLOR_BGR2RGB)/255
        # return flattened HSV values and original HSV image
        return hsv.flatten().tolist()
    

    def feature_extraction(self):
        # create a dictionary to store the features for each class
        data={"Positive":[],"Negative":[]}
        # get a list of directories in the target path
        dirs=os.listdir(self.target_path)
        # iterate over each directory in the target path
        for item in dirs:
            if(item==".git"):continue
            # skip directories that are not "Positive" or "Negative"
            if(item!="Positive" and item!="Negative"):continue
            print(item)
            # get a list of images in the current directory
            for val in os.listdir(os.path.join(self.target_path,item)):
                if(val==".git"):continue
                # read the image file
                image=cv.imread(os.path.join(self.target_path,item,val))
                # get the most dominant color in the image
                arr,_=self.getMostDominantColor(image)
                # set the label for the image (0 for negative, 1 for positive)
                r=0
                if(item=="Positive"):r=1
                # append the label to the feature vector
                arr.append(r)
                # append the feature vector to the data dictionary
                data[item].append(arr)
        # convert the feature vectors for each class to numpy arrays
        data["Positive"]=np.array(data['Positive'])
        data["Negative"]=np.array(data['Negative'])
        # concatenate the positive and negative feature vectors into one array
        final_data=np.append(data["Positive"],data["Negative"],axis=0)
        # save the feature data to a pickle file
        pickle.dump(final_data,open(self.final_dataset_name,"wb"))
    
    def train_model(self,isTrain):
        # load the dataset from a pickle file
        final_data = pickle.load(open(self.final_dataset_name, "rb"))

        # split the data into input features (X) and target variable (y)
        X, y = final_data[:, :-1], final_data[:, -1]

        # split the data into training and testing sets
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=45, test_size=0.3)

        if(isTrain):
            # create an ExtraTreesClassifier model and train it on the training set
            model = ExtraTreesClassifier(n_estimators=52, random_state=8)
            model.fit(train_x, train_y)
            pickle.dump(model,open(self.model_path,"wb"))
        else:
            model = pickle.load(open(self.model_path,"rb"))
        # print the accuracy of the model on the testing set
        return model.score(test_x, test_y) * 100

    def loadCamera(self):
        """
        Opens the default camera and captures frames until the 's' key is pressed.
        Returns a list of detected eyes.
        """
        video = cv.VideoCapture(0)
        print("press S to stop")
        while True:
            _, img = video.read()
            target = self.detectEyeFromImage(img)
            cv.imshow("Video",img)
            if keyboard.is_pressed("s"):
                break
            cv.waitKey(1)
        return target

    def imageUpload(self,path):
        """
        Reads an image from the specified path and returns a list of detected eyes.
        """
        img = cv.imread(path)
        target = self.detectEyeFromImage(img)
        return target
    
    def findJaundice(self,target,model):
        """
        Determines if the eyes in the list have jaundice or not, 
        and returns a flag indicating if jaundice was detected and the probability.
        """
        not_have_prob = 0
        have_prob = 0
        for item in target:
            # Get the most dominant color in the eye region
            color = self.getMostDominantColor(item)
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
        
    def detectJaundice(self):
        print(self.train_model(isTrain=not os.path.exists(self.model_path)))
        try:
            model = pickle.load(open(self.model_path,"rb"))
            while True:
                # Prompt user to choose between camera or uploaded image
                use = int(input("Enter Type 1) Video Camera 2) Upload Image: "))
                if use == 1:
                    target = self.loadCamera()
                elif use == 2:
                    img_path = input("Enter Image Path : ")
                    target = self.imageUpload(img_path)
                else:
                    print("Invalid Key Enter")
                    continue
                
                # Determine if jaundice is present and output the result
                output, probability = self.findJaundice(target,model)
                if output == 0:
                    print("You Don't Have Jaundice with Accuracy %.2f%%"%(probability*100))
                elif(output==1):
                    print("You Have Jaundice with Accuracy %.2f%%"%(probability*100))
                else:
                    print("Eye Not Found")
                break
        except Exception as e:
            print(e)
            print("Image Not Found")
    
    def saveImage(self,img, name):
        # convert image to PIL Image object with RGB color format
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        # create directory if it doesn't already exist
        if(not os.path.exists(self.target_dir)):
            os.mkdir(self.target_dir)
        
        # save image to directory with given name
        img.save(os.path.join(self.target_dir, name))


# Run the JaundiceDetectionSystem class
if __name__ == "__main__":
    obj = JaundiceDetectionSystem()
    obj.detectJaundice()