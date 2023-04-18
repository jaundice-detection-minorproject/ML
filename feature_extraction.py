import cv2 as cv
import numpy as np
import os
import pickle

# function to get the most dominant color in an image
def getMostDominantColor(image):
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
    return hsv.flatten().tolist(),hsv

# function to extract features from images and save to a pickle file
def feature_extraction():
    # specify the path to the preprocessed dataset
    target_path="../Dataset/PreProcess Dataset"
    # create a dictionary to store the features for each class
    data={"Positive":[],"Negative":[]}
    # get a list of directories in the target path
    dirs=os.listdir(target_path)
    # iterate over each directory in the target path
    for item in dirs:
        # skip directories that are not "Positive" or "Negative"
        if(item!="Positive" and item!="Negative"):continue
        print(item)
        # get a list of images in the current directory
        for val in os.listdir(os.path.join(target_path,item)):
            # read the image file
            image=cv.imread(os.path.join(target_path,item,val))
            # get the most dominant color in the image
            arr,_=getMostDominantColor(image)
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
    pickle.dump(final_data,open("dataset.pkl","wb"))

# main function to call feature_extraction()
if __name__=="__main__":
    feature_extraction()
