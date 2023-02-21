import cv2 as cv
import numpy as np
import os
import pickle

def getMostDominantColor(image):
    image=cv.resize(image,(50,50))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3)) 
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.75)
    retval, labels, centers = cv.kmeans(pixel_vals, 5, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
    image=segmented_image
    hsv=cv.cvtColor(cv.cvtColor(image,cv.COLOR_BGR2RGB),cv.COLOR_RGB2HSV)
    hsv=cv.cvtColor(hsv,cv.COLOR_BGR2RGB)/255
    return hsv.flatten().tolist(),hsv
    
def feature_extraction():
    target_path="../Dataset/PreProcess Dataset"
    data={"Positive":[],"Negative":[]}
    dirs=os.listdir(target_path)
    labels=[]
    for item in dirs:
        if(item!="Positive" and item!="Negative"):continue
        print(item)
        for val in os.listdir(os.path.join(target_path,item)):
            image=cv.imread(os.path.join(target_path,item,val))
            arr,_=getMostDominantColor(image)
            r=0
            if(item=="Positive"):r=1
            arr.append(r)
            data[item].append(arr)

    data["Positive"]=np.array(data['Positive'])
    data["Negative"]=np.array(data['Negative'])
    final_data=np.append(data["Positive"],data["Negative"],axis=0)
    pickle.dump(final_data,open("dataset.pkl","wb"))
    # data=pd.DataFrame(final_data)
    # data.to_csv("dataset.csv",index=False)
if __name__=="__main__":
    feature_extraction()

