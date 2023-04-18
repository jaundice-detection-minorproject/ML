# import required libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import pickle

# load the dataset from a pickle file
final_data = pickle.load(open("dataset.pkl", "rb"))

# split the data into input features (X) and target variable (y)
X, y = final_data[:, :-1], final_data[:, -1]

# split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=45, test_size=0.3)

# create an ExtraTreesClassifier model and train it on the training set
model = ExtraTreesClassifier(n_estimators=52, random_state=8)
model.fit(train_x, train_y)

# print the accuracy of the model on the testing set
print("Accuracy Of Model Test on %d Data and train on %d is %.2f%%" % (test_x.shape[0], train_x.shape[0], model.score(test_x, test_y) * 100))

# print the number of positive and negative data points in the training and testing sets
print(f"Positive Data for Training: {train_y.tolist().count(1.0)}")
print(f"Positive Data for Testing: {test_y.tolist().count(1.0)}")
print(f"Negative Data for Training: {train_y.tolist().count(0.0)}")
print(f"Negative Data for Testing: {test_y.tolist().count(0.0)}")

# save the trained model to a pickle file
pickle.dump(model, open("model.pkl", "wb"))
