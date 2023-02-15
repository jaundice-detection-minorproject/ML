from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pandas as pd
final_data=pd.read_csv("dataset.csv").to_numpy()
X,y=final_data[:,:-1],final_data[:,-1]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=45)
model=LogisticRegression(max_iter=1000)
model_gv=GaussianNB()
mlp=MLPClassifier(hidden_layer_sizes=(1000,))
model.fit(train_x,train_y)
model_gv.fit(train_x,train_y)
mlp.fit(train_x,train_y)
print(model_gv.score(test_x,test_y))
print(model.score(test_x,test_y))
print(mlp.score(test_x,test_y))
