from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
final_data=pd.read_csv("dataset.csv").to_numpy()
X,y=final_data[:,:-1],final_data[:,-1]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=45)
random=RandomForestClassifier(random_state=6)
mlp=MLPClassifier(hidden_layer_sizes=(110,),random_state=54)
svc=SVC(random_state=25,probability=True)
vote=VotingClassifier([("svc",svc),("random",random)],voting="soft")
vote.fit(train_x,train_y)
print(vote.score(test_x,test_y))
# print(random.score(test_x,test_y),mlp.score(test_x,test_y),svc.score(test_x,test_y),vote.score(test_x,test_y))
pickle.dump(vote,open("model.pkl","wb"))


# hard 54 25

# soft 6 6