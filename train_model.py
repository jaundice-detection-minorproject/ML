from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
import pickle
final_data=pickle.load(open("dataset.pkl","rb"))
X,y=final_data[:,:-1],final_data[:,-1]
random=RandomForestClassifier(random_state=6)
svc=SVC(random_state=6,probability=True)
vote=VotingClassifier([("svc",svc),("random",random)],voting="soft")
vote.fit(X,y)
pickle.dump(vote,open("model.pkl","wb"))
