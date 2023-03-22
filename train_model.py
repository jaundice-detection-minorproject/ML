from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from lazypredict.Supervised import LazyClassifier
final_data=pickle.load(open("dataset.pkl","rb"))
X,y=final_data[:,:-1],final_data[:,-1]
train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=45,test_size=0.2)
random=RandomForestClassifier(random_state=6,max_depth=10,n_estimators=200,min_samples_split=2,min_samples_leaf=1,max_features="log2",verbose=0)
svc=SVC(random_state=6,probability=True,kernel="rbf",tol=0.001,max_iter=300,decision_function_shape="ovo")
vote=VotingClassifier([("svc",svc),("random",random)],voting="soft")
vote.fit(train_x,train_y)
req=LazyClassifier()
models,predict=req.fit(train_x,test_x,train_y,test_y)
models.to_csv("output.csv")
print("Accuracy Of Model Test on %d Data and train on %d is %.2f%%"%(test_x.shape[0],train_x.shape[0],vote.score(test_x,test_y)*100))
print(f"Positive Data for Training : {train_y.tolist().count(1.0)}")
print(f"Positive Data for Testing : {test_y.tolist().count(1.0)}")
print(f"Negative Data for Training : {train_y.tolist().count(0.0)}")
print(f"Negative Data for Testing : {test_y.tolist().count(0.0)}")
pickle.dump(vote,open("model.pkl","wb"))