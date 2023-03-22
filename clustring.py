from sklearn.cluster import KMeans,AffinityPropagation
import numpy as np
import pickle
import matplotlib.pyplot as plt
final_data=pickle.load(open("data.pkl","rb"))
X,y=final_data[:,:-1],final_data[:,-1]
model=KMeans(n_clusters=2,random_state=6,tol=0.00001,)
model.fit(X)
labels=model.labels_
c=0
print(labels)
# for i in range(10):
#     if(labels[i]!=y[i]):
#         plt.imshow(np.reshape(X[i],(50,50,3)))
#         plt.show()
#         print(labels[i],y[i])
#         c+=1
print(1-(c/len(labels)),)
