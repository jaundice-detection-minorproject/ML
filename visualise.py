import matplotlib.pyplot as plt
import pickle
import pandas as pd
data=pickle.load(open("dataset.pkl","rb"))
df=pd.DataFrame(data,columns=list(range(1,7501))+['Target'])
# for i in range(0,models.shape[0],5):
plt.xlabel("Models")
plt.ylabel("Time In ms")
plt.title("Models Vs Time Taken")
plt.scatter(df[1][df['Target']==1],df[7][df['Target']==1])
plt.scatter(df[1][df['Target']==0],df[7][df['Target']==0])
plt.show()