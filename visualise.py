import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("output.csv")

models=data["Model"]

accuracy=data['Time Taken']

# for i in range(0,models.shape[0],5):
plt.xlabel("Models")
plt.ylabel("Time In ms")
plt.title("Models Vs Time Taken")
plt.xticks(rotation=90)
plt.bar(models,accuracy)
plt.show()