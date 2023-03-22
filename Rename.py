import os
from detect_jaundice import findJaundice,imageUpload
def call(item):
    path=f"../Dataset/Raw Dataset/{item}"

    dirs=os.listdir(path)

    for i in range(len(dirs)):
        os.rename(os.path.join(path,dirs[i]),os.path.join(path,f"source_raw{i+1}.jpg"))

def check(item,tag):
    path=f"../Dataset/PreProcess Dataset/{item}"
    dirs=os.listdir(path)
    x=0
    total=len(dirs)
    for item in dirs:
        target=imageUpload(os.path.join(path,item),False)
        t,_=findJaundice(target)
        if(t==tag):
            x+=1
        elif(t==-1):
            total-=1
    print(x,total,total-x)  

check("Negative",0)