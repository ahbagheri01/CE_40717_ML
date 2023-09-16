import pandas as pd
import Preprocess as ps
import pandas as pd
import requests
import argparse
import torch
from torch import nn
import pandas as pd
import sklearn
import numpy as np
from sklearn.utils import shuffle
import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

def visualize_result(y_true,y_pred):
  cm = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cm, range(2), range(2))
  plt.figure(figsize=(10,7))
  sn.set(font_scale=1) 
  sn.heatmap(df_cm, annot=True, annot_kws={"size": 20},fmt="d",cmap="YlGnBu") # font size
  from sklearn.metrics import classification_report
  from sklearn.metrics import f1_score,recall_score,precision_score
  target_names = ['class 0', 'class 1']
  print("f1_score "+" is :{}%".format(f1_score(y_true=y_true , y_pred= y_pred)))
  print("recall_score "+" is :{}%".format(recall_score(y_true=y_true , y_pred= y_pred)))
  print("precision_score "+" is :{}%".format(precision_score(y_true=y_true , y_pred= y_pred)))
  print(classification_report(y_true, y_pred, target_names=target_names))
    
host = '127.0.0.1'

port = 8080

url = f'http://{host}:{port}/invocations'

headers = {
    'Content-Type': 'application/json',
}

data = pd.read_csv('PreModule/datasets/train_dataset.csv')
d = data.to_numpy()
for i in range(20):
    d = shuffle(d, random_state=i)
test_data = pd.DataFrame()
test_data[data.columns] = d[1:20000]
target = test_data["Sale"].to_numpy()

http_data = test_data.to_json(orient='split')

r = requests.post(url=url, headers=headers, data=http_data)
predict = r.text
predict = predict.replace("[","")
predict = predict.replace("]","")
l = predict.split(",")
pred = []
for ind in l:
    pred.append(int(ind))
visualize_result(target.astype(int),pred)
