
# import packages
import keras
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Activation
import pandas as pd
from keras.layers import Multiply
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras import Input
from keras import backend as K
import numpy as np
from tensorflow.python.client import device_lib
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
import time
from sklearn.utils import shuffle
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
import albumentations.pytorch
from matplotlib import pyplot as plt
import cv2
from keras import optimizers
import numpy as np
drive.mount("/content/drive")
path = "drive/MyDrive/Task/"
!ls "drive/MyDrive/Task/"

with np.load("drive/MyDrive/Task/dataSet.npz") as data:
    data_set, labels = data['X'], data['y']
data_set,labels = shuffle(data_set,labels,random_state=0)
data_set,labels = shuffle(data_set,labels,random_state=0)
data_set,labels = shuffle(data_set,labels,random_state=0)
data_set,labels = shuffle(data_set,labels,random_state=0)

x,m,n,z = data_set.shape
rows = 10
cols = 4
mylist = np.random.randint(0,x,rows*cols)
for row in range(rows):
    axes=[]
    fig = plt.gcf()
    fig.set_size_inches(10, 20)
    for col in range(cols):
        index = mylist[row*cols+col]
        axes.append( fig.add_subplot(rows, cols, col+1) )
        subplot_title=str(labels[index])
        axes[-1].set_title(subplot_title)
        plt.imshow(data_set[index])
    fig.tight_layout()    
    plt.show()

print(device_lib.list_local_devices())
print(data_set.shape)
print(labels.shape)
X_train, X_test, y_train, y_test = train_test_split(data_set, labels, test_size = 0.1,shuffle = True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def relu_advanced(x):
    return K.relu(x, max_value= 360)
new_input = Input(shape=(512, 512, 3))
model = keras.models.Sequential()
res = ResNet50(input_tensor=new_input, include_top=False)
inside_layer = keras.models.Sequential()
for l in res.layers:
  l.trainable = False
model.add(res)
model.add(Flatten())
model.add(Dense(1,kernel_initializer='random_normal'))
model.add(Activation(relu_advanced))
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error', 'mean_absolute_error'])

history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32)

model.save("drive/MyDrive/Task/model")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("drive/MyDrive/Task/history.csv")

model.evaluate(X_test,y_test)

print(history.history.keys())
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.title('model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("drive/MyDrive/Task/MSE_loss_plot.png")
plt.show()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.title('model absE loss')
plt.ylabel('absE loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("drive/MyDrive/Task/absE_loss_plot.png")
plt.show()

model = keras.models.load_model('drive/MyDrive/Task/model')

def read_img(path):
    img = Image.open(path)
    img = img.resize((512, 512))
    return np.array(img, dtype=np.uint8)
def get_angel(im,model):
  x = tf.convert_to_tensor(im)
  x = tf.expand_dims(x,axis = 0)
  for l in model.layers:
    x = l(x)
  x = x.numpy()
  return -x[0][0]
m,_,_,_ = X_test.shape
size = 5
i = 1
mylist = np.random.choice(range(m),size)
for index in mylist:
  image_arr = X_test[index]
  image = Image.fromarray(np.uint8(image_arr)).convert('RGB')
  rotate_angel = get_angel(image_arr,model)
  rotated_image = image.rotate(rotate_angel)
  fig = plt.gcf()
  fig.set_size_inches(18, 30)
  fig.add_subplot(1, 2, 1)
  plt.imshow(image)
  plt.axis('off')
  plt.title("before rotation")
  fig.add_subplot(1, 2, 2)
  plt.imshow(rotated_image)
  plt.axis('off')
  plt.title("after rotation by the opposite angel of model result")
  plt.savefig("drive/MyDrive/Task/results/result"+str(i)+".png")
  plt.show()
  i+=1
