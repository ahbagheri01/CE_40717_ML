import numpy as np
from PIL import Image
n = 512
shape = (0,n,n,3)
image_array=np.empty(shape, dtype=np.uint8)
lebels = np.empty((0,1), dtype=np.uint8)
from PIL import Image
def read_img(path):
    img = Image.open(path).convert('RGB')
    return img
def add_to_data(arr,label):
    global image_array
    global lebels
    image_array = np.append(image_array, arr, axis=0)
    lebels = np.append(lebels, label, axis=0)
def save_info():
    global image_array
    global lebels
    np.savez_compressed("data.npz", X=image_array, y=lebels)
def rotate_image_and_save(i):
    rotation = [0,45,90,135,180,225,270,315]
    another_rotation = [int(np.random.uniform(1,44,1)[0]),int(np.random.uniform(46,89,1)[0]),
                   int(np.random.uniform(91,134,1)[0]),int(np.random.uniform(136,179,1)[0]),
                   int(np.random.uniform(181,224,1)[0]),int(np.random.uniform(226,269,1)[0]),
                   int(np.random.uniform(271,314,1)[0]),int(np.random.uniform(316,159,1)[0]),
                       int(np.random.uniform(1,44,1)[0]),int(np.random.uniform(46,89,1)[0]),
                   int(np.random.uniform(91,134,1)[0]),int(np.random.uniform(136,179,1)[0]),
                   int(np.random.uniform(181,224,1)[0]),int(np.random.uniform(226,269,1)[0]),
                   int(np.random.uniform(271,314,1)[0]),int(np.random.uniform(316,159,1)[0]),
                       int(np.random.uniform(1,44,1)[0]),int(np.random.uniform(46,89,1)[0]),
                   int(np.random.uniform(91,134,1)[0]),int(np.random.uniform(136,179,1)[0]),
                   int(np.random.uniform(181,224,1)[0]),int(np.random.uniform(226,269,1)[0]),
                   int(np.random.uniform(271,314,1)[0]),int(np.random.uniform(316,159,1)[0])]
    image =read_img("./512*512/"+str(i)+".png")
    for rotate in rotation:
        new_image=np.array(image.rotate(rotate,expand = False), dtype=np.uint8) 
        add_to_data(np.array([new_image],dtype=np.uint8),np.array([[rotate]],dtype=int)) 
    for rotate in another_rotation:
        new_image=np.array(image.rotate(rotate,expand = False), dtype=np.uint8) 
        add_to_data(np.array([new_image],dtype=np.uint8),np.array([[rotate]],dtype=int)) 
for i in range(1,39):
    rotate_image_and_save(i)
save_info()

