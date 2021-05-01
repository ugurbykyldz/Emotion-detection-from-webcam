import keras
from keras.models import Sequential 
from keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Dropout,Flatten,BatchNormalization
from keras.preprocessing.image import load_img, img_to_array

from glob import glob 
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def okuResim(path):
    imgs = glob(path +"//*")
    array = np.zeros([len(imgs),48*48*1])
    i=0
    for im in imgs:
        img = load_img(im,grayscale=True,target_size=(48,48))
        img = img_to_array(img)
        img = img.flatten()
        array[i,:] = img
        i+=1
    return array


#angry
x_train_path_0 = "images/angry"
x_train_0 = okuResim(x_train_path_0)
print(x_train_0.shape)
y_train_0 = np.zeros([len(x_train_0),1])
print(y_train_0.shape)

#happy
x_train_path_1 = "images/happy"

x_train_1 = okuResim(x_train_path_1)
print(x_train_1.shape)
y_train_1 = np.ones([len(x_train_1),1])
print(y_train_1.shape)

#sad
x_train_path_2 = "images/sad"
x_train_2 = okuResim(x_train_path_2)
print(x_train_2.shape)
y_train_2 = 2 * np.ones([len(x_train_2),1])
print(y_train_2.shape)



#Birleştirme
x = np.concatenate((x_train_0,x_train_1,x_train_2),0)
print(x.shape)
y = np.concatenate((y_train_0,y_train_1,y_train_2),0)
print(y.shape)

x = x.astype(np.float32)
y =  y.astype(np.float32)

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True) 


x_train = np.reshape(x_train,(-1,48,48,1))
x_test = np.reshape(x_test,(-1,48,48,1))

import matplotlib.pyplot as plt
plt.imshow(x_train[4058,:])
plt.axis("off")
plt.show()


#0-1
x_train = x_train / 255.0
x_test = x_test / 255.0



# one hot encoidng
from keras.utils import to_categorical
y_train = to_categorical(y_train,3)
y_test = to_categorical(y_test,3)



input_shape = (48,48,1)
num_of_class = 3



#model oluşturma

# 1.katman

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape = input_shape, padding = "same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (5,5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))



# 2.katman

model.add(Conv2D(128, (5,5),  padding = "same"))
model.add(Activation("relu"))
model.add(Conv2D(128, (5,5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))


# 3.katman

model.add(Conv2D(256, (5,5), padding = "same"))
model.add(Activation("relu"))
model.add(Conv2D(256, (5,5), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))


#vektor haline çevirme

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

#cıktı

model.add(Dense(num_of_class))
model.add(Activation("softmax"))


#model derleme

model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = "adam")

model.summary()




# train
hist = model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=20,
          validation_data=(x_test,y_test),
          shuffle = True
          )



























