'''
Author:  Ren Silva
Purpose: Traing neural network model to drive car in Udacity's simulation track
'''

# imports
import pandas as pd
import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,Lambda,Cropping2D,Dropout
from keras.callbacks import ModelCheckpoint
from math import ceil


# files for training running in the track forward
FWD = './drive-data/june-23/'

# files for training running in the track backwards
BACK ='./drive-data/june-23-r/'

# read files usin Pandas
df_data_1 = pd.read_csv(FWD + 'driving_log.csv',header=None)
df_data_2 = pd.read_csv(BACK + 'driving_log.csv',header=None)

# confirm length of both files
print(len(df_data_1),len(df_data_2))

# ignore tail of both files (this is normally when I crashed)
df_data_1 = df_data_1[:-1000]
df_data_2 = df_data_2[:-300]

# pring length after trimming ends
print(len(df_data_1),len(df_data_2))

# assign foler to each file
df_data_1['dir'] = [FWD for _ in range(len(df_data_1))]
df_data_2['dir'] = [BACK for _ in range(len(df_data_2))]

# now concatenate both 
df_data = pd.concat([df_data_1,df_data_2],sort=False)

# rename columns to something more readable
df_data.columns=['center','left','right','steering_angle','throttle','break','speed','dir']

# print tail of df_data datagrame
df_data[-5:-1].head()

# ===> this was used in the notebook, but is not used in the python script
# ===> leaving it here for compatibility

# import matplotlib.pyplot as plt
# import numpy as np


def resolve_name(name,dirname):
    return dirname + 'IMG/{}'.format(name.split('/')[-1])

# ===> this was used in the notebook, but is not used in the python script
# ===> leaving it here for compatibility

# import cv2
# for center,dirname in zip(df_data['center'][:5],df_data['dir'][:5]):
#     img = cv2.imread(resolve_name(center,dirname))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = cv2.resize(img,(96,96))
#     print(img.shape)
#     plt.imshow(img)
#     plt.show()
    
# for center,dirname in zip(df_data['center'][-30:-5],df_data['dir'][-10:-5]):
#     try:
#         img = cv2.imread(resolve_name(center,dirname))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img = cv2.resize(img,(96,96))
#         print(img.shape)
#         plt.imshow(img)
#         plt.show()
#     except:
#         pass


# remove heading from dataframe .. in prapration for 
samples = df_data.values


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# data generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+(batch_size)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = resolve_name(batch_sample[0],batch_sample[7])
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # data augmentation .. adding flipped image
                flp_center_image = np.fliplr(center_image)
                flp_center_angle = -center_angle

                images.append(flp_center_image)
                angles.append(flp_center_angle)                

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# creating train and validation generators
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# define model
model = Sequential()

# crop top part of the image
model.add(Cropping2D(cropping=((80,0), (0,0)), input_shape=(160,320,3)))

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(80, 320, 3),
        output_shape=(80, 320,3)))

# now build a "lenet" inspired CNN
model.add(Conv2D(8,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256),activation='relu')
model.add(Dropout(.75))
model.add(Dense(128),activation='relu')
model.add(Dropout(.75))
model.add(Dense(64),activation='relu')
model.add(Dropout(.75))
model.add(Dense(1))  # no activation -> this is a regression problem

# print model definition
model.summary()

# define checkpoint to save weights for model with lowest validation loss
checkpoint = ModelCheckpoint(filepath='./models/best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True,verbose=1)

# compile model
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=20, verbose=1,callbacks=[checkpoint])

# load best weights for best model
model.load_weights('./models/best.h5')

# now save model
model.save('model-new.h5')

# ===> this was used in the notebook, but is not used in the python script
# ===> leaving it here for compatibility

# plt.figure(figsize=(10,6))
# plt.plot(history.history["loss"],label="Loss")
# plt.plot(history.history["val_loss"],label="Validation Loss")
# plt.legend()
# plt.show()

