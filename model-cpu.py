
# coding: utf-8

# In[1]:


import pandas as pd

FWD = './drive-data/june-23/'
BACK ='./drive-data/june-23-r/'

df_data_1 = pd.read_csv(FWD + 'driving_log.csv',header=None)
df_data_2 = pd.read_csv(BACK + 'driving_log.csv',header=None)

print(len(df_data_1),len(df_data_2))

df_data_1 = df_data_1[:-1000]
df_data_2 = df_data_2[:-300]

print(len(df_data_1),len(df_data_2))


# In[2]:


df_data_1['dir'] = [FWD for _ in range(len(df_data_1))]
df_data_2['dir'] = [BACK for _ in range(len(df_data_2))]


# In[5]:


df_data = pd.concat([df_data_1,df_data_2],sort=False)


# In[6]:


df_data.columns=['center','left','right','steering_angle','throttle','break','speed','dir']


# In[17]:


df_data[-5:-1].head()


# In[18]:


# import matplotlib.pyplot as plt
# import numpy as np

def resolve_name(name,dirname):
    return dirname + 'IMG/{}'.format(name.split('/')[-1])

# import cv2
# for center,dirname in zip(df_data['center'][:5],df_data['dir'][:5]):
#     img = cv2.imread(resolve_name(center,dirname))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = cv2.resize(img,(96,96))
#     print(img.shape)
#     plt.imshow(img)
#     plt.show()
    


# # In[19]:


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


# In[20]:


import os
import csv

samples = df_data.values


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
# import numpy as np

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
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # data augmentation .. flipped image
                flp_center_image = np.fliplr(center_image)
                flp_center_angle = -center_angle


            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format


# In[21]:


from keras.models import Sequential
from keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,Lambda,Cropping2D,Dropout

model = Sequential()

model.add(Cropping2D(cropping=((80,0), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(80, 320, 3),
        output_shape=(80, 320,3)))
model.add(Conv2D(8,(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(.75))
model.add(Dense(128))
model.add(Dropout(.75))
model.add(Dense(64))
model.add(Dropout(.75))
model.add(Dense(1))

model.summary()


# In[25]:

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='./models/best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True,verbose=1)


from math import ceil
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=20, verbose=1,callbacks=[checkpoint])


# In[ ]:

model.load_weights('./models/best.h5')
model.save('model.h5')


# In[52]:


# plt.figure(figsize=(10,6))
# plt.plot(history.history["loss"],label="Loss")
# plt.plot(history.history["val_loss"],label="Validation Loss")
# plt.legend()
# plt.show()

