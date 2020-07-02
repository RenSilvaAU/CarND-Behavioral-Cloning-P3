
# coding: utf-8

# In[1]:


import pandas as pd

FWD = './drive-data/june-23/'
BACK ='./drive-data/june-23-r/'

df_data_1 = pd.read_csv(FWD + 'driving_log.csv',header=None)
df_data_2 = pd.read_csv(BACK + 'driving_log.csv',header=None)


# In[2]:


df_data_1['dir'] = [FWD for _ in range(len(df_data_1))]
df_data_2['dir'] = [BACK for _ in range(len(df_data_2))]


# In[3]:


df_data = pd.concat([df_data_1,df_data_2],sort=False)


# In[4]:


df_data.columns=['center','left','right','steering_angle','throttle','break','speed','dir']


# In[5]:


df_data[-5:-1].head()


# In[7]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
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
    


# In[8]:


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


# In[9]:


import os
import csv

samples = df_data.values


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = resolve_name(batch_sample[0],batch_sample[7])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format


# In[10]:


# Set a couple flags for training - you can ignore these for now
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically

# Loads in InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2


# In[11]:



# Using Inception with ImageNet pre-trained weights
inception = InceptionResNetV2(weights=weights_flag, include_top=False,
                        input_shape=(row,col,3))


# In[31]:


for layer in inception.layers:
    layer.trainable = False


# In[35]:


from keras.models import Sequential
from keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,Lambda,GlobalAveragePooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col,ch)))


# In[36]:


model.add(inception)


# In[37]:


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1))


# In[38]:


model.summary()


# In[39]:


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='./models/best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)


# In[ ]:


from math import ceil
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=2, verbose=1)


# In[23]:


from keras.models import load_model


# In[ ]:


# model = models.load('./models/best.h5')


# In[ ]:


model.save('model-tl.h5')


# In[52]:


# plt.figure(figsize=(10,6))
# plt.plot(history.history["loss"],label="Loss")
# plt.plot(history.history["val_loss"],label="Validation Loss")
# plt.legend()
# plt.show()

