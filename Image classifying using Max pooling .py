#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ## Mood classfication using CNN (HAPPY / SAD)
### STEPS -
- Create 3 folder in your desktop 
- Training, Testing, Validation
- Inside training create 2 folder as happy or not happy
- paste all the photo in testing part 
# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
#image data generator is the package to lable the images & it will automatically lable all the images


# In[3]:


img = image.load_img(r'D:\Data Science with AI\14th-feb-2024\image classification\training\happy\5.jpeg')


# In[4]:


plt.imshow(img)


# In[5]:


i1 = cv2.imread(r'D:\Data Science with AI\14th-feb-2024\image classification\training\happy\5.jpeg')
i1
# 3 dimension metrics are created for the image
# the value ranges from 0-255


# In[6]:


i1.shape
# shape of your image height, weight, rgb


# In[7]:


train = ImageDataGenerator(rescale = 1/255)
validataion = ImageDataGenerator(rescale = 1/255)
# to scale all the images i need to divide with 255
# we need to resize the image using 200, 200 pixel


# In[8]:


train_dataset = train.flow_from_directory(r'D:\Data Science with AI\14th-feb-2024\image classification\training',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
validataion_dataset = validataion.flow_from_directory(r'D:\Data Science with AI\14th-feb-2024\image classification\validation',
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')


# In[9]:


train_dataset.class_indices


# In[10]:


train_dataset.classes


# In[11]:


# now we are applying maxpooling 

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filtr we applied hear
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),    
                                    #                       
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2), 
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation= 'sigmoid')
                                    ]
                                    )


# In[12]:


model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy']
              )


# In[13]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 30,
                     validation_data = validataion_dataset)


# In[25]:


dir_path = r'D:\Data Science with AI\14th-feb-2024\image classification\testing'
for i in os.listdir(dir_path ):
    print(i)
    #img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
   # plt.imshow(img)
   # plt.show()


# In[26]:


dir_path = r'D:\Data Science with AI\14th-feb-2024\image classification\testing'
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()


# In[29]:


dir_path = r'D:\Data Science with AI\14th-feb-2024\image classification\testing'
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
        
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    
    val = model.predict(images)
    if val == 0:
        print( ' i am  happy')
    else:
        print('i am not happy')


# In[30]:


dir_path = r'D:\Data Science with AI\14th-feb-2024\image classification\testing'


plt.figure(figsize=(15, 15))
columns = 3
rows = len(os.listdir(dir_path)) // columns + 1

for i, filename in enumerate(os.listdir(dir_path)):
    img_path = os.path.join(dir_path, filename)
    img = image.load_img(img_path, target_size=(200, 200))
    plt.subplot(rows, columns, i + 1)
    plt.imshow(img)
    plt.axis('off')  # Disable axis

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    val = model.predict(images)
    if val == 0:
        prediction = 'I am happy'
    else:
        prediction = 'I am not happy'
    plt.title(prediction)

plt.tight_layout()
plt.show()


# In[ ]:




