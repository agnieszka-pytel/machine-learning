#!/usr/bin/env python
# coding: utf-8

# # CNN for chest X-ray classification
# 
# *Project for Machine Learning course during Msc in Informatics and Econometrics at AGH University of Science and Technology in Cracow*
# 
# *May 2019*
# 
# ## 1. Introduction
# 
# Machine learning is gaining more and more popularity due to the possibility of using it in various fields, ease of implementation and very good results of their application for a given problem. Thanks to the ever-increasing computing power of computers, machine learning has developed strongly in recent years - especially deep learning,  which is currently one of the most frequently used tools in work with images, video or text. The use of deep learning to classify images often results in better results than in the case of human-made classification. This is especially important in the context of medical data analysis, where correct or incorrect classification is often a matter of life and death of the patient.
# 
# The aim of this project is to create a deep convolutional network for classifying X-ray images to detect a patient's pneumonia.

# ## 2. Data analysis and preprocessing

# * The dataset for the problem consists of 5856 images in JPEG format and a CSV file containing image labels.

# In[1]:


### Necessary imports and settings ###
import numpy as np
import pandas as pd
from os import listdir, mkdir, path
from matplotlib import image, pyplot

### Setting the seed to get reproducible results (works on CPU only) ###
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

root_dir = "../input/all/All/" # original path for the dataset


# The file named 'GTruth.csv' contains true labels of the images indicating the diagnosis:
# * **1** for healthy lungs
# * **0** for pneumonia.
# 
# Below are the first 5 lines of the file.  
# Information in this file allows to determine the distribution of data between classes.

# In[2]:


### File with true labels ###
gtruth = pd.read_csv(root_dir+'GTruth.csv', header = 0)
print(gtruth.head())

### Distribution of data in classes ###
labels = ('Healthy','Pneumonia')
counts = (np.count_nonzero(gtruth["Ground_Truth"]), np.count_nonzero(gtruth["Ground_Truth"]==0))

pyplot.pie(counts, labels = labels, colors = ('#70db70','#ff9966'), autopct='%1.f%%')
pyplot.axis('equal')
pyplot.show()


# As shown in the figure above, the dataset is unbalanced: only 27% of images indicate the case of pneumonia and there are 73% of images of healthy lungs - the ratio is almost 1:3. 

# In[3]:


### Loading dataset sample (first 10 images) ###
sample_size = 10
img_sample = [(fname, image.imread(root_dir + fname)) for i,fname in enumerate(listdir(root_dir)) if fname.endswith('.jpeg') and i < sample_size]

### Show sample images along with their filename and shape ###
rows, columns = 2, 5
fig = pyplot.figure(figsize=(16, 8))
for index, img in enumerate(img_sample):
    print('Filename:',img[0],' shape:',img[1].shape)
    fig.add_subplot(rows, columns, index+1)
    # matplotlib displays single-channel images in greenish color, so it's necessary to choose a gray colormap
    pyplot.imshow(img[1], cmap = pyplot.cm.gray)
pyplot.subplots_adjust(left = 1, right = 2)
pyplot.show()


# As presented in the output above, images come with different sizes (first two elements of "shape") and even different number of channels (the third element of "shape", if not present it means single channel), so it's necessary to transform the data before using it to train the model. 
# 
# For the purpose of Keras method *flow_from_directory* the data needs to be split into training, validation and test sets, which in this case are saparate directories. The main obstacle is that data is not catalogued in classes but mixed in one directory. 
# 
# Moreover, the order of filenames in GTruth.csv does not match the order of files in the directory, so the lists of filenames and corresponding labels are determined and stored in lists *class0* and *class1*.

# In[4]:


### Creating directories for train-validation-test sets ###
base_dir = '../pneumonia-chest-x-ray'

try:
    mkdir(base_dir)
except FileExistsError:
    pass

train_dir = path.join(base_dir, 'train')
validation_dir = path.join (base_dir, 'validation')
test_dir = path.join(base_dir, 'test')

try:
    mkdir(train_dir)
    mkdir(validation_dir)
    mkdir(test_dir)
except FileExistsError:
    pass
    
train_1_dir = path.join(train_dir,'healthy')
train_0_dir = path.join(train_dir,'pneumonia')
validation_1_dir = path.join(validation_dir,'healthy')
validation_0_dir = path.join(validation_dir,'pneumonia')
test_1_dir = path.join(test_dir,'healthy')
test_0_dir = path.join(test_dir,'pneumonia')

try:
    mkdir(train_1_dir)
    mkdir(train_0_dir)
    mkdir(validation_1_dir)
    mkdir(validation_0_dir)
    mkdir(test_1_dir)
    mkdir(test_0_dir)
except FileExistsError:
    pass

### Determine lists of id's of images in classes ###
class0 = [np.array2string(row[0]) for row in gtruth.values if row[1] == 0]
class1 = [np.array2string(row[0]) for row in gtruth.values if row[1] != 0]
print("Number of images in classes: \nclass 0 - pneumonia:",len(class0),"\nclass 1 - healthy:",len(class1))


# The data will be split in the following proportions:
# * train set: 80%
# * validation set: 10%
# * test set: 10%
# 
# To solve the problem of an unbalanced dataset (which could result in false accuracy - model correctly predicting only images from the majority class), X-ray images of pneumonia were copied triple and then the ImageDataGenerator class implemented in the Keras framework was used. It allows random transformations in a given range, such as: height change, width change, rotation of images loaded during model training.
# 
# Final number of images in train, validation and test sets is shown in the output below.

# In[5]:


### Splitting the data into train-val-test sets/directories ###
import shutil
for n in range(3):
    suffix = str(n) if n else ""
    for i, img in enumerate(class0):
            fname = img+'.jpeg'
            new_fname = img+suffix+'.jpeg'
            if i < 0.8 * len(class0):
                shutil.copyfile(path.join(root_dir, fname), path.join(train_0_dir, new_fname))
            elif i < 0.9 * len(class0):        
                shutil.copyfile(path.join(root_dir, fname), path.join(validation_0_dir, new_fname))
            else:
                shutil.copyfile(path.join(root_dir, fname), path.join(test_0_dir, new_fname))

for i, img in enumerate(class1):
        fname = img+'.jpeg'
        if i < 0.8 * len(class1):
            shutil.copyfile(path.join(root_dir, fname), path.join(train_1_dir, fname))
        elif i < 0.9 * len(class1):        
            shutil.copyfile(path.join(root_dir, fname), path.join(validation_1_dir, fname))
        else:
            shutil.copyfile(path.join(root_dir, fname), path.join(test_1_dir, fname))
            
### Number of images in train-validation-test sets ###
print('Train images:', len(listdir(train_1_dir))+len(listdir(train_0_dir)))
print('Validation images:', len(listdir(validation_1_dir))+len(listdir(validation_0_dir)))
print('Test images:', len(listdir(test_1_dir))+len(listdir(test_0_dir)))

### Determining the dividers of validation/test set size for batch creation ###
val_images_num = len(listdir(validation_1_dir))+len(listdir(validation_0_dir))
print("Dividers of validation/test set size:",[x for x in range(1, val_images_num) if val_images_num % x == 0])


# To determine sizes of validation and prediction batches for model it's best to find a divider of the sample size. The last line in the output above shows the dividers of validation/test set size.
# 
# Chosen size of validation and test batches based on the dividers above is **53 images in batch** for **17 steps of validation **

# In[6]:


### Validation/test batch size and validation steps ###
v_batch_size = 53
v_val_steps = 17


# Batch size is set to number that is a power of 2, because processors are optimized for binary operations. Input size of the images is 128x128 px. All images are loaded in greyscale.

# In[7]:


### Defining image generators with rescaling, resizing, in grayscale ###
from keras.preprocessing.image import ImageDataGenerator

batch_size = 64
input_size = (128, 128)
input_shape = input_size + (1,)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=input_size,
    color_mode='grayscale',
    batch_size=v_batch_size,
    class_mode='binary')


# ## 3. CNN
# A deep convolutional neural network (CNN) is used to solve this problem. It is recommended in working with images as it gives very good results. The main mechanism used in convolutional neural networks is convolution - a mathematical operation on two functions, resulting in a third function - it can be seen as a modification of the first function. In image processing, the convolution of the matrix of the image pixels with the filter is used, resulting in a new matrix of pixels, created on the basis of the neighborhood. The very idea of using convolution in image processing has its source in the analysis of human perception of the image - we pay attention not only to the object, but also to its surroundings.

# ## 4. Creating and training the model

# CNN model used in this case consists of 3 convolutional layers (each followed by batch normalization layer and max pooling layer) and 2 dense layers. Detailed summary od the model is shown in the output below.
# 
# In order to avoid model overfitting, a Dropout layer was used in the neural network - the given percentage of connections between neurons is removed during training. Additionally batch normalization is implemented after every convolutional layer.

# In[8]:


### CNN model with dropout and batch normalization ###
from keras import models, layers, optimizers, losses

model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), use_bias=False, input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(64, (3, 3), use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.SGD(lr=0.005, nesterov=True), loss=losses.binary_crossentropy, metrics=['accuracy'])

### Details of the model ###
model.summary()


# In order to make the training of the model more effective Keras callbacks were used. These are functions that allow to perform certain operations during training:
# * *ReduceLROnPlateau* allows to reduce learning rate dynamically when there is no improvement of the loss function score over given number of epochs
# * *EarlyStopping* allows to stop training the model wwhen there is no improvement of the loss function score over given number of epochs
# * *ModelCheckpoint* allows to save best model's weights before they get worse as the model starts to overfit.
# 
# *Steps per epoch* parameter is calculated based on the number of training images and batch size, set earlier manually.

# In[9]:


### Defining callbacks ###
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
model_checkpoint = ModelCheckpoint(filepath='weights.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

### Fitting model to the data ###
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    epochs=50,
    shuffle=False,
    validation_data=validation_generator,
    validation_steps=v_val_steps,
    callbacks=[reduce_lr, early_stopping, model_checkpoint])
model.load_weights('weights.h5')
model.save('pneumonia-chest-x-ray-1.h5')


# ## 5. Results and summary

# Below are the visualisations of model training - comparisons of training and validation accuracy and loss scores.

# In[10]:


### Accuracy and loss plots ###
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = [ n+1 for n in range(len(acc))]
fig = pyplot.figure(figsize=(16, 4))

fig.add_subplot(1,2,1)
pyplot.plot(epochs, acc, 'k', label='Training accuracy')
pyplot.plot(epochs, val_acc, 'b', label='Validation accuracy')
pyplot.title('Training and validation accuracy')
pyplot.legend()

fig.add_subplot(1,2,2)
pyplot.plot(epochs, loss, 'k', label='Training loss')
pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
pyplot.title('Training and validation loss')
pyplot.legend()

pyplot.show()


# Scores of the final model on test set are given below:

# In[11]:


### Predict classes for test images ###
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_size,
    color_mode='grayscale',
    shuffle=False,
    batch_size=v_batch_size,
    class_mode='binary')
test_score = model.evaluate_generator(
    test_generator,
    steps=1)
print("Test set:\n loss: %.4f, accuracy: %.4f" % (test_score[0],test_score[1]))


# Numerical predictions for images in test set were computed in order to create confusion matrix, classification metrics and to show sample predictions from pneumonia class.

# In[12]:


### Get numerical predictions for test set images ###
test_generator.reset()
predictions = model.predict_generator(
    test_generator,
    steps=v_val_steps)

### True and predicted labels ###
true_labels = test_generator.labels.tolist()
pred_labels = [1 if p > 0.5 else 0 for p in predictions.ravel()]

### Confusion matrix ###
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
print("Confusion matrix: \nTP: %3i   FP: %3i\nFN: %3i   TN: %3i" % (tp, fp, fn, tn))

### Classification metrics ###
from sklearn.metrics import classification_report
print("\nClassification metrics:")
print(classification_report(true_labels, pred_labels))


# However accuracy may not be the best metric for binary classification in case of imbalanced dataset. Precision, recall and F1 score are better indicators of wheter the model is good or not.
# Precision is the ratio of true positive predictions and all positive predictions.
# Recall is the ratio of true positive predictions and all actual positive.
# F1 is a function of precision and recall, given by formula:
# 
# $ F1 = 2 * (\frac{Precision * Recall}{Precision + Recall}) $
# 
# Model trained in this problem results in both high accuracy (~95%) and high F1 score (~0,95). The goal is achieved. 

# 10 sample images from pneumonia class along with their predicted labels are presented below:

# In[13]:


### Prepare list of predictions and corresponding filenames ###
pred_with_filenames = {}
files = test_generator.filenames
files.sort()
for filename, pred in zip(files, predictions):
    pred_with_filenames[filename.split('/')[1]] = pred[0]

### Show sample test images from class 0 (pneumonia) with their predictions ###
import random
imagelist = listdir(test_0_dir)
random.shuffle(imagelist)
test_img_sample = [(filename, image.imread(test_0_dir + '/' + filename)) for i,filename in enumerate(imagelist) if i < 10]
rows, columns = 2, 5
fig = pyplot.figure(figsize=(16, 8))
for index, img in enumerate(test_img_sample):    
    fig.add_subplot(rows, columns, index+1)
    pyplot.imshow(img[1], cmap = pyplot.cm.gray)
    value = pred_with_filenames[img[0]]
    label = "pneumonia" if value > 0.5 else "healthy"
    title = "Predicted value: %.2f\nPredicted label: %s" % (value, label)
    print(title)
    pyplot.title(title)
pyplot.subplots_adjust(left = 1, right = 2)
pyplot.show()


# Deep learning methods are currently being developed strongly due to the huge potential associated with their use in many areas, not only those closely related to IT. Problems associated with the processing of text, images, video or speech have a chance to be automated soon, which will allow saving time and avoiding mistakes made during preforming such operations by people (which are often results of physical fatigue, the problem that does not concern computers). Among the various fields, medicine can also benefit greatly from the use of deep learning not only for disease diagnosis, but also for more experimental problems, such as DNA research.

# *Note: produced results are not easily reproducible due to use of the GPU, which uses non-deterministic algorithms.*
