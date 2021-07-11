# Mammography-Abnormality-Detector

Deep Neural Network to detect abnormalities on Mammographies made in hospitals. Used [MIAS Mammography Dataset](https://www.kaggle.com/kmader/mias-mammography) and reached 96% Accuracy on Test data. The following sections will explain how the model works.


# Table of contents

* [Introduction](#Introduction)
* [Dataset](#Dataset)
* [Neural-Network](#Neural-Network)
* [Results](#Results)
* [Contact](#Contact)
* [License](#License)

# Introduction
>[Table of contents](#table-of-contents)

The rapid development of deep learning, a family of machine learning techniques, has spurred much interest in its application to medical imaging problems. Here, we develop a deep learning algorithm that can accurately detect breast cancer on screening mammograms using Convolutional Neural Networks (CNNs). This Code show that with little information and a reduced dataset, amazing results can be obtained and detect Breasts Abnormalitien with a 96% Accuracy.

# Dataset
>[Table of contents](#table-of-contents)

The dataset used was [MIAS Mammography Dataset](https://www.kaggle.com/kmader/mias-mammography) that cosists of 330 Breasts Mammography Images in a 1024x1024 Pixel Resolution, with aproximately 66% Images of normal Breasts and 33% Images of breasts with Abnormalities. Some examples can be seen in the following figures:


Full Image Size:
<img src="https://github.com/iancraz/Mammography-Abnormality-Detector/blob/main/docs/mam_full.png?raw=true" width=1024 align=center>

25 Example of Dataset images:

<img src="https://github.com/iancraz/Mammography-Abnormality-Detector/blob/main/docs/25_mam.png?raw=true" width=1024 align=center>

As can be seen, without being an expert in the subject, it is almost imposible to detect by just looking if a sigle breast has or not breast cancer. So for this matter, our Neural Network comes to help.

## Data Augmentation

Since 330 images are a limited number of samples, we need to make data augmentation. Firs of all, since 1024x1024 has a lot af pixels and hence redundancy, we decided to test wheater reducing the resolution would worsen our net, but by results taken, we could reduce the images into 128x128 pixels saving a lot of space and being able to introduce much more examples into our augmented dataset (remember we were restricted by the 15 GB of RAM given by Kaggle). We then implemented our data augmentation algorithm.

Since breasts abnormalities detection does not depend on the angle the mammography was taken, we decided to rotate every image in a 360 degrees, by a 2 degree factor, so for every image on the dataset, we generated 180 new images. With this now have 44280 images to work with (246*180). You may be asking yourself, Why 246 images used and not the holw 330 images? And you are right, we could have used the 330 images, but since the dataset was not balanced, we decided to take out 84 normal breasts mammographies to get the 50% division into samples, so our CNN cannot overfit an learn that all mammographies are normals.

With this in mind, we now have 44280 images with a 128x128 pixel resolution. Here is the code implementing the data loading and augmentation

```Python
img_path = []
last_label = []
IMG_SIZE = 128

for i in range(len(img_name)):
    
    img = cv2.imread(img_name[i], 0)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    rows, cols= img.shape
    for angle in range(180):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle*2, 1)    #Rotate 2 degrees
            img_rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
            img_path.append(img_rotated)
            if label[i] == 1:
                last_label.append(1)
            else:
                last_label.append(0)
```

### Further data augmentation

We couldnt augment more data for the restriction in Kaggle, however if this restriction is removed, or if other system is used, we have some reccomendations for further data augmentation:

* Mirroring Images: By mirroring images one can gain a 4x factor in the data augmented achieving 177120 total images.
* Rotating images by 1 Degree (or less): By doing this one can gain 2x factor or more, achieving 354240 total images.
* Random Jitter: One can resize images a little biger than 128x128 and randomly crop N images of 128x128 pixel gaining an Nx factor.


# Neural Network
>[Table of contents](#table-of-contents)

The model implemented was as shown below:

<img src="https://github.com/iancraz/Mammography-Abnormality-Detector/blob/main/docs/model_1.png?raw=true" width=500 align=center>


The code to implemet this model is:

```Python
ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)


model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer = ini,
                 input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Conv2D(64,
                 kernel_size=(3,3),
                 kernel_initializer = ini,
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, 
                 kernel_size=(3,3),
                 kernel_initializer = ini,
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Dense(64, 
                kernel_initializer = ini,
                activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-3)

model.compile(optimizer=optimizer,
              loss= 'binary_crossentropy',
              metrics=['accuracy'])
```

Note that the optimizer used was **Adam** with a learning rate of 1e-3. Finally to achieve the results we used an early stop callback with a 3 epochs patience. For training a 20% Validation split was used to test overfitting, and a 128 batch size was suitable.

```Python
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3,restore_best_weights=True, verbose=1)

...

train = True
if train:
    epochs=100
    history = model.fit(x_train,
                 y_train,
                 validation_split=0.2,
                 epochs=epochs,
                 batch_size=128,
                 callbacks=[early_stop, model_check_point])
else:
    model = tf.keras.models.load_model('./')
```

# Results
>[Table of contents](#table-of-contents)

The results achived were:

* 97.21% Accuracy on Training Data
* 96.13% Accuracy on Validation Data
* 96.01% Accuracy on Test data

Here are some images of the training epochs:

<img src="https://github.com/iancraz/Mammography-Abnormality-Detector/blob/main/docs/Acc.png?raw=true" width=500 align=center>

<img src="https://github.com/iancraz/Mammography-Abnormality-Detector/blob/main/docs/Loss.png?raw=true" width=500 align=center>

Finally let's see one example of the test data individually:

<img src="https://github.com/iancraz/Mammography-Abnormality-Detector/blob/main/docs/test.png?raw=true" width=500 align=center>

The Ground Truth is that this image **does not have abnoralities**

The result of out CNN was: **No Abnormality** with a 1.7e-7 probability of having an abnormality.

# Contact
>[Table of contents](#table-of-contents)

Please do not hesitate to reach out to me if you find any issue with the code or if you have any questions.

* Personal email: [idiaz@itba.edu.ar](mailto:idiaz@itba.edu.ar)

* LinkedIn Profile: [https://www.linkedin.com/in/iancraz/](https://www.linkedin.com/in/iancraz/)

# License
>[Table of contents](#table-of-contents)

```
MIT License

Copyright (c) 2021 Ian Cruz Diaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


