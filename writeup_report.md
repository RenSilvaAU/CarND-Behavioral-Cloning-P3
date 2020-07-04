# **Behavioral Cloning** 

## Ren Silva - Udacity




[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** (this document) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 132-154) 

The model uses Relu activation to introduce nonlinearity (model.py lines 132-154), and the data is normalized in the model using a Keras lambda layer (code line 136). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 149,151,153). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 91). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 164).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

In addition, I drove the track the both directions (the original, and then going backwards), and flipped each image during the preparation of the batch.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to initialy create a very simplistic lenet model, and only the first set of training data.

My first step was to use a convolution neural network model similar to the one we used in the previous assignment (traffic line recognition), wiht one twist - no activation on the last layer.

I thought this model might be appropriate because the model needs to make sense of the images, and convolutional networks are ideal to teach the model to recognize images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included dropout layers afer each dense layer.

The next step was to run the simulator to see how well the car was driving around track one. I notice that, despite all the changes, the vehicle would start well, and always turn into a dirt road to the right.

To correct that, I added a Lambda layer to discard the top part of the image (line 134) - but it still appeared to have a problem. 

Discussing the problem with a non-techincal friend, and explaining my frustration, he suggested I looked at how the model was "seeing" colors. That is when I realized that I was training the model with BRG images (read with opencv), but inference was being done with RGB images - I then corrected the data generator for that.

The final step was to run the simulator to see how well the car was driving around track one - and now it was driving pretty much the same way as I did.

(NB - I drive like a race driver, making use of the entire track, and making the curves as smooth as possible, so going from zebra to zebra) - and that was very much how my autonomous driver behaved.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the architecture shown below:


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
