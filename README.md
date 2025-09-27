### 🔍 Overview
<p align="justify">
This project aims to develop a lightweight binary classification model using data collected from a <b>Raspberry Pi camera</b>. The model can be used to assist a <b>self-driving robot</b> in detecting a sign of interest within a camera frame, upon which specific actions would be executed.

The model takes a frame from the camera as input and determines whether or not the target sign is present. To achieve this, two types of data were collected:
- <b>Positive Samples</b>: Images containing the sign of interest (a "traffic" sign), captured under various lighting conditions and from different angles.
- <b>Negative Samples</b>: Images without the sign, including lane tracks and surrounding areas relevant to the robot’s operating environment.

A total of 404 samples were obtained. Positive samples were cropped to emphasize the region of interest, enabling the model to learn more effectively
</p>

---

### ⚠️ Challenge: Overfitting on Small Dataset
<p align="justify">
Training a CNN with a small dataset can easily lead to overfitting—a situation where the model learns the training data too well but performs poorly on unseen data. To address this, several strategies were implemented:
</p>

---

### ✅ Techniques to Prevent Overfitting
<p align="justify">
<b>1. Data Augmentation</b>

After splitting the dataset into training, validation, and test sets, augmentation was applied only to the training data to manually increase variability:
- Random Rotation between ~10–15 degrees
- Random Contrast Adjustment

This was implemented using tf.keras.Sequential augmentation layers.  

<b>2. Hyperparameter Tuning</b>

Several hyperparameters were carefully selected and tuned:

- <b>Dropout:</b>  
To reduce model complexity and prevent overfitting, 20% dropout was applied after both Conv2D layers and one Dense layer. This randomly disables neurons during training, making the model less sensitive to noise.

- <b>L2 Regularization:</b>  
L2 regularization was applied to constrain the weights during training. For the model, a regularization rate of 0.01 provided the best balance between performance and generalization.

- Learning Rate Scheduling:  
A dynamic learning rate was used—starting at 0.01, and decreased exponentially after a certain number of epochs using LearningRateScheduler. This allows the model to learn faster in the beginning and fine-tune slowly as training progresses.

<b>3. Normalization</b>

Batch normalization was applied after each convolution layer. This standardizes activations across the mini-batch, stabilizing training and reducing overfitting due to the extra trainable parameters (mean and variance).

<b>4. Early Stopping</b>

Training was monitored using an EarlyStopping callback. If the validation loss stopped improving for a set number of epochs, training was halted early. This prevents the model from continuing to learn noise from the training data once generalization performance starts to decline.
</p>

---

### 🔍 Model Architecture

```bash
#SEQUENTIAL API
#Classifier Model
dropout_rate = 0.2
regularization_rate = 0.01
model = tf.keras.Sequential([InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)), #Specifies the shape input data shape

                             #Feature Extraction Layer
                             Conv2D(filters = 6, kernel_size = 3, strides = 1, padding='valid', activation='relu', kernel_regularizer= l2(regularization_rate)),      #ConvNet layer with 6 outputs(neurons)- each contains the kernel and bias
                             BatchNormalization(),
                             MaxPool2D(pool_size = 2,strides = 2),    #max pooling layer - selects the maximum value from each kernel window
                             Dropout(dropout_rate),

                             Conv2D(filters = 10, kernel_size = 3, strides = 1, padding='valid', activation='relu', kernel_regularizer= l2(regularization_rate)),      #ConvNet layer
                             BatchNormalization(),
                             MaxPool2D(pool_size = 2,strides = 2),    #max pooling layer
                             Dropout(dropout_rate),

                             
                             Flatten(), #flatten layer - converts 2D feature to 1D vector

                             #Classification Layer
                             Dense(15, activation = 'relu', kernel_regularizer= l2(regularization_rate)),      #dense layer with 10 neurons
                             Dropout(dropout_rate),
                             
                             Dense(1, activation = 'sigmoid'),      #Output layer (for 2-class classification)
])   
```

---

### 📈 Performance Evaluation
#### 📉 Model Loss & Accuracy
Training was stopped at epoch 19, just two epochs after signs of overfitting began to appear. As shown in the loss graph (Fig 1), both training and validation losses decreased over time. The closeness of the loss curves at later epochs suggests that the model was generalizing well.

<b>Final Validation Accuracy: 95%</b>
<p align="center">
<img width="500" height="350" alt="Model Loss" src="https://github.com/user-attachments/assets/0dad25d8-8480-4be2-b365-a93718ae2a2b" />  
</p>
<p align="center"><i>Fig 1: Training vs. Validation Loss Over Epochs</i></p>

#### 📉 Confusion Matrix
While accuracy is useful, it doesn't fully capture a model’s performance. To evaluate the model more thoroughly, a confusion matrix was generated:

<b>Final Validation Accuracy: 95%</b>
<p align="center">
  <img width="450" height="450" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f13eb3b6-1a8c-4456-a3a5-9e8bebd718cb" />
</p>
<p align="center"><i>Fig 2: Confusion Matrix</i></p>

From the confusion matrix, there are 35 true negatives, 5 true positives, 0 false negatives, and 1 false positives.
From this result:
- Precision: 83.3% - Proportion of correct positive predictions
- Recall: 100% - Ability to detect all actual positive cases
- Specificity: 97.2% - Ability to identify negative cases correctly
- F1-Score: 90.8% - Balanced measure of precision and recall

---

### 🚀 Deployed Model Test Results
The trained model was saved and later tested on varying conditions on raspberry pi which is interfaced with a camera. It performed reliably in detecting the sign, even with differences in angle and lighting. Test results are shown below:

<p align="center">
  <table>
    <!-- First Row -->
    <tr>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/cf245d02-4eba-4fcb-bc03-9c3c97859652" alt="Test Result 1" style="width: 100%;" />
        <br>
        <strong>Test Result 1</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/f6849fcf-5200-43c8-94f2-bf79ce15c092" alt="Test Result 2" style="width: 100%;" />
        <br>
        <strong>Test Result 2</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/6cd03f89-ebea-4deb-b1a2-ad941f8eedea" alt="Test Result 3" style="width: 100%;" />
        <br>
        <strong>Test Result 3</strong>
      </td>
    </tr>
    <!-- Second Row -->
    <tr>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/b2ee9e6b-200d-48b2-a0a7-eb460df845ed" alt="Test Result 4" style="width: 100%;" />
        <br>
        <strong>Test Result 4</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/6c928b23-c205-400c-af56-3c07b414261f" alt="Test Result 5" style="width: 100%;" />
        <br>
        <strong>Test Result 5</strong>
      </td>
      <td align="center" width="33.3%">
        <img src="https://github.com/user-attachments/assets/346bce75-0ecf-4740-a41a-1a80ad9944ba" alt="Test Result 6" style="width: 100%;" />
        <br>
        <strong>Test Result 6</strong>
      </td>
    </tr>
  </table>
</p>
