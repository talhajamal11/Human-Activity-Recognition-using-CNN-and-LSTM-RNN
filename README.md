# Human Activity Recognition using CNN and LSTM RNN models
## Final Year Individual Project
### Table of contents
* [General info](#general-info)
* [Technologies and Tools used](#technologies-and-tools-used)
* [Setup](#setup)
* [Results](#results)


### General info
* This was a Final Year Individual Project intended to compare and analyse the performance of two **Deep Learning Neural Network Models** - *Convolutional Neural Network* and a *LSTM Recurrent Neural Network* - in recognizing the *physical activities* of human beings using *time series data* from accelerometers in smartwatches and smartphones. 
* Both models were used to run an analysis on Big Data (csv format) from the UK’s biobank with samples from more than 100,000 volunteers.
* The results can be seen in detail in the Final Year Individual Project Report.
* The LSTM RNN Model was predicted to be better at recognising the physical activity through time series data than the CNN model considering the advantages LSTM cells bring to time series analysis - the implementation of the project proved this thesis.
* In general, both networks worked to a satisfactory accuracy – up to 70% for the LSTM RNN and 75% for the CNN – when tested on dataset that it was trained on.
* However, the accuracies and the resultant predictions went down when tested on dataset that the neural networks had not been trained on. The LSTM RNN got an accuracy of 50% on the separate dataset whereas the CNN got an accuracy of up to 21% on it. Therefore, the generalisation capability of both networks is weak. 

### Technologies and Tools used 
&#x1f6e0;
![](https://img.shields.io/badge/Python-3.6-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/TensorFlow-2.0-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Keras-2.3.0-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/GoogleCloudPlatform-VirtualMachines-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Spyder-4.1-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Jupyter--informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)

### Setup
The final two models used are "final_cnn.py" and "final_lstm_rnn.py". 
Both models were trained using the SPHERE dataset, tested on another part of the SPHERE dataset, and then validated on the WISDM dataset. 

Data from WISDM's textfile was extracted using a small part (10 line code) of a tutorial from the following repository:
https://github.com/ni79ls/har-keras-cnn/blob/master/20180903_Keras_HAR_WISDM_CNN_v1.0_for_medium.py

LRP analysis code between line 515 and 535 was coded with help from Andrius Vabalas, PhD candidate at the University of Manchester. 

The following files were used for data preprocessing:
  1. Combining_Indidual_training_dataset.py
  2. Final_Training_Set.py
 
The following files were generated as a result of that data preprocessing on the SPHERE dataset:
  1. final_training_set_10people.csv
  2. final_training_set_8people.csv (used to train the model)
  3. final_test_set_2people.csv (used to test the model)

### Results

#### LSTM RNN results
LSTM RNN on testing set             |  LSTM RNN on validation set
:-------------------------:|:-------------------------:
<img src="images/LSTM%20RNN%20CM%20on%20testing%20set.png" width="450" >  |   <img src="images/LSTM%20RNN%20CM%20on%20validation%20set.png" width="450" >

LSTM RNN Accuracy vs Epochs             |  LSTM RNN Model Loss with Epochs
:-------------------------:|:-------------------------:
<img src="images/LSTM%20RNN%20Accuracy%20vs%20Epochs.png" width="450" >   |   <img src="images/LSTM%20RNN%20model%20loss.png" width="450" >





#### CNN results
CNN on testing set             |  CNN on validation set
:-------------------------:|:-------------------------:
<img src="images/CNN%20CM%20on%20testing%20set.png" width="450">  |   <img src="images/CNN%20CM%20on%20validation%20set.png" width="450">

CNN Accuracy vs Epochs             |  CNN Model Loss with Epochs
:-------------------------:|:-------------------------:
<img src="images/CNN%20model%20accuracy%20vs%20epochs.png" width="450">  |   <img src="images/CNN%20model%20loss%20vs%20epochs.png" width="450">


### The Final Year Individual Project Report and the feedback from my supervisor can be read from the following two files:
  1. Final Year Individual Project Report (73% marks recieved)
  2. Project Feedback from Professor
