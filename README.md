# Human Activity Recognition using CNN and LSTM RNN models
Final Year Individual Project. 

The final two models used are "final_cnn.py" and "final_lstm_rnn.py". 
Both models were trained using the SPHERE dataset, tested on another part of the SPHERE dataset, and then validated on the WISDM dataset. 

Data from WISDM's textfile was extracted using a small part (10 line code) of a tutorial from the following repository:
https://github.com/ni79ls/har-keras-cnn/blob/master/20180903_Keras_HAR_WISDM_CNN_v1.0_for_medium.py

LRP analysis code between line 515 and 535 was coded with help from Andrius Vabalas, PhD candidate at the University of Manchester. 

The following files were used for data preprocessing:
  1. Combining_Indidual_training_dataset.py
  2. Final_Training_Set.py
 
The following files were generated as a result of that data preprocessing:
  1. final_training_set_10people.csv
  2. final_training_set_8people.csv
  3. final_test_set_2people.csv
