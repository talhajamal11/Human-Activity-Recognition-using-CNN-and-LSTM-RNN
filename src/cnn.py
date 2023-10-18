import numpy as np
import pandas as pd
from scipy import interpolate

import pickle # to serialise objects
from scipy import stats
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
RANDOM_SEED = 42

dataset_train = pd.read_csv('final_training_set_8people.csv')
training_set = pd.DataFrame(dataset_train.iloc[:,:].values)
training_set.columns = ["User","Activity", "Timeframe", "X axis", "Y axis", "Z axis"]

X = training_set.iloc[:, 3]
X = X.astype(float)
X = (X*1000000).astype('int64')

Y = training_set.iloc[:, 4]
Y = Y.astype(float)
Y = (Y*1000000).astype('int64')

Z = training_set.iloc[:, 5]
Z = Z.astype(float)
Z = (Z*1000000).astype('int64')

Old_T = (training_set.iloc[:, 2]).astype(float)
Old_T = (Old_T * 1000000)
Old_T = Old_T.astype('int64')

New_T = np.arange(0, 12509996000, 50000)
New_T = New_T.astype('int64')

# find interpolation function
interpolate_function = interpolate.interp1d(Old_T, X, axis = 0, fill_value="extrapolate")
X_Final = interpolate_function((New_T))
interpolate_function = interpolate.interp1d(Old_T, Y, axis = 0, fill_value="extrapolate")
Y_Final = interpolate_function((New_T))

interpolate_function = interpolate.interp1d(Old_T, Z, axis = 0, fill_value="extrapolate")
Z_Final = interpolate_function((New_T))

#Combining data into one pandas dataframe
Dataset = pd.DataFrame()

Dataset['X_Final'] = X_Final
Dataset['Y_Final'] = Y_Final
Dataset['Z_Final'] = Z_Final

Dataset['New_Timeframe'] = New_T
Dataset = Dataset/1e6
Dataset = Dataset[['New_Timeframe', 'X_Final', 'Y_Final', 'Z_Final']]
Dataset['New_Activity'] = ""
#Dataset = Dataset.astype('int64')
Dataset = Dataset[['New_Activity', 'New_Timeframe', 'X_Final', 'Y_Final', 'Z_Final']]


#function to fill in new dataset with related activity
Dataset = Dataset.to_numpy()
training_set = training_set.to_numpy()

time = 0
temp = training_set[0][1]
var_to_assign = ""
last_row = 0
new_row = 0
for i in range(len(training_set)-1):
    if(training_set[i][1] == temp):
        continue
    
    if (training_set[i][1] != temp):
        var_to_assign = temp
        temp = training_set[i][1]
        time = training_set[i][2]
        
        a1 = [x for x in Dataset[:, 1] if x <= time]
        new_row = len(a1)
        
        Dataset[last_row:new_row+1, 0] = var_to_assign
        last_row = new_row
        continue


#converting both arrays back to Dataframes
Dataset = pd.DataFrame(Dataset)
Dataset.columns = ['New_Activity', 'New_Timeframe', 'X_Final', 'Y_Final', 'Z_Final']
    
training_set = pd.DataFrame(training_set)   
training_set.columns = ["User","Activity", "Timeframe", "X axis", "Y axis", "Z axis"]

#Filling empty Dataset values
#Checking to see which index values are empty
df_missing = pd.DataFrame()
df_missing = Dataset[Dataset.isnull().any(axis=1)]

#Filling all empty values with preceding values
Dataset['New_Activity'].fillna(method = 'ffill', inplace = True)

Dataset = Dataset[:-7]

#to confirm no empty dataframes are present
df_empty = pd.DataFrame()
df_empty = Dataset[Dataset['New_Activity']=='']
        
#Combining smaller classes into larger/main classes

Dataset = Dataset.to_numpy()

for i in range(0, len(Dataset)-1): 
    if Dataset[i][0] == "a_loadwalk" or Dataset[i][0] == "a_jump":
        Dataset[i][0] = "a_walk"
    if Dataset[i][0] == "p_squat" or Dataset[i][0] == "p_kneel" or Dataset[i][0] == "p_lie" or Dataset[i][0] == "t_lie_sit" or Dataset[i][0] == "t_sit_lie" or Dataset[i][0] == "t_sit_stand":
        Dataset[i][0] = "p_sit"
    if Dataset[i][0] == "p_bent" or Dataset[i][0] == "t_bend" or Dataset[i][0] == "t_kneel_stand" or Dataset[i][0] == "t_stand_kneel" or Dataset[i][0] == "t_stand_sit" or Dataset[i][0] == "t_straighten" or Dataset[i][0] == "t_turn":
        Dataset[i][0] = "p_stand"
    if Dataset[i][0] == "unknown":
        Dataset[i][0] = Dataset[i-1][0]


Dataset = pd.DataFrame(Dataset)
Dataset.columns = ['New_Activity', 'New_Timeframe', 'X_Final', 'Y_Final', 'Z_Final']

#Encoding the Activity
from sklearn.preprocessing import LabelEncoder
Label = LabelEncoder()
Dataset['Label'] = Label.fit_transform(Dataset['New_Activity'])

Label_Encoder_mapping = dict(zip(Label.classes_, Label.transform(Label.classes_)))

#Adding Standardized Scaling to data
X = Dataset[['X_Final', 'Y_Final', 'Z_Final']]
y = Dataset[['Label']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(data=X, columns = ['X_Final', 'Y_Final', 'Z_Final'])
scaled_X['Label'] = y.values


#Feature Generation and Data Transformation
TIME_STEPS = 200
N_FEATURES = 3
STEP = 20

segments = []
labels = []

for i in range(0, len(Dataset) - TIME_STEPS, STEP): #To give the starting point of each batch
    xs = scaled_X['X_Final'].values[i: i + TIME_STEPS]
    ys = scaled_X['Y_Final'].values[i: i + TIME_STEPS]
    zs = scaled_X['Z_Final'].values[i: i + TIME_STEPS]
    label = stats.mode(scaled_X['Label'][i: i + TIME_STEPS]) #this statement returns mode and count
    label = label[0][0] #to ge value of mode
    segments.append([xs, ys, zs])
    labels.append(label)
     
#reshaping our data
reshaped_segments = np.asarray(segments, dtype = np.float32).reshape(-1, TIME_STEPS, N_FEATURES)

labels = np.asarray(labels)


"""#Using one hot encoding
l = pd.DataFrame(labels)
l_one_hot = pd.get_dummies(l)

labels_columns = l_one_hot.idxmax(axis = 1)

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32) 
"""
#labels.shape

X_train = reshaped_segments
y_train = labels


#Importing Test Set

#Importing Test DataSet
Test_set = pd.read_csv('final_test_set_2people.csv')
Test_set.drop(['Unnamed: 0'], axis = 1, inplace = True)


#combing smaller classes to bigger classes

Test_set = Test_set.to_numpy()
for i in range(0, len(Test_set)-1):
    if Test_set[i][1] == "a_loadwalk" or Test_set[i][1] == "a_jump":
        Test_set[i][1] = "a_walk"
    if Test_set[i][1] == "p_squat" or Test_set[i][1] == "p_kneel" or Test_set[i][1] == "p_lie" or Test_set[i][1] == "t_lie_sit" or Test_set[i][1] == "t_sit_lie" or Test_set[i][1] == "t_sit_stand":
        Test_set[i][1] = "p_sit"
    if Test_set[i][1] == "p_bent" or Test_set[i][1] == "t_bend" or Test_set[i][1] == "t_kneel_stand" or Test_set[i][1] == "t_stand_kneel" or Test_set[i][1] == "t_stand_sit" or Test_set[i][1] == "t_straighten" or Test_set[i][1] == "t_turn":
        Test_set[i][1] = "p_stand"
    if Test_set[i][0] == " " or Test_set[i][0] == "unknown":
        Test_set[i][0] = Test_set[i-1][0]

Test_set = pd.DataFrame(Test_set)
Test_set.columns = ["User","New_Activity", "Timeframe", "X axis", "Y axis", "Z axis"]

#Filling empty Dataset values
#Checking to see which index values are empty
df_missing = pd.DataFrame()
df_missing = Test_set[Test_set.isnull().any(axis=1)]

#Filling all empty values with preceding values
Test_set['New_Activity'].fillna(method = 'ffill', inplace = True)

#Encoding the Activities
#Test_set.Activity.apply(str)
Test_set['New_Activity'] = Test_set.New_Activity.astype(str)
from sklearn.preprocessing import LabelEncoder
Test_Label = LabelEncoder()
Test_set['Test_Label'] = Test_Label.fit_transform(Test_set['New_Activity'])
Test_Label_Encoder_mapping = dict(zip(Test_Label.classes_, Test_Label.transform(Test_Label.classes_)))





#Scaling the data
test_X = Test_set[['X axis', 'Y axis', 'Z axis']]
test_y = Test_set[['Test_Label']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test_X = scaler.fit_transform(test_X)
test_scaled_X = pd.DataFrame(data=test_X, columns = ['X axis', 'Y axis', 'Z axis'])
test_scaled_X['Test_Label'] = test_y.values

TEST_TIME_STEPS = 200
TEST_N_FEATURES = 3
TEST_STEP = 20

test_segments = []
test_labels = []

for i in range(0, len(Test_set) - TEST_TIME_STEPS, TEST_STEP): #To give the starting point of each batch
    t_xs = test_scaled_X['X axis'].values[i: i + TEST_TIME_STEPS]
    t_ys = test_scaled_X['Y axis'].values[i: i + TEST_TIME_STEPS]
    t_zs = test_scaled_X['Z axis'].values[i: i + TEST_TIME_STEPS]
    test_label = stats.mode(test_scaled_X['Test_Label'][i: i + TEST_TIME_STEPS]) #this statement returns mode and count
    test_label = test_label[0][0] #to ge value of mode
    test_segments.append([t_xs, t_ys, t_zs])
    test_labels.append(test_label)
    
#reshaping our data

test_reshaped_segments = np.asarray(test_segments, dtype = np.float32).reshape(-1, TEST_TIME_STEPS, TEST_N_FEATURES)
test_labels = np.asarray(test_labels)

#Using one hot encoding
#test_labels = np.asarray(pd.get_dummies(test_labels), dtype = np.float32)

X_test = test_reshaped_segments
y_test = test_labels

test_df = pd.DataFrame(y_test)


#Importing Keras libraries and packages
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
#from tensorflow.keras.layers import MaxPooling1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

#LRP

import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plot
import os

import keras
import keras.backend
import keras.layers
import keras.models


verbose, epochs, batch_size = 0, 100, 32
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 5


import tensorflow as tf
"""
tf.compat.v1.disable_eager_execution()
tf.keras.backend.clear_session()


graph = tf.compat.v1.get_default_graph()
global graph
"""

regressor = Sequential()
regressor.add(Conv1D(filters = 32, kernel_size = 5, activation='relu', input_shape=(n_timesteps, n_features)))
regressor.add(Dropout(0.1))


regressor.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
regressor.add(Dropout(0.2))

regressor.add(tf.compat.v1.layers.MaxPooling1D(pool_size=2))

regressor.add(Flatten())

regressor.add(Dense(100, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(5, activation='softmax'))

regressor.compile(optimizer = 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting the Model

regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data= (X_test, y_test) ,verbose = 1)


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

y_pred = regressor.predict_classes(X_test)

print(accuracy_score(y_test, y_pred) * 100)

#mat = confusion_matrix(y_test, y_pred)
#plot_confusion_matrix(conf_mat=mat, class_names=Label.classes_, show_normed=True, figsize=(7,7), colorbar=True, show_absolute=False)







#Testing on the WISDM Dataset

#Loading WISDM dataset for validation
#change this code slightly to avoid plagiarism
def read_data(file_path):

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df

def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan
 
def show_basic_dataframe_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

# Load data set containing all the data from csv
wisdm_dataset = read_data('WISDM_ar_v1.1_raw.txt')


#preprocessing WISDM dataset
#normalising accelerometer values to be between 0 and 1
wisdm_dataset['x-axis'] = wisdm_dataset['x-axis']/10
wisdm_dataset['y-axis'] = wisdm_dataset['y-axis']/10
wisdm_dataset['z-axis'] = wisdm_dataset['z-axis']/10



indexNames = wisdm_dataset[ wisdm_dataset['activity'] == "Jogging" ].index
 
# Delete these row indexes from dataFrame
wisdm_dataset.drop(indexNames , inplace=True)



#to confirm no empty dataframes are present
w_df_jogging = pd.DataFrame()
w_df_jogging = wisdm_dataset[wisdm_dataset['activity']=='Jogging']


#renaming the wisdm dataset with the class names that classifier is trained in
wisdm_dataset = wisdm_dataset.to_numpy()

for i in range(0, len(wisdm_dataset)):
    if wisdm_dataset[i][1] == "Walking":
        wisdm_dataset[i][1] = "a_walk"
    if wisdm_dataset[i][1] == "Upstairs":
        wisdm_dataset[i][1] = "a_ascend"
    if wisdm_dataset[i][1] == "Downstairs":
        wisdm_dataset[i][1] = "a_descend"
    if wisdm_dataset[i][1] == "Sitting":
        wisdm_dataset[i][1] = "p_sit"
    if wisdm_dataset[i][1] == "Standing":
        wisdm_dataset[i][1] = "p_stand"
        

wisdm_dataset = pd.DataFrame(wisdm_dataset)
wisdm_dataset.columns = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']


from sklearn.preprocessing import LabelEncoder
WISDM_Label = LabelEncoder()
wisdm_dataset['Test_Label'] = WISDM_Label.fit_transform(wisdm_dataset['activity'])
WISDM_Label_Encoder_mapping = dict(zip(WISDM_Label.classes_, WISDM_Label.transform(WISDM_Label.classes_)))



#Scaling the data
wisdm_test_X = wisdm_dataset[['x-axis', 'y-axis', 'z-axis']]
wisdm_test_y = wisdm_dataset[['Test_Label']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
wisdm_test_X = scaler.fit_transform(wisdm_test_X)
wisdm_test_scaled_X = pd.DataFrame(data=wisdm_test_X, columns = ['x-axis', 'y-axis', 'z-axis'])
wisdm_test_scaled_X['Test_Label'] = wisdm_test_y.values




#segmenting the data
WISDM_TEST_TIME_STEPS = 200
WISDM_TEST_N_FEATURES = 3
WISDM_TEST_STEP = 20

wisdm_test_segments = []
wisdm_test_labels = []

for i in range(0, len(wisdm_dataset) - WISDM_TEST_TIME_STEPS, WISDM_TEST_STEP): #To give the starting point of each batch
    w_t_xs = wisdm_test_scaled_X['x-axis'].values[i: i + WISDM_TEST_TIME_STEPS]
    w_t_ys = wisdm_test_scaled_X['y-axis'].values[i: i + WISDM_TEST_TIME_STEPS]
    w_t_zs = wisdm_test_scaled_X['z-axis'].values[i: i + WISDM_TEST_TIME_STEPS]
    wisdm_test_label = stats.mode(wisdm_test_scaled_X['Test_Label'][i: i + WISDM_TEST_TIME_STEPS]) #this statement returns mode and count
    wisdm_test_label = wisdm_test_label[0][0] #to ge value of mode
    wisdm_test_segments.append([w_t_xs, w_t_ys, w_t_zs])
    wisdm_test_labels.append(wisdm_test_label)

    
#reshaping our data
wisdm_test_reshaped_segments = np.asarray(wisdm_test_segments, dtype = np.float32).reshape(-1, WISDM_TEST_TIME_STEPS, WISDM_TEST_N_FEATURES)
#reshaped_segments.shape
wisdm_test_labels = np.asarray(wisdm_test_labels)
#Using one hot encoding
#wisdm_test_labels = np.asarray(pd.get_dummies(wisdm_test_labels), dtype = np.float32)


wisdm_X_test = wisdm_test_reshaped_segments
wisdm_y_test = wisdm_test_labels
wisdm_test_df = pd.DataFrame(wisdm_y_test)

wisdm_y_pred = regressor.predict_classes(wisdm_X_test)

print(accuracy_score(wisdm_y_test, wisdm_y_pred) * 100)

#wisdm_mat = confusion_matrix(wisdm_y_test, wisdm_y_pred)
#plot_confusion_matrix(conf_mat=wisdm_mat, class_names=Label.classes_, show_normed=True, figsize=(8,8))

"""
import innvestigate
import innvestigate.utils as iutils

#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.keras.backend.clear_session()

model = regressor


#graph = tf.compat.v1.get_default_graph()
#global graph

model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

data_size = 200

resultA = np.zeros((data_size, X_test.shape[2])).reshape(1,data_size,X_test.shape[2])
resultT = np.zeros((data_size, X_test.shape[2])).reshape(1,data_size,X_test.shape[2])
rezimage = np.zeros((data_size, X_test.shape[2])).reshape(1,data_size,X_test.shape[2])

#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.keras.backend.clear_session()
#
#graph = tf.compat.v1.get_default_graph()
#global graph
tf.compat.v1.disable_v2_behavior()


for n in range(X_test.shape[0]):
    image = X_test[n:n+1]
    correct_class = y_test[n]
    prediction_class = y_pred[n]
    #Creating LRP analyser
    LRP_epsilon = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(model, epsilon=1e-07, bias=True, neuron_selection_mode="index")
    #Applying the analyzer
    
    analysisT = LRP_epsilon.analyze(image, 0)
    analysisA = LRP_epsilon.analyze(image, 1)
    
    resultT = np.vstack((resultT,analysisT))
    resultA = np.vstack((resultA,analysisA))        
    imageraw = X_test[n:n+1]
    rezimage = np.vstack((rezimage,imageraw)) 
    

print("1")


    


#
## 0 - x_coordiate, 1 - y_coordiate, 2 - z_coordiate, 3 - velocity_data, 4 - acceleration_data, 5 - jerk_data, 6 - azimuth, 7 - elevation, 8 - roll  
#
#  
#      
#fig_X = plot.figure()
#ax = fig.add_subplot(9, 1, x+1)
#
#ax.set_title('True_label='+str(int(correct_class))+', predicted_label='+str(predicted_class))
#         
#    
#ax.set_ylabel(X_axis)
#ax.set_yticklabels([])
#ax.plot(analysis2[:,:,x].squeeze())
#
#
##    ax2 = fig.add_subplot(9, 1, x+1)
##    ax2.set_yticklabels([])
##    ax2.plot(image[:,:,x].squeeze())
#
#
#
#
#
#plot.figure(1)
#plot.plot(image.squeeze())
#plot.ylabel('Vertical amplitude')
#plot.title('True label = %f, %f' %correct_class, predicted_class)
#plot.title('True_label='+str(int(class_correct))+', predicted_label='+str(class_predicted))
#
#plot.figure(2)
#plot.plot(analysis2.squeeze())
#plot.ylabel('LRP_epsilon relevance')
#
#
#
## Creating an analyzer
#gradient_analyzer = innvestigate.create_analyzer("gradient", model)
#
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.keras.backend.clear_session()
#graph = tf.compat.v1.get_default_graph()
#import keras.backend as K
#K.clear_session()
#global graph
#
#
#
#analysis = gradient_analyzer.analyze(X_test)
#
#
#
#
#analyzer = innvestigate.create_analyzer("lrp.z", model)
#
#analysis = analyzer.analyze(X_test)
#
"""