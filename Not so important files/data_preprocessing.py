m#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import pickle # to serialise objects
from scipy import stats
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


dataset_train = pd.read_csv('acceleration_labelled_data.csv') 
training_set = pd.DataFrame(dataset_train.iloc[:, 1:6].values)
training_set.columns = ["Activity", "Timeframe", "X axis", "Y axis", "Z axis"]

training_set.info()

#Interpolating the values to have a value at 20 Hz exactly

X = training_set.iloc[:, 1:3]

X = X.set_index(['Timeframe'])
X.index = pd.to_datetime(X.index, unit='s')
X = X.astype(float)

X = X.resample('0.001S','mean').interpolate()
X = X.resample('0.05S', 'mean').interpolate()

X_old = training_set.iloc[:, 1:3]
X_old = X_old.set_index(['Timeframe'])
X_old.index = pd.to_datetime(X_old.index, unit='s')

X = X.reset_index()
X_old = X_old.reset_index()
plt.plot(X['Timeframe'], X['X axis'], color = 'red')
plt.plot(X_old['Timeframe'], X_old['X axis'], color = 'blue')


Y = training_set.iloc[:, [1,3]]

Y = Y.set_index(['Timeframe'])
Y.index = pd.to_datetime(Y.index, unit='s')
Y = Y.astype(float)

Y = Y.resample('0.001S','mean').interpolate()
Y = Y.resample('0.05S', 'mean').interpolate()

Y_old = training_set.iloc[:, [1,3]]
Y_old = Y_old.set_index(['Timeframe'])
Y_old.index = pd.to_datetime(Y_old.index, unit='s')

Y = Y.reset_index()
Y_old = Y_old.reset_index()
plt.plot(Y['Timeframe'], Y['Y axis'], color = 'red')
plt.plot(Y_old['Timeframe'], Y_old['Y axis'], color = 'blue')



Z = training_set.iloc[:, [1,4]]

Z = Z.set_index(['Timeframe'])
Z.index = pd.to_datetime(Z.index, unit='s')
Z = Z.astype(float)

Z = Z.resample('0.001S','mean').interpolate()
Z = Z.resample('0.05S', 'mean').interpolate()

Z_old = training_set.iloc[:, [1,4]]
Z_old = Z_old.set_index(['Timeframe'])
Z_old.index = pd.to_datetime(Z_old.index, unit='s')

Z = Z.reset_index()
Z_old = Z_old.reset_index()
plt.plot(Z['Timeframe'], Z['Z axis'], color = 'red')
plt.plot(Z_old['Timeframe'], Z_old['Z axis'], color = 'blue')

#Final Set used for training

Final_df = pd.concat([X, Y, Z], axis = 1)
Final_df.columns = ['Time', 'X axis', 'remove', 'Y axis', 'remove', 'Z axis']
Final_df = Final_df.drop(Final_df.columns[2], axis = 1)


#Evaluating and observing the given data

countofActivity = training_set['Activity'].value_counts()
print(countofActivity)

countofActivity.plot(kind = 'bar', title = 'Training examples by activity type')

#Data preprocessing

N_timesteps = 200
N_features = 3
step = 20
segments = []
labels = []

for i in range(0, len(Final_df) - N_timesteps, step):
    xs = Final_df['x-axis'].values[i: i + N_timesteps]
    ys = Final_df['y_axis'].values[i: i + N_timesteps]
    zs = Final_df['z-axis'].values[i: i + N_timesteps]
    label = stats.mode(Final_df['Activity'])[i: i+ N_timesteps] #returns mode and count
    label = label[0][0]    #value of mode
    segments.append([xs,ys,zs])
    labels.append(label)
    
    
#reshaping the data
    
    
    
#One Hot encoding
    
    
    
#splitting the dataset into training and testing
    
    
#Building the model
    #relu for hidden layer
    #softmax for output layer
    
    







upsampled_X = pd.DataFrame(X.resample('50ms').mean())

X_axis_final = pd.DataFrame(upsampled_X.interpolate(method = 'linear'))

us_X = pd.DataFrame(upsampled_X)

X_axis_final = upsampled_X.interpolate(method = 'linear')

plt.plot(training_set['Timeframe'], training_set['X axis'])

plt.plot(training_set['Timeframe'], training_set['X axis'])





new


new_timeframe = np.arange(0, 1824, 0.05)



interpolate_function = interpolate.interp1d(training_set.iloc[:,1], training_set.iloc[:,2], axis = 0, fill_value="extrapolate")
X_axis_final = interpolate_function((new_timeframe.astype('int64')))

interpolate_function = interpolate.interp1d(training_set.iloc[:,1].astype('int64'), training_set.iloc[:,3], axis = 0, fill_value="extrapolate")
Y_axis_final = interpolate_function((new_timeframe.astype('datetime64[ns]')).astype('int64'))

interpolate_function = interpolate.interp1d(training_set.iloc[:,1], training_set.iloc[:,4], axis = 0, fill_value="extrapolate")
Z_axis_final = interpolate_function((new_timeframe)







interpolate_function = interpolate.interp1d(wrist_left_time_fsef.astype('int64'), wrist_left_accelerometer_x,axis=0,fill_value="extrapolate") # find interpolation function
X_axis_final = interpolate_function((wrist_left_time.astype('datetime64[ns]')).astype('int64'))


"""
Tns = int(1e9 * (1/wrist_left_fs_effective)) # as sampling number is not round this will introduce some slight shift
wrist_left_time_fsef = np.asarray(Tns * (np.linspace(0, len(wrist_left_ppg), len(wrist_left_ppg), endpoint=False)),dtype='datetime64[ns]')
dt3 = ecg_start_peak - wrist_left_time_fsef[0]
wrist_left_time_fsef = wrist_left_time_fsef + dt3
 
# Re-sample onto the correct 200 Hz timebase
interpolate_function = interpolate.interp1d(wrist_left_time_fsef.astype('int64'), wrist_left_ppg,axis=0,fill_value="extrapolate") # find interpolation function
wrist_left_ppg_final = interpolate_function((wrist_left_time.astype('datetime64[ns]')).astype('int64'))

interpolate_function             = interpolate.interp1d(wrist_left_time_fsef.astype('int64'), wrist_left_accelerometer_x,axis=0,fill_value="extrapolate") # find interpolation function
wrist_left_accelerometer_x_final = interpolate_function((wrist_left_time.astype('datetime64[ns]')).astype('int64'))
interpolate_function             = interpolate.interp1d(wrist_left_time_fsef.astype('int64'), wrist_left_accelerometer_y,axis=0,fill_value="extrapolate") # find interpolation function
wrist_left_accelerometer_y_final = interpolate_function((wrist_left_time.astype('datetime64[ns]')).astype('int64'))
interpolate_function             = interpolate.interp1d(wrist_left_time_fsef.astype('int64'), wrist_left_accelerometer_z,axis=0,fill_value="extrapolate") # find interpolation function
wrist_left_accelerometer_z_final = interpolate_function((wrist_left_time.astype('datetime64[ns]')).astype('int64'))
"""
 




















