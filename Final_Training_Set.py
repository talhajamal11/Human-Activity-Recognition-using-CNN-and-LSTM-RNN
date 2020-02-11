import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing individual user data
user_1 = pd.read_csv('User_1.csv')
user_2 = pd.read_csv('User_2.csv')
user_3 = pd.read_csv('User_3.csv')
user_4 = pd.read_csv('User_4.csv')
user_5 = pd.read_csv('User_5.csv')
user_6 = pd.read_csv('User_6.csv')
user_7 = pd.read_csv('User_7.csv')
user_8 = pd.read_csv('User_8.csv')
user_9 = pd.read_csv('User_9.csv')
user_10 = pd.read_csv('User_10.csv')

#Fixing User 1 table
#user_1 = pd.DataFrame(user_1.iloc[:34247,:].values)
#user_1.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis"]
##user_1["Timeframe"] = user_1["Timeframe"] - 45.944096
#Export_csv = user_1.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_1.csv')
#User_2 = User_2.to_numpy()
#User_2 = pd.DataFrame(User_2)


#Fixing Timeframe for each user

#for user 2
user_1 = user_1.to_numpy()
user_2 = user_2.to_numpy()
user_2[0][2] = user_1[34246][2] + user_2[0][6]
for i in range(1, len(user_2)):
    user_2[i][2] = user_2[i-1][2] + user_2[i][6]

user_1 = pd.DataFrame(user_1)
user_1.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis"]
user_2 = pd.DataFrame(user_2)
user_2.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


#for user 3
user_2 = user_2.to_numpy()
user_3 = user_3.to_numpy()
user_3[0][2] = user_2[32450][2] + user_3[0][6]
for i in range(1, len(user_3)):
    user_3[i][2] = user_3[i-1][2] + user_3[i][6]

user_2 = pd.DataFrame(user_2)
user_2.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_3 = pd.DataFrame(user_3)
user_3.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


#for user 4
user_3 = user_3.to_numpy()
user_4 = user_4.to_numpy()
user_4[0][2] = user_3[29623][2] + user_4[0][6]
for i in range(1, len(user_4)):
    user_4[i][2] = user_4[i-1][2] + user_4[i][6]

user_3 = pd.DataFrame(user_3)
user_3.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_4 = pd.DataFrame(user_4)
user_4.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]

#for user 5
user_4 = user_4.to_numpy()
user_5 = user_5.to_numpy()
user_5[0][2] = user_4[30116][2] + user_5[0][6]
for i in range(1, len(user_5)):
    user_5[i][2] = user_5[i-1][2] + user_5[i][6]

user_4 = pd.DataFrame(user_4)
user_4.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_5 = pd.DataFrame(user_5)
user_5.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


#for user 6
user_5 = user_5.to_numpy()
user_6 = user_6.to_numpy()
user_6[0][2] = user_5[29682][2] + user_6[0][6]
for i in range(1, len(user_6)):
    user_6[i][2] = user_6[i-1][2] + user_6[i][6]

user_5 = pd.DataFrame(user_5)
user_5.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_6 = pd.DataFrame(user_6)
user_6.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


#for user 7
user_6 = user_6.to_numpy()
user_7 = user_7.to_numpy()
user_7[0][2] = user_6[16611][2] + user_7[0][6]
for i in range(1, len(user_7)):
    user_7[i][2] = user_7[i-1][2] + user_7[i][6]

user_6 = pd.DataFrame(user_6)
user_6.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_7 = pd.DataFrame(user_7)
user_7.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


#for user 8
user_7 = user_7.to_numpy()
user_8 = user_8.to_numpy()
user_8[0][2] = user_7[len(user_7)-1][2] + user_8[0][6]
for i in range(1, len(user_8)):
    user_8[i][2] = user_8[i-1][2] + user_8[i][6]

user_7 = pd.DataFrame(user_7)
user_7.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_8 = pd.DataFrame(user_8)
user_8.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]

#for user 9
user_8 = user_8.to_numpy()
user_9 = user_9.to_numpy()
user_9[0][2] = user_8[len(user_8)-1][2] + user_9[0][6]
for i in range(1, len(user_9)):
    user_9[i][2] = user_9[i-1][2] + user_9[i][6]

user_8 = pd.DataFrame(user_8)
user_8.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_9 = pd.DataFrame(user_9)
user_9.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


#for user 10
user_9 = user_9.to_numpy()
user_10 = user_10.to_numpy()
user_10[0][2] = user_9[len(user_9)-1][2] + user_10[0][6]
for i in range(1, len(user_10)):
    user_10[i][2] = user_10[i-1][2] + user_10[i][6]

user_9 = pd.DataFrame(user_9)
user_9.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]
user_10 = pd.DataFrame(user_10)
user_10.columns = ["User", "Activity", "Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]

#dropping time difference column
user_2 = user_2.drop(['Timedifference'], axis = 1)
user_3 = user_3.drop(['Timedifference'], axis = 1)
user_4 = user_4.drop(['Timedifference'], axis = 1)
user_5 = user_5.drop(['Timedifference'], axis = 1)
user_6 = user_6.drop(['Timedifference'], axis = 1)
user_7 = user_7.drop(['Timedifference'], axis = 1)
user_8 = user_8.drop(['Timedifference'], axis = 1)
user_9 = user_9.drop(['Timedifference'], axis = 1)
user_10 = user_10.drop(['Timedifference'], axis = 1)


#creating one large dataset

#Final_Training_set = pd.DataFrame(user_1, columns = ['User', 'Activity', 'Timeframe', 'X axis', 'Y axis', 'Z axis',])

Final_Training_set = pd.concat([user_1, user_2])
Final_Training_set = pd.concat([Final_Training_set, user_3])
Final_Training_set = pd.concat([Final_Training_set, user_4])
Final_Training_set = pd.concat([Final_Training_set, user_5])
Final_Training_set = pd.concat([Final_Training_set, user_6])
Final_Training_set = pd.concat([Final_Training_set, user_7])
Final_Training_set = pd.concat([Final_Training_set, user_8])
#Final_Training_set = pd.concat([Final_Training_set, user_9])
#Final_Training_set = pd.concat([Final_Training_set, user_10])


#Exporting Final Training Dataset
Export_csv = Final_Training_set.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/final_training_set_8people.csv')


#Creating Testset from User 9 and 10
Final_Test_set = pd.concat([user_9, user_10])
Export2_csv = Final_Test_set.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/final_test_set_2people.csv')




