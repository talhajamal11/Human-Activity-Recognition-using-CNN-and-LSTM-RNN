import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#For User 1

User_1 = pd.read_csv('acceleration_labelled_data.csv')
User_1 = pd.DataFrame(User_1.iloc[:, 1:6].values)
User_1.columns = ["Activity", "Timeframe", "X axis", "Y axis", "Z axis"]
User_1["Timeframe"] = User_1["Timeframe"] - 0.017856

"""Export_csv = User_1.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/User_1.csv')
"""

#For User 2
User_2 = pd.read_csv('acceleration.csv')
User_2 = pd.DataFrame(User_2.iloc[:, 0:4].values)
User_2.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_2.insert(0, "Activity", "", True)

User_2_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_2.insert(5, "Timedifference", "", True)
User_2 = User_2.to_numpy()
for i in range(1, 33442):
    User_2[i][5] = User_2[i][1] - User_2[i-1][1]

User_2 = pd.DataFrame(User_2)
User_2.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_2 = User_2.to_numpy()
User_2_annotations = User_2_annotations.to_numpy()

for i in range(0, 337):
    for j in range(0, 33442):
        if (User_2[j][1] > User_2_annotations[i][0]) and (User_2[j][1] < User_2_annotations[i][1]):
            User_2[j][0] = User_2_annotations[i][2]
            
        
User_2 = pd.DataFrame(User_2)
User_2.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
        
#dropping empty dataframes at start and end
User_2 = User_2.iloc[446:32897,]

#exporting file
Export_User2_csv = User_2.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/User_2.csv')


#For User 3
User_3 = pd.read_csv('acceleration.csv')
User_3 = pd.DataFrame(User_3.iloc[:, 0:4].values)
User_3.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_3.insert(0, "Activity", "", True)

User_3_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_3.insert(5, "Timedifference", "", True)
User_3 = User_3.to_numpy()
for i in range(1, len(User_3)):
    User_3[i][5] = User_3[i][1] - User_3[i-1][1]

User_3 = pd.DataFrame(User_3)
User_3.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_3 = User_3.to_numpy()
User_3_annotations = User_3_annotations.to_numpy()

for i in range(0, len(User_3_annotations)):
    for j in range(0, len(User_3)):
        if (User_3[j][1] > User_3_annotations[i][0]) and (User_3[j][1] < User_3_annotations[i][1]):
            User_3[j][0] = User_3_annotations[i][2]
            
        
User_3 = pd.DataFrame(User_3)
User_3.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_3_annotations = pd.DataFrame(User_3_annotations)
User_3_annotations.columns = ["Start", "End", "Activity", "Type"]        
#dropping empty dataframes at start and end
User_3 = User_3.iloc[1604:31228,]

#exporting file
Export_User3_csv = User_3.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_3.csv')



#For User 4

User_4 = pd.read_csv('acceleration.csv')
User_4 = pd.DataFrame(User_4.iloc[:, 0:4].values)
User_4.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_4.insert(0, "Activity", "", True)

User_4_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_4.insert(5, "Timedifference", "", True)
User_4 = User_4.to_numpy()
for i in range(1, len(User_4)):
    User_4[i][5] = User_4[i][1] - User_4[i-1][1]

User_4 = pd.DataFrame(User_4)
User_4.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_4 = User_4.to_numpy()
User_4_annotations = User_4_annotations.to_numpy()

for i in range(0, len(User_4_annotations)):
    for j in range(0, len(User_4)):
        if (User_4[j][1] > User_4_annotations[i][0]) and (User_4[j][1] < User_4_annotations[i][1]):
            User_4[j][0] = User_4_annotations[i][2]
            
        
User_4 = pd.DataFrame(User_4)
User_4.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_4_annotations = pd.DataFrame(User_4_annotations)
User_4_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_4 = User_4.iloc[562:30679,]

#exporting file
Export_User4_csv = User_4.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_4.csv')


#For User 5

User_5 = pd.read_csv('acceleration.csv')
User_5 = pd.DataFrame(User_5.iloc[:, 0:4].values)
User_5.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_5.insert(0, "Activity", "", True)

User_5_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_5.insert(5, "Timedifference", "", True)
User_5 = User_5.to_numpy()
for i in range(1, len(User_5)):
    User_5[i][5] = User_5[i][1] - User_5[i-1][1]

User_5 = pd.DataFrame(User_5)
User_5.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_5 = User_5.to_numpy()
User_5_annotations = User_5_annotations.to_numpy()

for i in range(0, len(User_5_annotations)):
    for j in range(0, len(User_5)):
        if (User_5[j][1] > User_5_annotations[i][0]) and (User_5[j][1] < User_5_annotations[i][1]):
            User_5[j][0] = User_5_annotations[i][2]
            
        
User_5 = pd.DataFrame(User_5)
User_5.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_5_annotations = pd.DataFrame(User_5_annotations)
User_5_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_5 = User_5.iloc[950:30633,]

#exporting file
Export_User5_csv = User_5.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_5.csv')


#For User 6

User_6 = pd.read_csv('acceleration.csv')
User_6 = pd.DataFrame(User_6.iloc[:, 0:4].values)
User_6.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_6.insert(0, "Activity", "", True)

User_6_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_6.insert(5, "Timedifference", "", True)
User_6 = User_6.to_numpy()
for i in range(1, len(User_6)):
    User_6[i][5] = User_6[i][1] - User_6[i-1][1]

User_6 = pd.DataFrame(User_6)
User_6.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_6 = User_6.to_numpy()
User_6_annotations = User_6_annotations.to_numpy()

for i in range(0, len(User_6_annotations)):
    for j in range(0, len(User_6)):
        if (User_6[j][1] > User_6_annotations[i][0]) and (User_6[j][1] < User_6_annotations[i][1]):
            User_6[j][0] = User_6_annotations[i][2]
            
        
User_6 = pd.DataFrame(User_6)
User_6.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_6_annotations = pd.DataFrame(User_6_annotations)
User_6_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_6 = User_6.iloc[717:17329,]

#exporting file
Export_User6_csv = User_6.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_6.csv')


#For User 7

User_7 = pd.read_csv('acceleration.csv')
User_7 = pd.DataFrame(User_7.iloc[:, 0:4].values)
User_7.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_7.insert(0, "Activity", "", True)

User_7_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_7.insert(5, "Timedifference", "", True)
User_7 = User_7.to_numpy()
for i in range(1, len(User_7)):
    User_7[i][5] = User_7[i][1] - User_7[i-1][1]

User_7 = pd.DataFrame(User_7)
User_7.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_7 = User_7.to_numpy()
User_7_annotations = User_7_annotations.to_numpy()

for i in range(0, len(User_7_annotations)):
    for j in range(0, len(User_7)):
        if (User_7[j][1] > User_7_annotations[i][0]) and (User_7[j][1] < User_7_annotations[i][1]):
            User_7[j][0] = User_7_annotations[i][2]
            
        
User_7 = pd.DataFrame(User_7)
User_7.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_7_annotations = pd.DataFrame(User_7_annotations)
User_7_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_7 = User_7.iloc[152:29269,]

#exporting file
Export_User7_csv = User_7.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_7.csv')


#For User 8

User_8 = pd.read_csv('acceleration.csv')
User_8 = pd.DataFrame(User_8.iloc[:, 0:4].values)
User_8.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_8.insert(0, "Activity", "", True)

User_8_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_8.insert(5, "Timedifference", "", True)
User_8 = User_8.to_numpy()
for i in range(1, len(User_8)):
    User_8[i][5] = User_8[i][1] - User_8[i-1][1]

User_8 = pd.DataFrame(User_8)
User_8.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_8 = User_8.to_numpy()
User_8_annotations = User_8_annotations.to_numpy()

for i in range(0, len(User_8_annotations)):
    for j in range(0, len(User_8)):
        if (User_8[j][1] > User_8_annotations[i][0]) and (User_8[j][1] < User_8_annotations[i][1]):
            User_8[j][0] = User_8_annotations[i][2]
            
        
User_8 = pd.DataFrame(User_8)
User_8.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_8_annotations = pd.DataFrame(User_8_annotations)
User_8_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_8 = User_8.iloc[837:30073,]

#exporting file
Export_User8_csv = User_8.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_8.csv')


#For User 9

User_9 = pd.read_csv('acceleration.csv')
User_9 = pd.DataFrame(User_9.iloc[:, 0:4].values)
User_9.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_9.insert(0, "Activity", "", True)

User_9_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_9.insert(5, "Timedifference", "", True)
User_9 = User_9.to_numpy()
for i in range(1, len(User_9)):
    User_9[i][5] = User_9[i][1] - User_9[i-1][1]

User_9 = pd.DataFrame(User_9)
User_9.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_9 = User_9.to_numpy()
User_9_annotations = User_9_annotations.to_numpy()

for i in range(0, len(User_9_annotations)):
    for j in range(0, len(User_9)):
        if (User_9[j][1] > User_9_annotations[i][0]) and (User_9[j][1] < User_9_annotations[i][1]):
            User_9[j][0] = User_9_annotations[i][2]
            
        
User_9 = pd.DataFrame(User_9)
User_9.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_9_annotations = pd.DataFrame(User_9_annotations)
User_9_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_9 = User_9.iloc[210:27920,]

#exporting file
Export_User9_csv = User_9.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_9.csv')

#For User 10

User_10 = pd.read_csv('acceleration.csv')
User_10 = pd.DataFrame(User_10.iloc[:, 0:4].values)
User_10.columns = ["Timeframe", "X axis", "Y axis", "Z axis"]
User_10.insert(0, "Activity", "", True)

User_10_annotations = pd.read_csv('annotations_0.csv')

#adding timedifference column
User_10.insert(5, "Timedifference", "", True)
User_10 = User_10.to_numpy()
for i in range(1, len(User_10)):
    User_10[i][5] = User_10[i][1] - User_10[i-1][1]

User_10 = pd.DataFrame(User_10)
User_10.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]


User_10 = User_10.to_numpy()
User_10_annotations = User_10_annotations.to_numpy()

for i in range(0, len(User_10_annotations)):
    for j in range(0, len(User_10)):
        if (User_10[j][1] > User_10_annotations[i][0]) and (User_10[j][1] < User_10_annotations[i][1]):
            User_10[j][0] = User_10_annotations[i][2]
            
        
User_10 = pd.DataFrame(User_10)
User_10.columns = ["Activity","Timeframe", "X axis", "Y axis", "Z axis", "Timedifference"]  
User_10_annotations = pd.DataFrame(User_10_annotations)
User_10_annotations.columns = ["Start", "End", "Activity", "index"]        
#dropping empty dataframes at start and end
User_10 = User_10.iloc[260:32899,]

#exporting file
Export_User10_csv = User_10.to_csv(r'/Users/talhajamal/Documents/Year 3/Individual Project/Files for project/Final Training Dataset/User_10.csv')
