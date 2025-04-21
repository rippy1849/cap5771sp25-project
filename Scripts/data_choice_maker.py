import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib



data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_2_Pre_One_Hot.csv'
csv2 = 'Data_3_Pre_One_Hot.csv'


csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)


df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)


# for index, row in df1.iterrows():

# data = {'data1' :[1,2], 'data2' : [1,2,3]}

# df = pd.DataFrame(data)

# print(df)
    