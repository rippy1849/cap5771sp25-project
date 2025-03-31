import pandas as pd
import numpy as np 
from pathlib import Path

data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'data_2_Mental Health Dataset.csv'
csv2 = 'data_3_Student Mental health.csv'
csv3 = 'Data_2_NumberOfSymptoms.csv'
csv4 = 'Data_3_NumberOfSymptoms.csv'


csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv3_path = Path(data_path + csv3)
csv4_path = Path(data_path + csv4)




df = pd.read_csv(csv1_path)

df['self_employed'] = df['self_employed'].map({np.nan : 'No'})

# rows = []

# for index,row in df.iterrows():

#     s1 = row['Growing_Stress']
#     s2 = row['Changes_Habits']
#     s3 = row['Mood_Swings']
#     s4 = row['Coping_Struggles']
#     s5 = row['Work_Interest']
#     s6 = row['Social_Weakness']
    
    
#     if pd.isna(s1):
#         s1 = 0
#     if pd.isna(s2):
#         s2 = 0
#     if pd.isna(s3):
#         s3 = 0
#     if pd.isna(s4):
#         s4 = 0
#     if pd.isna(s5):
#         s5 = 0
#     if pd.isna(s6):
#         s6 = 0
#     num_symptoms = int(s1) + int(s2) + int(s3) + int(s4) + int(s5) + int(s6)
    
    
#     rows.append(num_symptoms)
    
df2 = pd.read_csv(csv3_path) 
  
    
df['Number of Symptoms'] = df2['Number of Symptoms']

df = df.drop('Timestamp', axis=1)
df = df.drop('mental_health_interview', axis=1)
df = df.drop('self_employed', axis=1)


df.to_csv('Data_2_Pre_One_Hot.csv', index=False)




df = pd.read_csv(csv2_path)

# df['self_employed'] = df['self_employed'].map({np.nan : 'No'})

# rows = []

    
# for index,row in df.iterrows():
    
#     s1 = row['Depression']
#     s2 = row['Anxiety']
#     s3 = row['Panic']
    
#     if pd.isna(s1):
#         s1 = 0
#     if pd.isna(s2):
#         s2 = 0
#     if pd.isna(s3):
#         s3 = 0
        
#     if s1 == 'Yes':
#         s1 = 1
#     else:
#         s1 = 0
    
#     if s2 == 'Yes':
#         s2 = 1
#     else:
#         s2 = 0

#     if s1 == 'Yes':
#         s1 = 1
#     else:
#         s1 = 0
    
#     num_symptoms = int(s1) + int(s2) + int(s3)
    
#     # rows.append([num_symptoms])
#     rows.append(num_symptoms)


    
df = df.drop('Timestamp', axis=1)

df.columns = ['Gender', 'Age','Course','Study Year', 'GPA','Married','Depression','Anxiety','Panic','Treatment']



df2 = pd.read_csv(csv4_path) 
  
    
df['Number of Symptoms'] = df2['Number of Symptoms']

    
df = df.drop('Study Year', axis=1)



# df['Study Year'] = df['Study Year'].map({'year 1' : 'Year 1', 'year 2' : 'Year 2', 'year 3' : 'Year 3', 'year 4' : 'Year 4'})


df.to_csv('Data_3_Pre_One_Hot.csv', index=False)




