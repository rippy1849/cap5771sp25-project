import pandas as pd
import numpy as np
from pathlib import Path


data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_1_1_Features.csv'
csv2 = 'Data_1_2_Features.csv'


csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)


df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

prev_country = ""

entry_row = []
rows = []
columns = ['Country','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']

for index, row in df1.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,schizophrenia]
            
    else:
        entry_row.append(schizophrenia)
        
df = pd.DataFrame(rows,columns=columns)

# df = df.set_index('Country')

df.to_csv('Data_1_1_Schizophrenia.csv',index=False)

# print(df.loc[df['Country'].isin(['Afghanistan'])])
rows = []
for index, row in df1.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,depressive]
            
    else:
        entry_row.append(depressive)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_1_Depression.csv')

rows = []

for index, row in df1.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    # depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,anxiety]
            
    else:
        entry_row.append(anxiety)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_1_Anxiety.csv')


rows = []
for index, row in df1.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    # depressive = row['Depressive']
    # anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,bipolar]
            
    else:
        entry_row.append(bipolar)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_1_Bipolar.csv')

rows = []
for index, row in df1.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    # depressive = row['Depressive']
    # anxiety = row['Anxiety']
    # bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,eating]
            
    else:
        entry_row.append(eating)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_1_Eating.csv')

rows = []

for index, row in df2.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,schizophrenia]
            
    else:
        entry_row.append(schizophrenia)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_2_Schizophrenia.csv')

rows = []
for index, row in df2.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,depressive]
            
    else:
        entry_row.append(depressive)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_2_Depression.csv')

rows = []

for index, row in df2.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    # depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,anxiety]
            
    else:
        entry_row.append(anxiety)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_2_Anxiety.csv')

rows = []

for index, row in df2.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    # depressive = row['Depressive']
    # anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,bipolar]
            
    else:
        entry_row.append(bipolar)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_2_Bipolar.csv')

rows = []
for index, row in df2.iterrows():
    
    country = row['Country']
    # decade = row['Decade']
    # schizophrenia = row['Schizophrenia']
    # depressive = row['Depressive']
    # anxiety = row['Anxiety']
    # bipolar = row['Bipolar']
    # eating = row['Eating']
    
    if prev_country != country:
        if index != 0:
            rows.append(entry_row)
        prev_country = country
        entry_row = [country,eating]
            
    else:
        entry_row.append(eating)
        
df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_2_Eating.csv')

