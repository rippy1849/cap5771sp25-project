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

prevDecade = 0
prevCountry = ""


s_avg_decade = 0
d_avg_decade = 0
a_avg_decade = 0
b_avg_decade = 0
e_avg_decade = 0
decade_count = 0


rows = []

for index, row in df1.iterrows():
    
    country = row['Country']
    decade = row['Decade']
    schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prevDecade != int(decade):
        

        
        if index != 0:
            rows.append([prevCountry,prevDecade,s_avg_decade/decade_count,d_avg_decade/decade_count,a_avg_decade/decade_count,b_avg_decade/decade_count,e_avg_decade/decade_count])
            
        if prevCountry != country:
            prevCountry = country
        
        prevDecade = int(decade)
        
        s_avg_decade = schizophrenia
        d_avg_decade = depressive
        a_avg_decade = anxiety
        b_avg_decade = bipolar
        e_avg_decade = eating
        decade_count = 1
    else:
        s_avg_decade += schizophrenia
        d_avg_decade += depressive
        a_avg_decade += anxiety
        b_avg_decade += bipolar
        e_avg_decade += eating   
        decade_count += 1

rows.append([prevCountry,prevDecade,s_avg_decade/decade_count,d_avg_decade/decade_count,a_avg_decade/decade_count,b_avg_decade/decade_count,e_avg_decade/decade_count])

      
      
      
columns = ['Country', 'Decade','Schizophrenia','Depressive','Anxiety','Bipolar','Eating']  
df3 = pd.DataFrame(rows,columns=columns)
            

prevDecade = 0
prevCountry = ""


s_avg_decade = 0
d_avg_decade = 0
a_avg_decade = 0
b_avg_decade = 0
e_avg_decade = 0
decade_count = 0


rows = []

for index, row in df2.iterrows():
    
    country = row['Country']
    decade = row['Decade']
    schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    if prevDecade != int(decade):
                
        if index != 0:
            rows.append([prevCountry,prevDecade,s_avg_decade/decade_count,d_avg_decade/decade_count,a_avg_decade/decade_count,b_avg_decade/decade_count,e_avg_decade/decade_count])
            
        if prevCountry != country:
            prevCountry = country
        
        prevDecade = int(decade)
        
        s_avg_decade = schizophrenia
        d_avg_decade = depressive
        a_avg_decade = anxiety
        b_avg_decade = bipolar
        e_avg_decade = eating
        decade_count = 1
    else:
        s_avg_decade += schizophrenia
        d_avg_decade += depressive
        a_avg_decade += anxiety
        b_avg_decade += bipolar
        e_avg_decade += eating   
        decade_count += 1

rows.append([prevCountry,prevDecade,s_avg_decade/decade_count,d_avg_decade/decade_count,a_avg_decade/decade_count,b_avg_decade/decade_count,e_avg_decade/decade_count])

      
      
      
columns = ['Country', 'Decade','Schizophrenia','Depressive','Anxiety','Bipolar','Eating']  
df4 = pd.DataFrame(rows,columns=columns)

# print(df4)
        
# print(df4)

# exit()

rows = []


prevCountry = ""

for index,row in df3.iterrows():
    

    
    country = row['Country']
    decade = row['Decade']
    schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    
    if prevCountry != country:
        prevCountry = country
        index_count = 0
        # print(country)
    
    index = np.array(df4.index[df4.Country == country].values)
    # print(index)
    index_value = str(index).strip('[]').split(" ")
    # print(len(index_value), index_value)
    # print(index_value)
    # if len(index_value) == 3:
    #     print(index_value[0],index_value[1],index_value[2])
    #     print(int(index_value[0]),int(index_value[1]),int(index_value[2]))
    
    if len(index_value) == 3:
        row2 = df4.iloc[int(index_value[index_count])]    
        # print(row2)
        
        # decade = row2['Decade']
        schizophrenia2 = row2['Schizophrenia']
        depressive2 = row2['Depressive']
        anxiety2 = row2['Anxiety']
        bipolar2 = row2['Bipolar']
        eating2 = row2['Eating']
        
        rows.append([country,decade,schizophrenia,schizophrenia2,depressive,depressive2,anxiety,anxiety2,bipolar,bipolar2,eating,eating2])
        
    index_count += 1
    # print(index_count)
    
        
        
        
columns = ['Country', 'Decade','Schizophrenia Avg Percent','Schizophrenia Avg DALYs' ,'Depressive Avg Percent','Depressive Avg DALYs','Anxiety Avg Percent', 'Anxiety Avg DALYs','Bipolar Avg Percent', 'Bipolar Avg DALYs','Eating Avg Percent', 'Eating Avg DALYs']
df = pd.DataFrame(rows,columns=columns)

df.to_csv('MLModelAvgPercentDALY.csv', index=False)

# df.to_csv('MLModelAvgPercentDALY.csv', index=False)
    

