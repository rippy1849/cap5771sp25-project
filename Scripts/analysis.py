import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'
# data_path = ''

# data_folder = Path(data_path + csv1)
# print(data_folder)

csv1 = 'data_1_1-mental-illnesses-prevalence.csv'
csv2 = 'data_1_2-burden-disease-from-each-mental-illness.csv'
csv3 = 'data_1_3-adult-population-covered-in-primary-data-on-the-prevalence-of-major-depression.csv'
csv4 = 'data_1_4-adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'
csv5 = 'data_1_5-anxiety-disorders-treatment-gap.csv'
csv6 = 'data_1_6-depressive-symptoms-across-us-population.csv'
csv7 = 'data_1_7-number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv'
csv8 = 'data_2_Mental Health Dataset.csv'
csv9 = 'data_3_Student Mental health.csv'

#DATA 1 CSV 1
csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv3_path = Path(data_path + csv3)
csv4_path = Path(data_path + csv4)
csv5_path = Path(data_path + csv5)
csv6_path = Path(data_path + csv6)
csv7_path = Path(data_path + csv7)
csv8_path = Path(data_path + csv8)
csv9_path = Path(data_path + csv9)





df = pd.read_csv(csv1_path)


total_number_of_entries = len(df)
schizophrenia_avg = 0
depressive_avg = 0
anxiety_avg = 0
bipolar_avg = 0
eating_avg = 0

schizophrenia_median = 0
depressive_median = 0
anxiety_median = 0
bipolar_median = 0
eating_median = 0

schizophrenia_median_array = []
depressive_median_array = []
anxiety_median_array = []
bipolar_median_array = []
eating_median_array = []

for index, row in df.iterrows():
    schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
    depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
    anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
    bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
    eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    
    country = row['Entity']
    
    schizophrenia_avg += schizophrenia
    depressive_avg += depressive
    anxiety_avg += anxiety
    bipolar_avg += bipolar
    eating_avg += eating
    
    # if anxiety > 8:
    #     print(country)
    
    
    schizophrenia_median_array.append(float(schizophrenia))
    depressive_median_array.append(float(depressive))
    anxiety_median_array.append(float(anxiety))
    bipolar_median_array.append(float(bipolar))
    eating_median_array.append(float(eating))
    
schizophrenia_median = np.median(schizophrenia_median_array)
depressive_median = np.median(depressive_median_array)
anxiety_median = np.median(anxiety_median_array)
bipolar_median = np.median(bipolar_median_array)
eating_median = np.median(eating_median_array)
    
    

schizophrenia_avg = schizophrenia_avg/total_number_of_entries
depressive_avg = depressive_avg/total_number_of_entries
anxiety_avg = anxiety_avg/total_number_of_entries
bipolar_avg = bipolar_avg/total_number_of_entries
eating_avg = eating_avg/total_number_of_entries


schizophrenia_std_dev = 0
depressive_std_dev = 0
anxiety_std_dev = 0
bipolar_std_dev = 0
eating_std_dev = 0

for index, row in df.iterrows():
    schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
    depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
    anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
    bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
    eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    
    schizophrenia_std_dev += np.power((schizophrenia-schizophrenia_avg),2)
    depressive_std_dev += np.power((depressive-depressive_avg),2)
    anxiety_std_dev += np.power((anxiety-anxiety_avg),2)
    bipolar_std_dev += np.power((bipolar-bipolar_avg),2)
    eating_std_dev += np.power((eating-eating_avg),2)

schizophrenia_std_dev = np.power((schizophrenia_std_dev/(total_number_of_entries-1)),1/2)
depressive_std_dev = np.power((depressive_std_dev/(total_number_of_entries-1)),1/2)
anxiety_std_dev = np.power((anxiety_std_dev/(total_number_of_entries-1)),1/2)
bipolar_std_dev = np.power((bipolar_std_dev/(total_number_of_entries-1)),1/2)
eating_std_dev = np.power((eating_std_dev/(total_number_of_entries-1)),1/2)

# print("Schizophrenia ",schizophrenia_avg,schizophrenia_median, schizophrenia_std_dev)
# print("Depressive ",depressive_avg,depressive_median, depressive_std_dev)
# print("Anxiety ",anxiety_avg,anxiety_median, anxiety_std_dev)
# print("Bipolar ",bipolar_avg,bipolar_median, bipolar_std_dev)
# print("Eating ",eating_avg,eating_median, eating_std_dev)

schizophrenia_skew = 3*(schizophrenia_avg-schizophrenia_median)/schizophrenia_std_dev
depressive_skew = 3*(depressive_avg-depressive_median)/depressive_std_dev
anxiety_skew = 3*(anxiety_avg-anxiety_median)/anxiety_std_dev
bipolar_skew = 3*(bipolar_avg-bipolar_median)/bipolar_std_dev
eating_skew = 3*(eating_avg-eating_median)/eating_std_dev

# print("schizophrenia_skew ", schizophrenia_skew)
# print("depressive_skew ", depressive_skew)
# print("anxiety_skew", anxiety_skew)
# print("bipolar_skew", bipolar_skew)
# print("eating_skew", eating_skew)


schizophrenia_z_score_array = []
depressive_z_score_array = []
anxiety_z_score_array = []
bipolar_z_score_array = []
eating_z_score_array = []

for index, row in df.iterrows():
    schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
    depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
    anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
    bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
    eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    
    schizophrenia_z_score = (schizophrenia - schizophrenia_avg)/schizophrenia_std_dev
    schizophrenia_z_score_array.append(schizophrenia_z_score)
    
    depressive_z_score = (depressive - depressive_avg)/depressive_std_dev
    depressive_z_score_array.append(depressive_z_score)
    
    anxiety_z_score = (anxiety - anxiety_avg)/anxiety_std_dev
    anxiety_z_score_array.append(anxiety_z_score)
    
    bipolar_z_score = (bipolar - bipolar_avg)/bipolar_std_dev
    bipolar_z_score_array.append(bipolar_z_score)
    
    eating_z_score = (eating-eating_avg)/eating_std_dev


schizophrenia_outliers = []
depressive_outliers = []
anxiety_outliers = []
bipolar_outliers = []
eating_outliers = []


for i in range(0,len(schizophrenia_z_score_array)):
    z_score = float(schizophrenia_z_score_array[i])
    if abs(z_score) >= 3:
        schizophrenia_outliers.append([i,z_score])
        
for i in range(0,len(depressive_z_score_array)):
    z_score = float(depressive_z_score_array[i])
    if abs(z_score) >= 3:
        depressive_outliers.append([i,z_score])
        
for i in range(0,len(anxiety_z_score_array)):
    z_score = float(anxiety_z_score_array[i])
    if abs(z_score) >= 3:
        anxiety_outliers.append([i,z_score])
        
for i in range(0,len(bipolar_z_score_array)):
    z_score = float(bipolar_z_score_array[i])
    if abs(z_score) >= 3:
        bipolar_outliers.append([i,z_score])
        
for i in range(0,len(eating_z_score_array)):
    z_score = float(eating_z_score_array[i])
    if abs(z_score) >= 3:
        # print("hi")
        eating_outliers.append([i,z_score])
        


schizophrenia_df=pd.DataFrame(schizophrenia_outliers, columns=['data index', 'z score']) 
schizophrenia_df.to_csv(data_path + 'data_1_1_schizophrenia_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

depressive_df=pd.DataFrame(depressive_outliers, columns=['data index', 'z score']) 
depressive_df.to_csv(data_path + 'data_1_1_depressive_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

anxiety_df=pd.DataFrame(anxiety_outliers, columns=['data index', 'z score']) 
anxiety_df.to_csv(data_path + 'data_1_1_anxiety_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

bipolar_df=pd.DataFrame(bipolar_outliers, columns=['data index', 'z score']) 
bipolar_df.to_csv(data_path + 'data_1_1_bipolar_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

eating_df=pd.DataFrame(eating_outliers, columns=['data index', 'z score']) 
eating_df.to_csv(data_path + 'data_1_1_eating_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)





#DATA 1 CSV 2

df = pd.read_csv(csv2_path)

total_number_of_entries = len(df)
schizophrenia_avg = 0
depressive_avg = 0
anxiety_avg = 0
bipolar_avg = 0
eating_avg = 0

schizophrenia_median = 0
depressive_median = 0
anxiety_median = 0
bipolar_median = 0
eating_median = 0

schizophrenia_median_array = []
depressive_median_array = []
anxiety_median_array = []
bipolar_median_array = []
eating_median_array = []



for index, row in df.iterrows():
    depressive = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders']
    schizophrenia = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia']
    anxiety = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder']
    bipolar = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders']
    eating = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders']
    
    country = row['Entity']
    
    schizophrenia_avg += schizophrenia
    depressive_avg += depressive
    anxiety_avg += anxiety
    bipolar_avg += bipolar
    eating_avg += eating

    # # print(anxiety)
    # if schizophrenia > 275:
    #     print(country)


    schizophrenia_median_array.append(float(schizophrenia))
    depressive_median_array.append(float(depressive))
    anxiety_median_array.append(float(anxiety))
    bipolar_median_array.append(float(bipolar))
    eating_median_array.append(float(eating))
    
schizophrenia_median = np.median(schizophrenia_median_array)
depressive_median = np.median(depressive_median_array)
anxiety_median = np.median(anxiety_median_array)
bipolar_median = np.median(bipolar_median_array)
eating_median = np.median(eating_median_array)




schizophrenia_avg = schizophrenia_avg/total_number_of_entries
depressive_avg = depressive_avg/total_number_of_entries
anxiety_avg = anxiety_avg/total_number_of_entries
bipolar_avg = bipolar_avg/total_number_of_entries
eating_avg = eating_avg/total_number_of_entries

schizophrenia_std_dev = 0
depressive_std_dev = 0
anxiety_std_dev = 0
bipolar_std_dev = 0
eating_std_dev = 0

for index, row in df.iterrows():
    depressive = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders']
    schizophrenia = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia']
    anxiety = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder']
    bipolar = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders']
    eating = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders']
    
    schizophrenia_std_dev += np.power((schizophrenia-schizophrenia_avg),2)
    depressive_std_dev += np.power((depressive-depressive_avg),2)
    anxiety_std_dev += np.power((anxiety-anxiety_avg),2)
    bipolar_std_dev += np.power((bipolar-bipolar_avg),2)
    eating_std_dev += np.power((eating-eating_avg),2)

schizophrenia_std_dev = np.power((schizophrenia_std_dev/(total_number_of_entries-1)),1/2)
depressive_std_dev = np.power((depressive_std_dev/(total_number_of_entries-1)),1/2)
anxiety_std_dev = np.power((anxiety_std_dev/(total_number_of_entries-1)),1/2)
bipolar_std_dev = np.power((bipolar_std_dev/(total_number_of_entries-1)),1/2)
eating_std_dev = np.power((eating_std_dev/(total_number_of_entries-1)),1/2)

# print("Schizophrenia ",schizophrenia_avg,schizophrenia_median, schizophrenia_std_dev)
# print("Depressive ",depressive_avg,depressive_median, depressive_std_dev)
# print("Anxiety ",anxiety_avg,anxiety_median, anxiety_std_dev)
# print("Bipolar ",bipolar_avg,bipolar_median, bipolar_std_dev)
# print("Eating ",eating_avg,eating_median, eating_std_dev)


schizophrenia_skew = 3*(schizophrenia_avg-schizophrenia_median)/schizophrenia_std_dev
depressive_skew = 3*(depressive_avg-depressive_median)/depressive_std_dev
anxiety_skew = 3*(anxiety_avg-anxiety_median)/anxiety_std_dev
bipolar_skew = 3*(bipolar_avg-bipolar_median)/bipolar_std_dev
eating_skew = 3*(eating_avg-eating_median)/eating_std_dev

# print("schizophrenia_skew ", schizophrenia_skew)
# print("depressive_skew ", depressive_skew)
# print("anxiety_skew", anxiety_skew)
# print("bipolar_skew", bipolar_skew)
# print("eating_skew", eating_skew)


schizophrenia_z_score_array = []
depressive_z_score_array = []
anxiety_z_score_array = []
bipolar_z_score_array = []
eating_z_score_array = []

for index, row in df.iterrows():
    depressive = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders']
    schizophrenia = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia']
    anxiety = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder']
    bipolar = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders']
    eating = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders']
    
    schizophrenia_z_score = (schizophrenia - schizophrenia_avg)/schizophrenia_std_dev
    schizophrenia_z_score_array.append(schizophrenia_z_score)
    
    depressive_z_score = (depressive - depressive_avg)/depressive_std_dev
    depressive_z_score_array.append(depressive_z_score)
    
    anxiety_z_score = (anxiety - anxiety_avg)/anxiety_std_dev
    anxiety_z_score_array.append(anxiety_z_score)
    
    bipolar_z_score = (bipolar - bipolar_avg)/bipolar_std_dev
    bipolar_z_score_array.append(bipolar_z_score)
    
    eating_z_score = (eating-eating_avg)/eating_std_dev
    

schizophrenia_outliers = []
depressive_outliers = []
anxiety_outliers = []
bipolar_outliers = []
eating_outliers = []


for i in range(0,len(schizophrenia_z_score_array)):
    z_score = float(schizophrenia_z_score_array[i])
    if abs(z_score) >= 3:
        schizophrenia_outliers.append([i,z_score])
        
for i in range(0,len(depressive_z_score_array)):
    z_score = float(depressive_z_score_array[i])
    if abs(z_score) >= 3:
        depressive_outliers.append([i,z_score])
        
for i in range(0,len(anxiety_z_score_array)):
    z_score = float(anxiety_z_score_array[i])
    if abs(z_score) >= 3:
        anxiety_outliers.append([i,z_score])
        
for i in range(0,len(bipolar_z_score_array)):
    z_score = float(bipolar_z_score_array[i])
    if abs(z_score) >= 3:
        bipolar_outliers.append([i,z_score])
        
for i in range(0,len(eating_z_score_array)):
    z_score = float(eating_z_score_array[i])
    if abs(z_score) >= 3:
        # print("hi")
        eating_outliers.append([i,z_score])
        


schizophrenia_df=pd.DataFrame(schizophrenia_outliers, columns=['data index', 'z score']) 
schizophrenia_df.to_csv(data_path + 'data_1_2_schizophrenia_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

depressive_df=pd.DataFrame(depressive_outliers, columns=['data index', 'z score']) 
depressive_df.to_csv(data_path + 'data_1_2_depressive_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

anxiety_df=pd.DataFrame(anxiety_outliers, columns=['data index', 'z score']) 
anxiety_df.to_csv(data_path + 'data_1_2_anxiety_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

bipolar_df=pd.DataFrame(bipolar_outliers, columns=['data index', 'z score']) 
bipolar_df.to_csv(data_path + 'data_1_2_bipolar_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

eating_df=pd.DataFrame(eating_outliers, columns=['data index', 'z score']) 
eating_df.to_csv(data_path + 'data_1_2_eating_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)



#DATA 1, CSV 3

df = pd.read_csv(csv3_path)

total_number_of_entries = len(df)
depression_score_avg = 0
depression_score_median_array = []
depression_score_median = 0


for index, row in df.iterrows():
    depression_score = row['Major depression']
    depression_score_avg += depression_score
    
    depression_score_median_array.append(depression_score)

depression_score_avg = depression_score_avg/total_number_of_entries
depression_score_median = np.median(depression_score_median_array)

depression_std_dev = 0

for index, row in df.iterrows():

    depression_score = row['Major depression']
    depression_std_dev += np.power((depression_score-depression_score_avg),2)

depression_std_dev = np.power((depression_std_dev/(total_number_of_entries-1)),1/2)


# print("Depression Score ", depression_score_avg, depression_score_median, depression_std_dev)

depression_score_z_score_array = []


for index, row in df.iterrows():
    depression_score = row['Major depression']
    
    depression_score_z_score = (depression_score - depression_score_avg)/depression_std_dev
    depression_score_z_score_array.append(depression_score_z_score)

depression_score_outliers = []

for i in range(0,len(depression_score_z_score_array)):
    z_score = float(depression_score_z_score_array[i])
    if abs(z_score) >= 3:
        depression_score_outliers.append([i,z_score])
        
depression_score_df=pd.DataFrame(depression_score_outliers, columns=['data index', 'z score']) 
depression_score_df.to_csv(data_path + 'data_1_3_depression_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)


#DATA 1, CSV 4

df = pd.read_csv(csv4_path)

# print(df)
total_number_of_entries = len(df)
depression_score_avg = 0
bipolar_score_avg = 0
eating_score_avg = 0
dysthymia_score_avg = 0
schizophrenia_score_avg = 0
anxiety_score_avg = 0


depression_median = 0
bipolar_median = 0
eating_median = 0
dysthymia_median = 0
schizophrenia_median = 0
anxiety_median = 0

depression_median_array = []
bipolar_median_array = []
eating_median_array = []
dysthymia_median_array = []
schizophrenia_median_array = []
anxiety_median_array = []


for index, row in df.iterrows():
    depression = row['Major depression']
    bipolar = row['Bipolar disorder']
    eating = row['Eating disorders']
    dysthymia = row['Dysthymia']
    schizophrenia = row['Schizophrenia']
    anxiety = row['Anxiety disorders']

    depression_score_avg += float(depression)
    bipolar_score_avg += float(bipolar)
    eating_score_avg += float(eating)
    dysthymia_score_avg += float(dysthymia)
    schizophrenia_score_avg += float(schizophrenia)
    anxiety_score_avg += float(anxiety)
    
    depression_median_array.append(depression)
    bipolar_median_array.append(bipolar)
    eating_median_array.append(eating)
    dysthymia_median_array.append(dysthymia)
    schizophrenia_median_array.append(schizophrenia)
    anxiety_median_array.append(anxiety)
    

depression_median = np.median(depression_median_array)
bipolar_median = np.median(bipolar_median_array)
eating_median = np.median(eating_median_array)
dysthymia_median = np.median(dysthymia_median_array)
schizophrenia_median = np.median(schizophrenia_median_array)
anxiety_median = np.median(anxiety_median_array)



depression_score_avg = depression_score_avg/total_number_of_entries
bipolar_score_avg = bipolar_score_avg/total_number_of_entries
eating_score_avg = eating_score_avg/total_number_of_entries
dysthymia_score_avg = dysthymia_score_avg/total_number_of_entries
schizophrenia_score_avg = schizophrenia_score_avg/total_number_of_entries
anxiety_score_avg = anxiety_score_avg/total_number_of_entries


depression_std_dev = 0
bipolar_std_dev = 0
eating_std_dev = 0
dysthymia_std_dev = 0
schizophrenia_std_dev = 0
anxiety_std_dev = 0


for index, row in df.iterrows():
    depression = row['Major depression']
    bipolar = row['Bipolar disorder']
    eating = row['Eating disorders']
    dysthymia = row['Dysthymia']
    schizophrenia = row['Schizophrenia']
    anxiety = row['Anxiety disorders']
    
    depression_std_dev += np.power((depression-depression_score_avg),2)
    bipolar_std_dev += np.power((bipolar-bipolar_score_avg),2)
    eating_std_dev += np.power((eating-eating_score_avg),2)
    dysthymia_std_dev += np.power((dysthymia-dysthymia_score_avg),2)
    schizophrenia_std_dev += np.power((schizophrenia-schizophrenia_score_avg),2)
    anxiety_std_dev += np.power((anxiety-anxiety_score_avg),2)
    
    
depression_std_dev = np.power((depression_std_dev/(total_number_of_entries-1)),1/2)
bipolar_std_dev = np.power((bipolar_std_dev/(total_number_of_entries-1)),1/2)
eating_std_dev = np.power((eating_std_dev/(total_number_of_entries-1)),1/2)
dysthymia_std_dev = np.power((dysthymia_std_dev/(total_number_of_entries-1)),1/2)
schizophrenia_std_dev = np.power((schizophrenia_std_dev/(total_number_of_entries-1)),1/2)
anxiety_std_dev = np.power((anxiety_std_dev/(total_number_of_entries-1)),1/2)




# print("Depression ",depression_score_avg,depression_median, depression_std_dev)
# print("Bipolar ",bipolar_score_avg,bipolar_median, bipolar_std_dev)
# print("Eating ",eating_score_avg,eating_median, eating_std_dev)
# print("Dysthymia ", dysthymia_score_avg,dysthymia_median,dysthymia_std_dev)
# print("Schizophrenia ",schizophrenia_score_avg,schizophrenia_median, schizophrenia_std_dev)
# print("Anxiety ",anxiety_score_avg,anxiety_median, anxiety_std_dev)


schizophrenia_skew = 3*(schizophrenia_score_avg-schizophrenia_median)/schizophrenia_std_dev
depression_skew = 3*(depression_score_avg-depression_median)/depression_std_dev
anxiety_skew = 3*(anxiety_score_avg-anxiety_median)/anxiety_std_dev
bipolar_skew = 3*(bipolar_score_avg-bipolar_median)/bipolar_std_dev
eating_skew = 3*(eating_score_avg-eating_median)/eating_std_dev
dysthymia_skew = 3*(dysthymia_score_avg-dysthymia_median)/dysthymia_std_dev

# print("schizophrenia_skew ", schizophrenia_skew)
# print("depressive_skew ", depressive_skew)
# print("anxiety_skew", anxiety_skew)
# print("bipolar_skew", bipolar_skew)
# print("eating_skew", eating_skew)
# print("Dysthymia ", dysthymia_skew)



depression_z_score_array = []
bipolar_z_score_array = []
eating_z_score_array = []
dysthymia_z_score_array = []
schizophrenia_z_score_array = []
anxiety_z_score_array = []


for index, row in df.iterrows():
    depression = row['Major depression']
    bipolar = row['Bipolar disorder']
    eating = row['Eating disorders']
    dysthymia = row['Dysthymia']
    schizophrenia = row['Schizophrenia']
    anxiety = row['Anxiety disorders']
    
    depression_z_score = (depression - depression_score_avg)/depression_std_dev
    depression_z_score_array.append(depression_z_score)
    
    bipolar_z_score = (bipolar - bipolar_score_avg)/bipolar_std_dev
    bipolar_z_score_array.append(bipolar_z_score)
    
    eating_z_score = (eating - eating_score_avg)/eating_std_dev
    eating_z_score_array.append(eating_z_score)

    dysthymia_z_score = (dysthymia - dysthymia_score_avg)/dysthymia_std_dev
    dysthymia_z_score_array.append(dysthymia_z_score)

    schizophrenia_z_score = (schizophrenia - schizophrenia_score_avg)/schizophrenia_std_dev
    schizophrenia_z_score_array.append(schizophrenia_z_score)
    
    anxiety_z_score = (anxiety - anxiety_score_avg)/anxiety_std_dev
    anxiety_z_score_array.append(anxiety_z_score)
    
    
depression_outliers = []
bipolar_outliers = []
eating_outliers = []
dysthymia_outliers = []
schizophrenia_outliers = []
anxiety_outliers = []


for i in range(0,len(depression_z_score_array)):
    z_score = float(depression_z_score_array[i])
    if abs(z_score) >= 3:
        depression_outliers.append([i,z_score])
        
for i in range(0,len(bipolar_z_score_array)):
    z_score = float(bipolar_z_score_array[i])
    if abs(z_score) >= 3:
        bipolar_outliers.append([i,z_score])

for i in range(0,len(eating_z_score_array)):
    z_score = float(eating_z_score_array[i])
    if abs(z_score) >= 3:
        eating_outliers.append([i,z_score])

for i in range(0,len(dysthymia_z_score_array)):
    z_score = float(dysthymia_z_score_array[i])
    if abs(z_score) >= 3:
        dysthymia_outliers.append([i,z_score])
        
for i in range(0,len(schizophrenia_z_score_array)):
    z_score = float(schizophrenia_z_score_array[i])
    if abs(z_score) >= 3:
        schizophrenia_outliers.append([i,z_score])
        
for i in range(0,len(anxiety_z_score_array)):
    z_score = float(anxiety_z_score_array[i])
    if abs(z_score) >= 3:
        anxiety_outliers.append([i,z_score])
        
        
depression_score_df=pd.DataFrame(depression_outliers, columns=['data index', 'z score']) 
depression_score_df.to_csv(data_path + 'data_1_4_depression_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

bipolar_score_df=pd.DataFrame(bipolar_outliers, columns=['data index', 'z score']) 
bipolar_score_df.to_csv(data_path + 'data_1_4_bipolar_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

eating_score_df=pd.DataFrame(eating_outliers, columns=['data index', 'z score']) 
eating_score_df.to_csv(data_path + 'data_1_4_eating_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

dysthymia_score_df=pd.DataFrame(dysthymia_outliers, columns=['data index', 'z score']) 
dysthymia_score_df.to_csv(data_path + 'data_1_4_dysthymia_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

schizophrenia_score_df=pd.DataFrame(schizophrenia_outliers, columns=['data index', 'z score']) 
schizophrenia_score_df.to_csv(data_path + 'data_1_4_schizophrenia_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

anxiety_score_df=pd.DataFrame(anxiety_outliers, columns=['data index', 'z score']) 
anxiety_score_df.to_csv(data_path + 'data_1_4_anxiety_score_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)


#DATA 1, CSV 5

df = pd.read_csv(csv5_path)

total_number_of_entries = len(df)

potentially_adequate_avg = 0
other_avg = 0
untreated_avg = 0


potentially_adequate_median = 0
other_median = 0
untreated_median = 0

potentially_adequate_median_array = []
other_median_array = []
untreated_median_array = []

for index, row in df.iterrows():
    potentially_adequate = row['Potentially adequate treatment, conditional']
    other = row['Other treatments, conditional']
    untreated = row['Untreated, conditional']
    
    potentially_adequate_avg += potentially_adequate
    other_avg += other
    untreated_avg += untreated
    
    potentially_adequate_median_array.append(potentially_adequate)
    other_median_array.append(other)
    untreated_median_array.append(untreated)

potentially_adequate_median = np.median(potentially_adequate_median_array)
other_median = np.median(other_median_array)
untreated_median = np.median(untreated_median_array)
    

potentially_adequate_avg = potentially_adequate_avg/total_number_of_entries
other_avg = other_avg/total_number_of_entries
untreated_avg = untreated_avg/total_number_of_entries

potentially_adequate_std_dev = 0
other_std_dev = 0
untreated_std_dev = 0

for index, row in df.iterrows():
    potentially_adequate = row['Potentially adequate treatment, conditional']
    other = row['Other treatments, conditional']
    untreated = row['Untreated, conditional']
    
    potentially_adequate_std_dev += np.power((potentially_adequate-potentially_adequate_avg),2)
    other_std_dev += np.power((other-other_avg),2)
    untreated_std_dev += np.power((untreated-untreated_avg),2)
    
    

potentially_adequate_std_dev = np.power((potentially_adequate_std_dev/(total_number_of_entries-1)),1/2)
other_std_dev = np.power((other_std_dev/(total_number_of_entries-1)),1/2)
untreated_std_dev = np.power((untreated_std_dev/(total_number_of_entries-1)),1/2)




potentially_adequate_skew = 3*(potentially_adequate_avg-potentially_adequate_median)/potentially_adequate_std_dev
other_skew = 3*(other_avg-other_median)/other_std_dev
untreated_skew = 3*(untreated_avg-untreated_median)/untreated_std_dev


print("Potentially Adequate ", potentially_adequate_skew)
print("Other ", other_skew)
print("Unreated ", untreated_skew)

# print("schizophrenia_skew ", schizophrenia_skew)
# print("depressive_skew ", depressive_skew)
# print("anxiety_skew", anxiety_skew)
# print("bipolar_skew", bipolar_skew)
# print("eating_skew", eating_skew)
# print("Dysthymia ", dysthymia_skew)


# print("Potentially Adequate ", potentially_adequate_avg, potentially_adequate_median, potentially_adequate_std_dev)
# print("Other ", other_avg, other_median, other_std_dev)
# print("Untreated ", untreated_avg, untreated_median, untreated_std_dev)

potentially_adequate_z_score_array = []
other_z_score_array = []
untreated_z_score_array = []


for index, row in df.iterrows():
    potentially_adequate = row['Potentially adequate treatment, conditional']
    other = row['Other treatments, conditional']
    untreated = row['Untreated, conditional']
    
    potentially_adequate_z_score = (potentially_adequate - potentially_adequate_avg)/potentially_adequate_std_dev
    potentially_adequate_z_score_array.append(potentially_adequate_z_score)
    
    other_z_score = (other - other_avg)/other_std_dev
    other_z_score_array.append(other_z_score)
    
    untreated_adequate_z_score = (untreated - untreated_avg)/untreated_std_dev
    untreated_z_score_array.append(untreated_adequate_z_score)
    

potentially_adequate_outliers = []
other_outliers = []
untreated_outliers = []

for i in range(0,len(potentially_adequate_z_score_array)):
    z_score = float(potentially_adequate_z_score_array[i])
    if abs(z_score) >= 3:
        potentially_adequate_outliers.append([i,z_score])
        
for i in range(0,len(other_z_score_array)):
    z_score = float(other_z_score_array[i])
    if abs(z_score) >= 3:
        other_outliers.append([i,z_score])
        
for i in range(0,len(untreated_z_score_array)):
    z_score = float(untreated_z_score_array[i])
    if abs(z_score) >= 3:
        untreated_outliers.append([i,z_score])
        
potentially_adequate_df=pd.DataFrame(potentially_adequate_outliers, columns=['data index', 'z score']) 
potentially_adequate_df.to_csv(data_path + 'data_1_5_potentially_adequate_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

other_df=pd.DataFrame(other_outliers, columns=['data index', 'z score']) 
other_df.to_csv(data_path + 'data_1_5_other_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)

untreated_df=pd.DataFrame(untreated_outliers, columns=['data index', 'z score']) 
untreated_df.to_csv(data_path + 'data_1_5_untreated_outliers.csv', sep=',', encoding='utf-8', index=False, header=True)



#Figure creation Data_1_1
df = pd.read_csv(csv1_path)

schizophrenia_histogram_data_2019 = []
depressive_histogram_data_2019 = []
anxiety_histogram_data_2019 = []
bipolar_histogram_data_2019 = []
eating_histogram_data_2019 = []


for index, row in df.iterrows():
    year = row['Year']
    
    schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
    depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
    anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
    bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
    eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    
    if year == 2019:
        schizophrenia_histogram_data_2019.append(schizophrenia)
        depressive_histogram_data_2019.append(depressive)
        anxiety_histogram_data_2019.append(anxiety)
        bipolar_histogram_data_2019.append(bipolar)
        eating_histogram_data_2019.append(eating)





schizophrenia_over_the_years_by_country = {}
depressive_over_the_years_by_country = {}
anxiety_over_the_years_by_country = {}
bipolar_over_the_years_by_country = {}
eating_over_the_years_by_country = {}

xaxis = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

for index, row in df.iterrows():
    country = row['Entity']
    schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
    depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
    anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
    bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
    eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    
    
    
    if country not in schizophrenia_over_the_years_by_country:
        schizophrenia_over_the_years_by_country[country] = [schizophrenia]
    else:
        schizophrenia_over_the_years_by_country[country].append(schizophrenia)
        
    if country not in depressive_over_the_years_by_country:
        depressive_over_the_years_by_country[country] = [depressive]
    else:
        depressive_over_the_years_by_country[country].append(depressive)

    if country not in anxiety_over_the_years_by_country:
        anxiety_over_the_years_by_country[country] = [anxiety]
    else:
        anxiety_over_the_years_by_country[country].append(anxiety)

    if country not in bipolar_over_the_years_by_country:
        bipolar_over_the_years_by_country[country] = [bipolar]
    else:
        bipolar_over_the_years_by_country[country].append(bipolar)

    if country not in eating_over_the_years_by_country:
        eating_over_the_years_by_country[country] = [eating]
    else:
        eating_over_the_years_by_country[country].append(eating)

for country,yaxis in schizophrenia_over_the_years_by_country.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    
    # plt.scatter(xaxis, yaxis)
    
plt.rcParams['figure.dpi'] = 700
  

plt.xlabel('Year')
plt.ylabel('Percent of Population with Schizophrenia')
plt.title('Frequency of Schizophrenia across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_1_Schizophrenia_World_over_the_years.png')

plt.clf()



for country,yaxis in depressive_over_the_years_by_country.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    
plt.rcParams['figure.dpi'] = 700
  

plt.xlabel('Year')
plt.ylabel('Percent of Population that is Depressive')
plt.title('Frequency of Depressive across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_1_Depressive_World_over_the_years.png')

plt.clf()

for country,yaxis in anxiety_over_the_years_by_country.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    
plt.rcParams['figure.dpi'] = 700
  

plt.xlabel('Year')
plt.ylabel('Percent of Population that is Anxious')
plt.title('Frequency of Anxiety across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_1_Anxiety_World_over_the_years.png')

plt.clf()

for country,yaxis in bipolar_over_the_years_by_country.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    
plt.rcParams['figure.dpi'] = 700
  

plt.xlabel('Year')
plt.ylabel('Percent of Population that is Bipolar')
plt.title('Frequency of Bipolar across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_1_Bipolar_World_over_the_years.png')

plt.clf()

for country,yaxis in eating_over_the_years_by_country.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    
plt.rcParams['figure.dpi'] = 700
  

plt.xlabel('Year')
plt.ylabel('Percent of Population that has Eating Disorder')
plt.title('Frequency of Eating disorder across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_1_Eating_World_over_the_years.png')

plt.clf()


plt.rcParams['figure.dpi'] = 700

plt.hist(schizophrenia_histogram_data_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('Percent of Population with Schizophrenia')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency of Schizophrenia across the World in 2019')

plt.savefig(data_path + 'Data_1_1_Schizophrenia_Histogram.png')

plt.clf()
plt.rcParams['figure.dpi'] = 700

plt.hist(depressive_histogram_data_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('Percent of Population that is Depressive')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Depressive across the World in 2019')

plt.savefig(data_path + 'Data_1_1_Depressive_Histogram.png')

plt.clf()
plt.rcParams['figure.dpi'] = 700

plt.hist(anxiety_histogram_data_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('Percent of Population that is Anxious')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Anxiety across the World in 2019')

plt.savefig(data_path + 'Data_1_1_Anxiety_Histogram.png')

plt.clf()
plt.rcParams['figure.dpi'] = 700

plt.hist(bipolar_histogram_data_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('Percent of Population that is Bipolar')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Bipolar across the World in 2019')

plt.savefig(data_path + 'Data_1_1_Bipolar_Histogram.png')

plt.clf()

plt.rcParams['figure.dpi'] = 700
plt.hist(eating_histogram_data_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('Percent of Population that has Eating Disorder')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Eating Disorder across the World in 2019')

plt.savefig(data_path + 'Data_1_1_Eating_Histogram.png')

plt.clf()

# plt.rcParams['figure.dpi'] = 300
# print(df)



df = pd.read_csv(csv2_path)

schizophrenia_DALY_histogram_2019 = []
depressive_DALY_histogram_2019 = []
anxiety_DALY_histogram_2019 = []
bipolar_DALY_histogram_2019 = []
eating_DALY_histogram_2019 = []

schizophrenia_DALY_over_the_years = {}
depressive_DALY_over_the_years = {}
anxiety_DALY_over_the_years = {}
bipolar_DALY_over_the_years = {}
eating_DALY_over_the_years = {}

for index, row in df.iterrows():
    year = row['Year']
    country = row['Entity']
    
    schizophrenia = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia']
    depressive = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders']
    anxiety = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders']
    bipolar = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder']
    eating = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders']
    
    if country not in schizophrenia_DALY_over_the_years:
        schizophrenia_DALY_over_the_years[country] = [schizophrenia]
    else:
        schizophrenia_DALY_over_the_years[country].append(schizophrenia)
    
    if country not in depressive_DALY_over_the_years:
        depressive_DALY_over_the_years[country] = [depressive]
    else:
        depressive_DALY_over_the_years[country].append(depressive)

    if country not in anxiety_DALY_over_the_years:
        anxiety_DALY_over_the_years[country] = [anxiety]
    else:
        anxiety_DALY_over_the_years[country].append(anxiety)

    if country not in bipolar_DALY_over_the_years:
        bipolar_DALY_over_the_years[country] = [bipolar]
    else:
        bipolar_DALY_over_the_years[country].append(bipolar)

    if country not in eating_DALY_over_the_years:
        eating_DALY_over_the_years[country] = [eating]
    else:
        eating_DALY_over_the_years[country].append(eating)

    
    if year == 2019:
        schizophrenia_DALY_histogram_2019.append(schizophrenia)
        depressive_DALY_histogram_2019.append(depressive)
        anxiety_DALY_histogram_2019.append(anxiety)
        bipolar_DALY_histogram_2019.append(bipolar)
        eating_DALY_histogram_2019.append(eating)
    
#Line Graphs

xaxis = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

plt.rcParams['figure.dpi'] = 700

for country,yaxis in schizophrenia_DALY_over_the_years.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    # plt.scatter(xaxis, yaxis)
    
# plt.ylim(0,200)
plt.xlabel('Year')
plt.ylabel('DALYs for Schizophrenia')
plt.title('DALYs for Schizophrenia across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_2_Schizophrenia_World_over_the_years.png')

plt.clf()

# print(schizophrenia_DALY_over_the_years)

plt.rcParams['figure.dpi'] = 700

for country,yaxis in depressive_DALY_over_the_years.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    # plt.scatter(xaxis, yaxis)
    
# plt.ylim(0,200)
plt.xlabel('Year')
plt.ylabel('DALYs for Depressive')
plt.title('DALYs for Depressive across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_2_Depressive_World_over_the_years.png')

plt.clf()


plt.rcParams['figure.dpi'] = 700

for country,yaxis in anxiety_DALY_over_the_years.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    # plt.scatter(xaxis, yaxis)
    
# plt.ylim(0,200)
plt.xlabel('Year')
plt.ylabel('DALYs for Anxiety')
plt.title('DALYs for Anxiety across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_2_Anxiety_World_over_the_years.png')

plt.clf()

plt.rcParams['figure.dpi'] = 700

for country,yaxis in bipolar_DALY_over_the_years.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    # plt.scatter(xaxis, yaxis)
    
# plt.ylim(0,200)
plt.xlabel('Year')
plt.ylabel('DALYs for Bipolar')
plt.title('DALYs for Bipolar across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_2_Bipolar_World_over_the_years.png')

plt.clf()

plt.rcParams['figure.dpi'] = 700

for country,yaxis in eating_DALY_over_the_years.items():
    # print(country)


    plt.plot(xaxis, yaxis, label = country)
    # plt.scatter(xaxis, yaxis)
    
# plt.ylim(0,200)
plt.xlabel('Year')
plt.ylabel('DALYs for Eating Disorder')
plt.title('DALYs for Eating Disorder across the World in 1990-2019')    
    
plt.savefig(data_path + 'Data_1_2_Eating_World_over_the_years.png')

plt.clf()

#Histograms

plt.rcParams['figure.dpi'] = 700
plt.hist(schizophrenia_DALY_histogram_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('DALYs for Schizophrenia')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Schizophrenia DALYs across the World in 2019')

plt.savefig(data_path + 'Data_1_2_Schizophrenia_Histogram.png')

plt.clf() 

plt.rcParams['figure.dpi'] = 700
plt.hist(depressive_DALY_histogram_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('DALYs for Depressive')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Depressive DALYs across the World in 2019')

plt.savefig(data_path + 'Data_1_2_Depressive_Histogram.png')

plt.clf() 

plt.rcParams['figure.dpi'] = 700
plt.hist(anxiety_DALY_histogram_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('DALYs for Anxiety')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Anxiety DALYs across the World in 2019')

plt.savefig(data_path + 'Data_1_2_Anxiety_Histogram.png')

plt.clf() 
    
plt.rcParams['figure.dpi'] = 700
plt.hist(bipolar_DALY_histogram_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('DALYs for Bipolar')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Bipolar DALYs across the World in 2019')

plt.savefig(data_path + 'Data_1_2_Bipolar_Histogram.png')

plt.clf() 


plt.rcParams['figure.dpi'] = 700
plt.hist(eating_DALY_histogram_2019, bins=50, color='skyblue', edgecolor='black')

plt.xlabel('DALYs for Eating Disorder')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Eating Disorder DALYs across the World in 2019')

plt.savefig(data_path + 'Data_1_2_Eating_Histogram.png')

plt.clf() 



#Data 1_3
df = pd.read_csv(csv3_path)

depression_histogram_data_2008 = []

for index, row in df.iterrows():
    depression_score = row['Major depression']
    depression_histogram_data_2008.append(depression_score)
    
    

plt.rcParams['figure.dpi'] = 700
plt.hist(depression_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Depression Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Depression Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_3_Depression_Histogram.png')

plt.clf() 


#Data 1_4

df = pd.read_csv(csv4_path)


schizophrenia_histogram_data_2008 = []
depression_histogram_data_2008 = []
anxiety_histogram_data_2008 = []
bipolar_histogram_data_2008 = []
eating_histogram_data_2008 = []
dysthymia_histogram_data_2008 = []

for index, row in df.iterrows():
    schizophrenia = row['Schizophrenia']
    depression = row['Major depression']
    anxiety = row['Anxiety disorders']
    bipolar = row['Bipolar disorder']
    eating = row['Eating disorders']
    dysthymia = row['Dysthymia']
    
    schizophrenia_histogram_data_2008.append(schizophrenia)
    depression_histogram_data_2008.append(depression)
    anxiety_histogram_data_2008.append(anxiety)
    bipolar_histogram_data_2008.append(bipolar)
    eating_histogram_data_2008.append(eating)
    dysthymia_histogram_data_2008.append(dysthymia)
    


plt.rcParams['figure.dpi'] = 700
plt.hist(schizophrenia_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Schizophrenia Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Schizophrenia Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_4_Schizophrenia_Histogram.png')

plt.clf()

plt.rcParams['figure.dpi'] = 700
plt.hist(depression_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Depression Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Depression Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_4_Depression_Histogram.png')

plt.clf()  

plt.rcParams['figure.dpi'] = 700
plt.hist(anxiety_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Anxiety Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Anxiety Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_4_Anxiety_Histogram.png')

plt.clf()  


plt.rcParams['figure.dpi'] = 700
plt.hist(bipolar_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Bipolar Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Bipolar Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_4_Bipolar_Histogram.png')

plt.clf()  

plt.rcParams['figure.dpi'] = 700
plt.hist(eating_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Eating Disorder Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Eating Disorder Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_4_Eating_Histogram.png')

plt.clf() 

plt.rcParams['figure.dpi'] = 700
plt.hist(dysthymia_histogram_data_2008, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Dysthymia Score')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency Prevalence of Dysthymia Across World Regions in 2008')

plt.savefig(data_path + 'Data_1_4_Dysthymia_Histogram.png')

plt.clf() 


#DATA 1_5
#Years are not the same, 

df = pd.read_csv(csv5_path)


potentially_adequate_histogram = []
other_histogram = []
untreated_histogram = []

for index, row in df.iterrows():
    potentially_adequate = row['Potentially adequate treatment, conditional']
    other = row['Other treatments, conditional']
    untreated = row['Untreated, conditional']
    
    potentially_adequate_histogram.append(potentially_adequate)
    other_histogram.append(other)
    untreated_histogram.append(untreated)
    

plt.rcParams['figure.dpi'] = 700
plt.hist(potentially_adequate_histogram, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Percent of people getting Potentially Adequate Treatment')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency of people getting potentially adequate treatment (percent)')

plt.savefig(data_path + 'Data_1_5_Potentially_Adequate_Histogram.png')

plt.clf() 



plt.rcParams['figure.dpi'] = 700
plt.hist(other_histogram, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Percent of people getting Other Treatment')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency of people getting other treatment (percent)')

plt.savefig(data_path + 'Data_1_5_Other_Histogram.png')

plt.clf()

plt.rcParams['figure.dpi'] = 700
plt.hist(untreated_histogram, bins=30, color='skyblue', edgecolor='black')

plt.xlabel('Percent of people untreated')
plt.ylabel('Frequency Across World Countries')
plt.title('Frequency of people remaining untreated (percent)')

plt.savefig(data_path + 'Data_1_5_Untreated_Histogram.png')

plt.clf()


#DATA 1_6

df = pd.read_csv(csv6_path)

for index, row in df.iterrows():
    
    type_of_disease = row['Entity']
    
    
    part1 = row['Nearly every day']
    part2 = row['More than half the days']
    part3 = row['Several days']
    part4 = row['Not at all']
    
    pie_chart_entries = [part1,part2,part3,part4]
    
    labels = ['Nearly every day','More than half the days','Several days','Not at all']
    
    plt.pie(pie_chart_entries, labels=labels)
    # plt.xlabel('Percent')
    # plt.ylabel('Frequency Across World Countries')
    plt.title('Responses (Percent) for ' + type_of_disease)

    plt.savefig(data_path + 'Data_1_6_' + type_of_disease + '.png')

    plt.clf()
    
    
#DATA 1_7

df = pd.read_csv(csv7_path)


labels = []
disorder_numbers = []
for index, row in df.iterrows():
    
    number = row['Number of countries with primary data on prevalence of mental disorders']
    type_of_disorder = row['Entity']
    
    disorder_numbers.append(number)
    labels.append(type_of_disorder)


# fig, ax = plt.subplots()
# ax.bar(labels, disorder_numbers, color='skyblue')
# ax.set_title('Countries with primary data on prevalence of mental disorders in 2019')
# ax.set_xlabel('Disorder')
# ax.set_ylabel('Number of countries')
# plt.xticks(rotation=90)
# fig.subplots_adjust(bottom=0.7, top=0.8)


# plt.savefig(data_path + 'Data_1_7_Countries_Bar_Chart.png')

# plt.clf()


# DATA_2

df = pd.read_csv(csv8_path)

country_treatment_numbers = {}
gender_treatment_numbers = {}
occupation_treatment_numbers = {}

treatment_knowledge_split = {}

for index, row in df.iterrows():
    treatment = row['treatment']
    country = row['Country']
    gender = row['Gender']
    occupation = row['Occupation']
    care_options = row['care_options']
    
    if treatment == 'Yes':
        if country not in country_treatment_numbers:
            country_treatment_numbers[country] = [1,0]
        else:
            country_treatment_numbers[country][0] += 1
    else:
        if country not in country_treatment_numbers:
            country_treatment_numbers[country] = [0,1]
        else:
            country_treatment_numbers[country][1] += 1


    if treatment == 'Yes':
        if gender not in gender_treatment_numbers:
            gender_treatment_numbers[gender] = [1,0]
        else:
            gender_treatment_numbers[gender][0] += 1
    else:
        if gender not in gender_treatment_numbers:
            gender_treatment_numbers[gender] = [0,1]
        else:
            gender_treatment_numbers[gender][1] += 1 
            
            
    if treatment == 'Yes':
        if occupation not in occupation_treatment_numbers:
            occupation_treatment_numbers[occupation] = [1,0]
        else:
            occupation_treatment_numbers[occupation][0] += 1
    else:
        if occupation not in occupation_treatment_numbers:
            occupation_treatment_numbers[occupation] = [0,1]
        else:
            occupation_treatment_numbers[occupation][1] += 1 
            
    
    # dict[gender][treatment][0,1,2] 0 - yes 1 - no 2 - not sure
    
    gender_treatment_index = 0
    
    if care_options == 'Yes':
        gender_treatment_index = 0
    if care_options == 'No':
        gender_treatment_index = 1
    if care_options == 'Not sure':
        gender_treatment_index = 2
    
    if gender not in treatment_knowledge_split:
        treatment_knowledge_split[gender] = {'Yes' : [0,0,0], 'No' : [0,0,0]}
        
        treatment_knowledge_split[gender][treatment][gender_treatment_index] += 1
    else:
        treatment_knowledge_split[gender][treatment][gender_treatment_index] += 1
        
            
                   
# print(treatment_knowledge_split)


    
            

yes_array = []
no_array = []
country_labels = []


for country,array in country_treatment_numbers.items():
    yes = (country_treatment_numbers[country])[0]
    no = (country_treatment_numbers[country])[1]
    country_labels.append(country)
    yes_array.append(yes)
    no_array.append(no)
    
    
    

# x = np.arange(5) 
# y1 = [34, 56, 12, 89, 67] 
# y2 = [12, 56, 78, 45, 90] 
# y3 = [14, 23, 45, 25, 89] 
# width = 0.2
  
# # plot data in grouped manner of bar type 
# plt.bar(x-0.2, y1, width, color='green') 
# plt.bar(x, y2, width, color='red') 
# # plt.bar(x+0.2, y3, width, color='green') 
# plt.xticks(x, ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']) 
# plt.xlabel("Teams") 
# plt.ylabel("Scores") 


# plt.savefig(data_path + 'Data_2_treatment_by_country.png')

# plt.clf()

# print(country_labels)
# print(len(country_labels))
# print(len(yes_array))
# print(len(no_array))

x = np.arange(len(country_labels))

# print(len(x)) 

width = 0.2
plt.subplots_adjust(bottom=0.5)
# plot data in grouped manner of bar type 
plt.bar(x-0.2, yes_array, width, color='green') 
plt.bar(x, no_array, width, color='red') 
# plt.bar(x+0.2, y3, width, color='green') 
plt.xticks(x, country_labels) 
plt.xlabel("Countries") 
plt.ylabel("Yes (Green) No (Red)") 
plt.title('People seeking mental health treatment per country')
plt.xticks(rotation=90)


plt.savefig(data_path + 'Data_2_treatment_by_country.png')

plt.clf()

gender_labels = []
yes_array = []
no_array = []

for gender,array in gender_treatment_numbers.items():
    yes = (gender_treatment_numbers[gender])[0]
    no = (gender_treatment_numbers[gender])[1]
    gender_labels.append(gender)
    yes_array.append(yes)
    no_array.append(no)
    
    
x = np.arange(len(gender_labels))

# print(len(x)) 

width = 0.2
plt.subplots_adjust(bottom=0.3)
# plot data in grouped manner of bar type 
plt.bar(x-0.2, yes_array, width, color='green') 
plt.bar(x, no_array, width, color='red') 
# plt.bar(x+0.2, y3, width, color='green') 
plt.xticks(x, gender_labels) 
plt.xlabel("Genders") 
plt.ylabel("Yes (Green) No (Red)") 
plt.title('People seeking mental health treatment per gender, worldwide')
plt.xticks(rotation=90)


plt.savefig(data_path + 'Data_2_treatment_by_gender.png')

plt.clf()


occupation_labels = []
yes_array = []
no_array = []

for occupation,array in occupation_treatment_numbers.items():
    yes = (occupation_treatment_numbers[occupation])[0]
    no = (occupation_treatment_numbers[occupation])[1]
    occupation_labels.append(occupation)
    yes_array.append(yes)
    no_array.append(no)
    
    
x = np.arange(len(occupation_labels))

# print(len(x)) 

width = 0.2
plt.subplots_adjust(bottom=0.3)
# plot data in grouped manner of bar type 
plt.bar(x-0.2, yes_array, width, color='green') 
plt.bar(x, no_array, width, color='red') 
# plt.bar(x+0.2, y3, width, color='green') 
plt.xticks(x, occupation_labels) 
plt.xlabel("Occupations") 
plt.ylabel("Yes (Green) No (Red)") 
plt.title('People seeking mental health treatment per occupation, worldwide')
plt.xticks(rotation=90)


plt.savefig(data_path + 'Data_2_treatment_by_occupation.png')

plt.clf()


# treatment_knowledge_split
# dict[gender][treatment][0,1,2] 0 - yes 1 - no 2 - not sure


# pie_chart_entries = [part1,part2,part3,part4]

# labels = ['Nearly every day','More than half the days','Several days','Not at all']

# plt.pie(pie_chart_entries, labels=labels)
# # plt.xlabel('Percent')
# # plt.ylabel('Frequency Across World Countries')
# plt.title('Responses (Percent) for ' + type_of_disease)

# plt.savefig(data_path + 'Data_1_6_' + type_of_disease + '.png')

# plt.clf()

labels = ['F, Treatment, Knows Options', 'F, Treatment, Does not know options', 'F, Treatment, Unsure of options','F, No Treatment, Knows Options', 'F, No Treatment, Does not know options', 'F, No Treatment, Unsure of options','M, Treatment, Knows Options', 'M, Treatment, Does not know options', 'M, Treatment, Unsure of options','M, No Treatment, Knows Options', 'M, No Treatment, Does not know options', 'M, No Treatment, Unsure of options']

part1 = treatment_knowledge_split['Female']['Yes'][0]
part2 = treatment_knowledge_split['Female']['Yes'][1]
part3 = treatment_knowledge_split['Female']['Yes'][2]

part4 = treatment_knowledge_split['Female']['No'][0]
part5 = treatment_knowledge_split['Female']['No'][1]
part6 = treatment_knowledge_split['Female']['No'][2]

part7 = treatment_knowledge_split['Male']['Yes'][0]
part8 = treatment_knowledge_split['Male']['Yes'][1]
part9 = treatment_knowledge_split['Male']['Yes'][2]

part10 = treatment_knowledge_split['Male']['No'][0]
part11 = treatment_knowledge_split['Male']['No'][1]
part12 = treatment_knowledge_split['Male']['No'][2]

pie_chart_entries = [part1,part2,part3,part4,part5,part6,part7,part8,part9,part10,part11,part12]

plt.rcParams.update({'font.size': 8})
# plt.subplots_adjust(left=0.3)
plt.figure(figsize=(10, 10))
plt.pie(pie_chart_entries, labels=labels)
plt.title('Treatment Knowledge vs In Treatment vs Gender')

plt.savefig(data_path + 'Data_2_treatment_knowledge_split.png')

plt.clf()



#DATA 3

df = pd.read_csv(csv9_path)



student_treatment_breakdown = {}
student_treatment_split = {}

student_gpa_disorder_breakdown = {}

#Depression - 0, anxiety - 1, panic - 2

for index, row in df.iterrows():
    gender = row['Choose your gender']
    depression = row['Do you have Depression?']
    anxiety = row['Do you have Anxiety?']
    panic = row['Do you have Panic attack?']
    treatment = row['Did you seek any specialist for a treatment?']
    gpa = row['What is your CGPA?']
    
    
    
    #dict[gender][treament][type]
    
    if gender not in student_treatment_breakdown:
        student_treatment_breakdown[gender] = {'Yes': {'Yes' : [0,0,0], 'No' : [0,0,0]}, 'No' : {'Yes' : [0,0,0], 'No' : [0,0,0]}}
        
        student_treatment_breakdown[gender][treatment][depression][0] += 1
        student_treatment_breakdown[gender][treatment][anxiety][1] += 1
        student_treatment_breakdown[gender][treatment][panic][2] += 1
        
    else:  
        student_treatment_breakdown[gender][treatment][depression][0] += 1
        student_treatment_breakdown[gender][treatment][anxiety][1] += 1
        student_treatment_breakdown[gender][treatment][panic][2] += 1
        
        
    if gender not in student_treatment_split:
        student_treatment_split[gender] = {'Yes' : {'Yes' : 0, 'No' : 0}, 'No' : {'Yes' : 0, 'No' : 0}}
        
        
        if depression == 'Yes' or anxiety == 'Yes' or panic == 'Yes':
            student_treatment_split[gender][treatment]['Yes'] += 1
        else:
            student_treatment_split[gender][treatment]['No'] += 1
    else:
 
        if depression == 'Yes' or anxiety == 'Yes' or panic == 'Yes':
            student_treatment_split[gender][treatment]['Yes'] += 1
        else:
            student_treatment_split[gender][treatment]['No'] += 1  
            
    #small fix for bad data
    
    if gpa == '3.50 - 4.00 ':
        gpa = '3.50 - 4.00'
    #0 - depression, 1 -anxiety, 2 - panic
    if gpa not in student_gpa_disorder_breakdown:
        
        
        student_gpa_disorder_breakdown[gpa] = [0,0,0] 
        
        if depression == 'Yes':
            student_gpa_disorder_breakdown[gpa][0] += 1
        if anxiety == 'Yes':
            student_gpa_disorder_breakdown[gpa][1] += 1
        if panic == 'Yes':
            student_gpa_disorder_breakdown[gpa][2] += 1  
    else:
        if depression == 'Yes':
            student_gpa_disorder_breakdown[gpa][0] += 1
        if anxiety == 'Yes':
            student_gpa_disorder_breakdown[gpa][1] += 1
        if panic == 'Yes':
            student_gpa_disorder_breakdown[gpa][2] += 1
    
# print(student_treatment_breakdown)

labels = ['F, Treatment, Depression', 'F, Treatment, Anxiety', 'F, Treatment, Panic Attacks', 'F, No Treatment, Depression', 'F, No Treatment, Anxiety', 'F, No Treatment, Panic Attacks', 'M, Treatment, Depression', 'M, Treatment, Anxiety', 'M, Treatment, Panic Attacks', 'M, No Treatment, Depression', 'M, No Treatment, Anxiety', 'M, No Treatment, Panic Attacks']

part1 = student_treatment_breakdown['Female']['Yes']['Yes'][0]
part2 = student_treatment_breakdown['Female']['Yes']['Yes'][1]
part3 = student_treatment_breakdown['Female']['Yes']['Yes'][2]

part4 = student_treatment_breakdown['Female']['No']['Yes'][0]
part5 = student_treatment_breakdown['Female']['No']['Yes'][1]
part6 = student_treatment_breakdown['Female']['No']['Yes'][2]

part7 = student_treatment_breakdown['Male']['Yes']['Yes'][0]
part8 = student_treatment_breakdown['Male']['Yes']['Yes'][1]
part9 = student_treatment_breakdown['Male']['Yes']['Yes'][2]

part10 = student_treatment_breakdown['Male']['No']['Yes'][0]
part11 = student_treatment_breakdown['Male']['No']['Yes'][1]
part12 = student_treatment_breakdown['Male']['No']['Yes'][2]
  
  
  
pie_chart_entries = [part1,part2,part3,part4,part5,part6,part7,part8,part9,part10,part11,part12]

plt.rcParams.update({'font.size': 5})
# plt.subplots_adjust(left=0.3)
plt.figure(figsize=(10, 10))
plt.pie(pie_chart_entries, labels=labels)
plt.title('Treatment vs Disorder vs Gender')

plt.savefig(data_path + 'Data_3_student_treatment_breakdown.png')

plt.clf()  
# x = np.arange(len(occupation_labels))

# print(student_treatment_split)

labels = ['F, Treatment, Has Illness', 'F, Treatment, Unspecified Illness', 'F, No Treatment, Has Illness', 'F, No Treatment, No Specified Illness','M, Treatment, Has Illness', 'M, Treatment, Unspecified Illness', 'M, No Treatment, Has Illness', 'M, No Treatment, No Specified Illness']

part1 = student_treatment_split['Female']['Yes']['Yes']
part2 = student_treatment_split['Female']['Yes']['No']

part3 = student_treatment_split['Female']['No']['Yes']
part4 = student_treatment_split['Female']['No']['No']

part5 = student_treatment_split['Male']['Yes']['Yes']
part6 = student_treatment_split['Male']['Yes']['No']

part7 = student_treatment_split['Male']['No']['Yes']
part8 = student_treatment_split['Male']['No']['No']


pie_chart_entries = [part1,part2,part3,part4,part5,part6,part7,part8]

plt.rcParams.update({'font.size': 5})
# plt.subplots_adjust(left=0.3)
plt.figure(figsize=(10, 10))
plt.pie(pie_chart_entries, labels=labels)
plt.title('Treatment vs Gender')

plt.savefig(data_path + 'Data_3_student_treatment_split.png')

plt.clf()  


breakdown_bar_labels = ['Treatment, Depression', 'Treatment, Anxiety', 'Treatment, Panic Attacks', 'No Treatment, Depression', 'No Treatment, Anxiety', 'No Treatment, Panic Attacks']


x = np.arange(len(breakdown_bar_labels))

part1 = student_treatment_breakdown['Female']['Yes']['Yes'][0]
part2 = student_treatment_breakdown['Female']['Yes']['Yes'][1]
part3 = student_treatment_breakdown['Female']['Yes']['Yes'][2]

part4 = student_treatment_breakdown['Female']['No']['Yes'][0]
part5 = student_treatment_breakdown['Female']['No']['Yes'][1]
part6 = student_treatment_breakdown['Female']['No']['Yes'][2]

part7 = student_treatment_breakdown['Male']['Yes']['Yes'][0]
part8 = student_treatment_breakdown['Male']['Yes']['Yes'][1]
part9 = student_treatment_breakdown['Male']['Yes']['Yes'][2]

part10 = student_treatment_breakdown['Male']['No']['Yes'][0]
part11 = student_treatment_breakdown['Male']['No']['Yes'][1]
part12 = student_treatment_breakdown['Male']['No']['Yes'][2]

female_array = [part1,part2,part3,part4,part5,part6]
male_array = [part7,part8,part9,part10,part11,part12]

# # print(len(x)) 

width = 0.2
plt.subplots_adjust(bottom=0.3)
plt.rcParams.update({'font.size': 15})
# plot data in grouped manner of bar type 
plt.bar(x-0.2, female_array, width, color='pink') 
plt.bar(x, male_array, width, color='blue') 
# plt.bar(x+0.2, y3, width, color='green') 
plt.xticks(x, breakdown_bar_labels) 
plt.xlabel("Breakdown") 
plt.ylabel("Female (Pink) Male (Blue)") 
plt.title('People seeking treatment for various conditions')
plt.xticks(rotation=90)



plt.savefig(data_path + 'Data_3_student_treatment_by_gender.png')

plt.clf()


gpa_labels = ['0 - 1.99','2.00 - 2.49','2.50 - 2.99','3.00 - 3.49','3.50 - 4.00']

x = np.arange(len(gpa_labels))

# print(student_gpa_disorder_breakdown)




part1 = student_gpa_disorder_breakdown['0 - 1.99'][0]
part2 = student_gpa_disorder_breakdown['2.00 - 2.49'][0]
part3 = student_gpa_disorder_breakdown['2.50 - 2.99'][0]
part4 = student_gpa_disorder_breakdown['3.00 - 3.49'][0]
part5 = student_gpa_disorder_breakdown['3.50 - 4.00'][0]

part6 = student_gpa_disorder_breakdown['0 - 1.99'][1]
part7 = student_gpa_disorder_breakdown['2.00 - 2.49'][1]
part8 = student_gpa_disorder_breakdown['2.50 - 2.99'][1]
part9 = student_gpa_disorder_breakdown['3.00 - 3.49'][1]
part10 = student_gpa_disorder_breakdown['3.50 - 4.00'][1]

part11 = student_gpa_disorder_breakdown['0 - 1.99'][2]
part12 = student_gpa_disorder_breakdown['2.00 - 2.49'][2]
part13 = student_gpa_disorder_breakdown['2.50 - 2.99'][2]
part14 = student_gpa_disorder_breakdown['3.00 - 3.49'][2]
part15 = student_gpa_disorder_breakdown['3.50 - 4.00'][2]

depression_array = [part1,part2,part3,part4,part5]
anxiety_array = [part6,part7,part8,part9,part10]
panic_array = [part11,part12,part13,part14,part15]

# # print(len(x)) 

width = 0.2
plt.subplots_adjust(bottom=0.3)
plt.rcParams.update({'font.size': 15})
# plot data in grouped manner of bar type 
plt.bar(x-0.2, depression_array, width, color='orange') 
plt.bar(x, anxiety_array, width, color='blue') 
plt.bar(x+0.2, panic_array, width, color='green') 
plt.xticks(x, gpa_labels) 
plt.xlabel("Breakdown") 
plt.ylabel("Depression (Orange) Anxiety (Blue) Panic Attacks (Green)") 
plt.title('GPA vs mental health conditions')
plt.xticks(rotation=90)

plt.savefig(data_path + 'Data_3_student_disorders_by_gpa.png')

plt.clf()