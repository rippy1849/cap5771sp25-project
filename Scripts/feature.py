import pandas as pd
from pathlib import Path
import numpy as np



data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'data_1_1-mental-illnesses-prevalence.csv'
csv2 = 'data_1_2-burden-disease-from-each-mental-illness.csv'
csv4 = 'data_1_4-adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'
csv5 = 'data_1_5-anxiety-disorders-treatment-gap.csv'
csv6 = 'data_1_6-depressive-symptoms-across-us-population.csv'
csv7 = 'data_1_7-number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv'
csv8 = 'Data2MentalHealthConverted.csv' # not right
csv9 = 'Data3StudentMentalHealth.csv' #not right


csv10 = 'Data_1_1_AveragePercent.csv'
csv11 = 'Data_1_2_AverageDalys.csv'



csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv4_path = Path(data_path + csv4)
csv5_path = Path(data_path + csv5)
csv6_path = Path(data_path + csv6)
csv7_path = Path(data_path + csv7)
csv8_path = Path(data_path + csv8)
csv9_path = Path(data_path + csv9)

csv10_path = Path(data_path + csv10)
csv11_path = Path(data_path + csv11)






#Average Percent

df = pd.read_csv(csv1_path)

prevCountry = ""
count = 0

s_total = 0
d_total = 0
a_total = 0
b_total = 0
e_total = 0

rows = []
rows2 = []


for index,row in df.iterrows():
    country = row['Entity']
    schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
    depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
    anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
    bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
    eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    year = row['Year']
    
    # print(country)

    if prevCountry != country:
        if index != 0:
            row = [country,s_total/count,d_total/count,a_total/count,b_total/count,e_total/count]
            rows.append(row)
            
        prevCountry = country
        count = 1
        s_total = schizophrenia
        d_total = depressive
        a_total = anxiety
        b_total = bipolar
        e_total = eating
        
    else:
        count+=1
        s_total += schizophrenia
        d_total += depressive
        a_total += anxiety
        b_total += bipolar
        e_total += eating 
    
    decade = 0
    if int(year) >= 1990 and int(year) < 2000:
        decade = 1
    if int(year) >= 2000 and int(year) < 2010:
        decade = 2
    if int(year) >= 2010:
        decade = 3
            
    
    rows2.append([country,decade,year,schizophrenia,depressive,anxiety,bipolar,eating])
        
        
columns = ['Country','Schizophrenia','Depressive','Anxiety','Bipolar','Eating']
columns2 = ['Country','Decade','Year','Schizophrenia','Depressive','Anxiety','Bipolar','Eating']

df = pd.DataFrame(rows,columns=columns)
df2 = pd.DataFrame(rows2,columns=columns2)

df.to_csv('Data_1_1_AveragePercent.csv', index=False)
df2.to_csv('Data_1_1_Features.csv', index=False)



#Average DALYs

df = pd.read_csv(csv2_path)

prevCountry = ""
count = 0

s_total = 0
d_total = 0
a_total = 0
b_total = 0
e_total = 0

rows = []
rows2 = []


for index,row in df.iterrows():
    country = row['Entity']
    schizophrenia = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia']
    depressive = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders']
    anxiety = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders']
    bipolar = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder']
    eating = row['DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders']
    year = row['Year']
    
    # print(country)

    if prevCountry != country:
        if index != 0:
            row = [country,s_total/count,d_total/count,a_total/count,b_total/count,e_total/count]
            rows.append(row)
            
        prevCountry = country
        count = 1
        s_total = schizophrenia
        d_total = depressive
        a_total = anxiety
        b_total = bipolar
        e_total = eating
        
    else:
        count+=1
        s_total += schizophrenia
        d_total += depressive
        a_total += anxiety
        b_total += bipolar
        e_total += eating 
    
    decade = 0
    if int(year) >= 1990 and int(year) < 2000:
        decade = 1
    if int(year) >= 2000 and int(year) < 2010:
        decade = 2
    if int(year) >= 2010:
        decade = 3
            
    
    rows2.append([country,decade,year,schizophrenia,depressive,anxiety,bipolar,eating])
        
        
columns = ['Country','Schizophrenia','Depressive','Anxiety','Bipolar','Eating']
columns2 = ['Country','Decade','Year','Schizophrenia','Depressive','Anxiety','Bipolar','Eating']

df = pd.DataFrame(rows,columns=columns)
df2 = pd.DataFrame(rows2,columns=columns2)

df.to_csv('Data_1_2_AverageDalys.csv', index=False)
df2.to_csv('Data_1_2_Features.csv', index=False)



#Data 1_4 Average Impact score

df = pd.read_csv(csv4_path)

rows = []


for index,row in df.iterrows():
    country = row['Entity']
    schizophrenia = row['Schizophrenia']
    depressive = row['Major depression']
    anxiety = row['Anxiety disorders']
    bipolar = row['Bipolar disorder']
    eating = row['Eating disorders']
    dysthymia = row['Dysthymia']
    year = row['Year']
    
    average_score = (schizophrenia + depressive + anxiety + bipolar + eating + dysthymia)/6
    if average_score != 0:
        rows.append([country,year,average_score])
        
columns = ['Country', 'Year', 'Average Impact Score']


df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_4_AverageImpactScore.csv', index=False)


#Data 1_5 Treated vs. Untreated

df = pd.read_csv(csv5_path)

rows = []

    
for index,row in df.iterrows():
    
    country = row['Entity']
    year = row['Year']
    untreated = row['Untreated, conditional']
    
    treated = 100 - int(untreated)
    
    rows.append([country, year,treated,untreated])


columns = ['Country','Year','Treated','Untreated']

df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_1_5_TreatedUntreated.csv', index=False)
    
    
    
#Mental Health Treatment vs. Number of symptoms


df = pd.read_csv(csv8_path)

rows = []

    
for index,row in df.iterrows():
    
    p1 = row['Gender']
    p2 = row['Country']
    p3 = row['Occupation']
    p4 = row['self_employed']
    p5 = row['family_history']
    p6 = row['treatment']
    p7 = row['Days_Indoors']
    p8 = row['Growing_Stress']
    p9 = row['Changes_Habits']
    p10 = row['Mental_Health_History']
    p11 = row['Mood_Swings']
    p12 = row['Coping_Struggles']
    p13 = row['Work_Interest']
    p14 = row['Social_Weakness']
    p15 = row['mental_health_interview']
    p16 = row['care_options']
    
    
    
    
    s1 = row['Growing_Stress']
    s2 = row['Changes_Habits']
    s3 = row['Mood_Swings']
    s4 = row['Coping_Struggles']
    s5 = row['Work_Interest']
    s6 = row['Social_Weakness']
    
    
    if pd.isna(s1):
        s1 = 0
    if pd.isna(s2):
        s2 = 0
    if pd.isna(s3):
        s3 = 0
    if pd.isna(s4):
        s4 = 0
    if pd.isna(s5):
        s5 = 0
    if pd.isna(s6):
        s6 = 0
    num_symptoms = int(s1) + int(s2) + int(s3) + int(s4) + int(s5) + int(s6)
    
    rows.append([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,num_symptoms])

    
    


columns = ['Gender','Country','Occupation','self_employed','family_history','treatment','Days_Indoors','Growing_Stress','Changes_Habits','Mental_Health_History','Mood_Swings','Coping_Struggles','Work_Interest','Social_Weakness','mental_health_interview','care_options','Number of Symptoms']

df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_2_NumberOfSymptoms.csv', index=False)




#Data3
    
    
df = pd.read_csv(csv9_path)

rows = []

    
for index,row in df.iterrows():
    
    s1 = row['Depression']
    s2 = row['Anxiety']
    s3 = row['Panic']
    
    if pd.isna(s1):
        s1 = 0
    if pd.isna(s2):
        s2 = 0
    if pd.isna(s3):
        s3 = 0
    
    num_symptoms = int(s1) + int(s2) + int(s3)
    
    # rows.append([num_symptoms])
    rows.append(num_symptoms)
    
    


df['Number of Symptoms'] = rows
df = df.drop('ts', axis=1)
# columns = ['Number of Symptoms']

# df = pd.DataFrame(rows,columns=columns)

df.to_csv('Data_3_NumberOfSymptoms.csv', index=False)


#Combine average percent and DALY's
# exit()
df = pd.read_csv(csv10_path)
df2 = pd.read_csv(csv11_path)

rows = []



for index,row in df.iterrows():
    country = row['Country']
    schizophrenia = row['Schizophrenia']
    depressive = row['Depressive']
    anxiety = row['Anxiety']
    bipolar = row['Bipolar']
    eating = row['Eating']
    
    index = np.array(df2.index[df2.Country == country].values)
    index_value = str(index).strip('[]')
    
    if index_value != "":
        row2 = df2.iloc[int(index_value)]    
        # print(row2)
        schizophrenia2 = row2['Schizophrenia']
        depressive2 = row2['Depressive']
        anxiety2 = row2['Anxiety']
        bipolar2 = row2['Bipolar']
        eating2 = row2['Eating']
        
        rows.append([country,schizophrenia,schizophrenia2,depressive,depressive2,anxiety,anxiety2,bipolar,bipolar2,eating,eating2])
        
        
        
columns = ['Country', 'Schizophrenia Avg Percent','Schizophrenia Avg DALYs' ,'Depressive Avg Percent','Depressive Avg DALYs','Anxiety Avg Percent', 'Anxiety Avg DALYs','Bipolar Avg Percent', 'Bipolar Avg DALYs','Eating Avg Percent', 'Eating Avg DALYs']
df = pd.DataFrame(rows,columns=columns)

df.to_csv('MLModelAvgPercentDALY.csv', index=False)






    