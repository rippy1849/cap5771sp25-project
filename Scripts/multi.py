import matplotlib
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.utils.fixes import parse_version
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path


data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'

csv1 = 'data_1_1-mental-illnesses-prevalence.csv'
csv2 = 'data_1_2-burden-disease-from-each-mental-illness.csv'
csv4 = 'data_1_4-adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'
csv5 = 'data_1_5-anxiety-disorders-treatment-gap.csv'
csv6 = 'data_1_6-depressive-symptoms-across-us-population.csv'
csv7 = 'data_1_7-number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv'
csv8 = 'data_2_Mental Health Dataset.csv'
csv9 = 'data_3_Student Mental health.csv'

csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv4_path = Path(data_path + csv4)
csv5_path = Path(data_path + csv5)
csv6_path = Path(data_path + csv6)
csv7_path = Path(data_path + csv7)
csv8_path = Path(data_path + csv8)
csv9_path = Path(data_path + csv9)



# def plot_permutation_importance(clf, X, y, ax):
#     result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
#     perm_sorted_idx = result.importances_mean.argsort()

#     # `labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
#     # renamed to `tick_labels`. The following code handles this, but as a
#     # scikit-learn user you probably can write simpler code by using `labels=...`
#     # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).
#     tick_labels_parameter_name = (
#         "tick_labels"
#         if parse_version(matplotlib.__version__) >= parse_version("3.9")
#         else "labels"
#     )
#     tick_labels_dict = {tick_labels_parameter_name: X.columns[perm_sorted_idx]}
#     ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
#     ax.axvline(x=0, color="k", linestyle="--")
#     return ax



# df = data.drop('B', axis=1)
# dfy = df['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']

# dfx = df.drop(['Entity','Code','Year','Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized'], axis=1)

# y_array = []
# X_array = []

# for index, row in df.iterrows():
#     schizophrenia = row['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized']
#     depressive = row['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized']
#     anxiety = row['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized']
#     bipolar = row['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized']
#     eating = row['Eating disorders (share of population) - Sex: Both - Age: Age-standardized']
    
#     x = [depressive,anxiety,bipolar,eating]
#     X_array.append(x)
#     y_array.append(schizophrenia)
    
# dfx = pd.DataFrame(X_array, columns=['Depressive','Anxiety','Bipolar','Eating'])
# dfy = pd.DataFrame(y_array, columns=['Schizophrenia'])


# X, y = load_breast_cancer(return_X_y=True, as_frame=True)
# print(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
# print(f"Baseline accuracy on test data: {clf.score(X_test, y_test):.2}")



df = pd.read_csv(csv1_path)

df.columns = ['Entity','Code','Year','Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar','Eating']

# print(df)

# exit()
# Assuming 'df' is your DataFrame
# 'predictor_columns' are the columns you want to check for multicollinearity
predictor_columns = ['Schizophrenia', 'Depressive', 'Anxiety','Bipolar','Eating']
corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_1_1")


plt.savefig(data_path + 'Data_1_1_multi_heatmap.png')

# plt.show()



# Assuming 'df' is your DataFrame, 'predictor_columns' lists your predictor variables.
X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# print(vif_data)


df = pd.read_csv(csv2_path)

df.columns = ['Entity', 'Code', 'Year','DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']



predictor_columns = ['DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']

corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_1_2")


plt.savefig(data_path + 'Data_1_2_multi_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# print(vif_data)



#DATA 1_4
df = pd.read_csv(csv4_path)


predictor_columns = ['Major depression', 'Bipolar disorder', 'Eating disorders', 'Dysthymia', 'Schizophrenia','Anxiety disorders']


corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_1_4")


plt.savefig(data_path + 'Data_1_4_multi_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# print(vif_data)

#DATA 1_5

df = pd.read_csv(csv5_path)

df.columns = ['Entity', 'Code', 'Year', 'Adequate', 'Other', 'Untreated']
predictor_columns = ['Adequate', 'Other', 'Untreated']


corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_1_5")


plt.savefig(data_path + 'Data_1_5_multi_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# print(vif_data)

#DATA 1_6

df = pd.read_csv(csv6_path)

predictor_columns = ['Nearly every day', 'More than half the days', 'Several days', 'Not at all']


corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_1_6")


plt.savefig(data_path + 'Data_1_6_multi_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# print(vif_data)


#DATA 1_7

# df = pd.read_csv(csv7_path)

# predictor_columns = ['Number of countries with primary data on prevalence of mental disorders']


# corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

# plt.rcParams.update({'font.size': 12})
# plt.figure(figsize=(10, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Correlation Matrix of Predictor Variables for Data_1_7")


# plt.savefig(data_path + 'Data_1_7_multi_heatmap.png')


# X = df[predictor_columns]
# X = add_constant(X)  # Add a constant (intercept)

# vif_data = pd.DataFrame()
# vif_data["feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]


#DATA 2
df = pd.read_csv(csv8_path)

country_map = {}
country_num = 0
for index, row in df.iterrows():
    country = row['Country']
    
    if country not in country_map:
        country_map[country] = country_num
        country_num += 1
    

#Convert Data to numerical
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
df['Country'] = df['Country'].map(country_map)
df['Occupation'] = df['Occupation'].map({'Corporate' : 0, 'Student':1, 'Business':2, 'Housewife' : 3, 'Others': 4})
df['self_employed'] = df['self_employed'].map({'No':0, 'Yes':1, np.nan : 2})
df['family_history'] = df['family_history'].map({'No':0, 'Yes':1})
df['Days_Indoors'] = df['Days_Indoors'].map({'Go out Every day' : 0, '1-14 days':1, '15-30 days':2, '31-60 days' : 3, 'More than 2 months': 4})
df['Growing_Stress'] = df['Growing_Stress'].map({'No':0, 'Yes':1})
df['Changes_Habits'] = df['Changes_Habits'].map({'No':0, 'Yes':1})
df['Mental_Health_History'] = df['Mental_Health_History'].map({'No':0, 'Yes':1})
df['Mood_Swings'] = df['Mood_Swings'].map({'Low':0, 'Medium':1, 'High' : 2})
df['Coping_Struggles'] = df['Coping_Struggles'].map({'No':0, 'Yes':1})
df['Work_Interest'] = df['Work_Interest'].map({'No':0, 'Yes':1})
df['Social_Weakness'] = df['Social_Weakness'].map({'No':0, 'Yes':1})
df['mental_health_interview'] = df['mental_health_interview'].map({'No':0, 'Yes':1})
df['care_options'] = df['care_options'].map({'No':0, 'Yes':1, 'Not sure' : 2})


predictor_columns = ['Occupation','Country', 'self_employed', 'family_history','Days_Indoors','Growing_Stress', 'Changes_Habits', 'Mental_Health_History','Mood_Swings', 'Coping_Struggles','Work_Interest','Social_Weakness','mental_health_interview','care_options']
corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_2")


plt.savefig(data_path + 'Data_2_multi_heatmap.png')

# plt.show()
df = df.dropna()

# the independent variables set
X = df[predictor_columns]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

# print(vif_data)



df = pd.read_csv(csv9_path)


df.columns = ['ts', 'Gender', 'Age','Course','Study Year', 'GPA','Married','Depression','Anxiety','Panic','Treatment']

course_categories = {}
course_number = 0
study_year_categories = {}
study_number = 0
gpa_categories = {'0 - 1.99' : 0,'2.00 - 2.49' :1,'2.50 - 2.99':2,'3.00 - 3.49':3,'3.50 - 4.00':4}




# df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
for index, row in df.iterrows():
    course = row['Course']
    study = row['Study Year']
    
    if course not in course_categories:
        course_categories[course] = course_number
        course_number += 1   
    if study.lower() not in study_year_categories:
        study_year_categories[study.lower()] = study_number
        study_number += 1
    
# 'ts', 'Gender', 'Age','Course','Study Year', 'GPA','Married','Depression','Anxiety','Panic','Treatment'
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
df['Course'] = df['Course'].map(course_categories)
df['Study Year'] = df['Study Year'].map(study_year_categories)
df['GPA'] = df['GPA'].map(gpa_categories)
df['Anxiety'] = df['Anxiety'].map({'No':0, 'Yes':1})
df['Married'] = df['Married'].map({'No':0, 'Yes':1})
df['Depression'] = df['Depression'].map({'No':0, 'Yes':1})
df['Panic'] = df['Panic'].map({'No':0, 'Yes':1})
df['Treatment'] = df['Treatment'].map({'No':0, 'Yes':1})


predictor_columns = ['Gender', 'Age','Course','Study Year', 'GPA','Married','Depression','Anxiety','Panic','Treatment']
corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Data_3")


plt.savefig(data_path + 'Data_3_multi_heatmap.png')

# plt.show()
df = df.dropna()

# the independent variables set
X = df[predictor_columns]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

# print(vif_data)

# print(df)