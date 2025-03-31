from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_1_1_Features.csv'
csv2 = 'Data_1_2_Features.csv'
csv3 = 'Data_2_NumberOfSymptoms.csv'
csv4 = 'Data_3_NumberOfSymptoms.csv'
csv5 = 'MLModelAvgPercentDALY.csv'



csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv3_path = Path(data_path + csv3)
csv4_path = Path(data_path + csv4)
csv5_path = Path(data_path + csv5)




df = pd.read_csv(csv1_path)

# df.columns = ['Entity', 'Code', 'Year','DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']



predictor_columns = ['Decade','Depressive','Schizophrenia' ,'Bipolar','Eating','Anxiety']

corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Percent of Population")


plt.savefig(data_path + 'Data_1_1_feature_corr_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# print(vif_data)
# print("==========================")



df = pd.read_csv(csv2_path)

# df.columns = ['Entity', 'Code', 'Year','DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']



predictor_columns = ['Decade','Depressive','Schizophrenia' ,'Bipolar','Eating','Anxiety']

corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for DALYs")


plt.savefig(data_path + 'Data_1_2_feature_corr_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# print(vif_data)
# print("==========================")


# exit()
# df = pd.read_csv(csv3_path)

# # df.columns = ['Entity', 'Code', 'Year','DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']



# predictor_columns = ['Occupation','Country', 'self_employed', 'family_history','Days_Indoors','Growing_Stress', 'Changes_Habits', 'Mental_Health_History','Mood_Swings', 'Coping_Struggles','Work_Interest','Social_Weakness','mental_health_interview','care_options','Number of Symptoms']


# corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

# plt.rcParams.update({'font.size': 12})
# plt.figure(figsize=(10, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Correlation Matrix of Predictor Variables for ")


# plt.savefig(data_path + 'Data_2_feature_corr_heatmap.png')


# X = df[predictor_columns]
# X = add_constant(X)  # Add a constant (intercept)

# vif_data = pd.DataFrame()
# vif_data["feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# print(vif_data)


df = pd.read_csv(csv4_path)

# df.columns = ['Entity', 'Code', 'Year','DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']

# exit()

predictor_columns = ['Gender','Course', 'GPA','Married','Depression','Anxiety','Panic','Treatment','Number of Symptoms']


corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for Student Mental Health")


plt.savefig(data_path + 'Data_3_feature_corr_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# print(vif_data)
# print("==========================")



df = pd.read_csv(csv5_path)

# df.columns = ['Entity', 'Code', 'Year','DALYs, Depressive','DALYs, Schizophrenia' ,'DALYs, Bipolar','DALYs, Eating Disorder','DALYs, Anxiety']

# exit()

predictor_columns = ['Decade','Schizophrenia Avg Percent','Schizophrenia Avg DALYs','Depressive Avg Percent','Depressive Avg DALYs','Anxiety Avg Percent','Anxiety Avg DALYs','Bipolar Avg Percent','Bipolar Avg DALYs','Eating Avg Percent','Eating Avg DALYs']


corr_matrix = df[predictor_columns].corr()  # Pearson is fine; use Spearman if needed

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Predictor Variables for ")


plt.savefig(data_path + 'Data_12_feature_corr_heatmap.png')


X = df[predictor_columns]
X = add_constant(X)  # Add a constant (intercept)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
# print("==========================")








