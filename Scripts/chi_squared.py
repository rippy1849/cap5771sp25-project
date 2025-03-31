import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_selection import chi2


from sklearn.preprocessing import OneHotEncoder

data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_2_Pre_One_Hot.csv'
csv2 = 'Data_3_Pre_One_Hot.csv'


csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)



df = pd.read_csv(csv1_path)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df, one_hot_df], axis=1)

df_encoded = df_encoded.drop(categorical_columns, axis=1)

# print(df_encoded)

y = df_encoded['Number of Symptoms'].to_numpy()


# print(y)
# print(np.array(y.T.values))/


df_encoded = df_encoded.drop('Number of Symptoms', axis=1)

X = df_encoded.to_numpy()

# print(X)

chi2_stats, p_values = chi2(X, y)

np.savetxt("Data_2_chi2_stats.csv", chi2_stats, delimiter=",")
np.savetxt("Data_2_p_values.csv", p_values, delimiter=",")


df = pd.read_csv(csv2_path)


categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df, one_hot_df], axis=1)

df_encoded = df_encoded.drop(categorical_columns, axis=1)

# print(df_encoded)

df_encoded = df_encoded.drop('Age', axis=1)

y = df_encoded['Number of Symptoms'].to_numpy()


# # print(y)
# # print(np.array(y.T.values))/



df_encoded = df_encoded.drop('Number of Symptoms', axis=1)

X = df_encoded.to_numpy()

# # print(X)

chi2_stats, p_values = chi2(X, y)


np.savetxt("Data_3_chi2_stats.csv", chi2_stats, delimiter=",")
np.savetxt("Data_3_p_values.csv", p_values, delimiter=",")

# print(p_values)
# print(chi2_stats)





