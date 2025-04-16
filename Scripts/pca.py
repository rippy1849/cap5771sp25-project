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
from sklearn.decomposition import PCA



data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_2_Pre_One_Hot.csv'
csv2 = 'Data_3_Pre_One_Hot.csv'

csv3 = 'Data_1_1_Features.csv'
csv4 = 'Data_1_2_Features.csv'




csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv3_path = Path(data_path + csv3)
csv4_path = Path(data_path + csv4)


df3 = pd.read_csv(csv2_path)
df4 = pd.read_csv(csv2_path)

df5 = pd.read_csv(csv3_path)
df6 = pd.read_csv(csv4_path)




df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv1_path)


categorical_columns = df1.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df1[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_1 = pd.concat([df1, one_hot_df], axis=1)

df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)


# print(df_encoded_1)
pca = PCA(n_components=2)

X_train = pca.fit_transform(df_encoded_1)



x = []
y = []
for x_comp in X_train:
    x.append(x_comp[0])
    y.append(x_comp[1])



plt.scatter(x,y)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.title('PCA for World Mental Health')

plt.savefig('pca_data2.png')

plt.clf()


df3 = df3.drop('Study Year',axis=1)
df3 = df3.drop(['Age'], axis=1)
# print(df3)


categorical_columns = df3.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df3[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_1 = pd.concat([df3, one_hot_df], axis=1)

df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)


# print(df_encoded_1)
pca = PCA(n_components=2)

X_train = pca.fit_transform(df_encoded_1)



x = []
y = []
for x_comp in X_train:
    x.append(x_comp[0])
    y.append(x_comp[1])



plt.scatter(x,y)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.title('PCA for Student Mental Health')
# plt.show()
plt.savefig('pca_data3.png')

# plt.show()


print(df5)

df5 = df5.drop('Country', axis=1)
df5 = df5.drop('Decade', axis=1)
df5 = df5.drop('Year', axis=1)

df6 = df6.drop('Country', axis=1)
df6 = df6.drop('Decade', axis=1)
df6 = df6.drop('Year', axis=1)



pca = PCA(n_components=2)

X_train = pca.fit_transform(df5)



x = []
y = []
for x_comp in X_train:
    x.append(x_comp[0])
    y.append(x_comp[1])



plt.scatter(x,y)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.title('PCA for Worldwide Percent Mental Illness')
# plt.show()
plt.savefig('pca_data1_1.png')



pca = PCA(n_components=2)

X_train = pca.fit_transform(df6)



x = []
y = []
for x_comp in X_train:
    x.append(x_comp[0])
    y.append(x_comp[1])



plt.scatter(x,y)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.title('PCA for Worldwide DALYs Mental Illness')
# plt.show()
plt.savefig('pca_data1_2.png')