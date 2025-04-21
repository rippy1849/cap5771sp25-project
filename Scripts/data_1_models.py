import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib

data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_1_1_Features.csv'
csv2 = 'Data_1_2_Features.csv'
csv3 = 'MLModelAvgPercentDALY.csv'


csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv3_path = Path(data_path + csv3)


df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
df3 = pd.read_csv(csv3_path)

lower = int(0.6*len(df1))



categorical_columns = ['Decade']

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df1[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_1 = pd.concat([df1, one_hot_df], axis=1)

df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]

x_train = X1.iloc[:lower,:]
x_test = X1.iloc[lower:,:]


y1_train = df1.iloc[:lower,:]['Schizophrenia']
y1_test = df1.iloc[lower:,:]['Schizophrenia']

y2_train = df1.iloc[:lower,:]['Depressive']
y2_test = df1.iloc[lower:,:]['Depressive']

y3_train = df1.iloc[:lower,:]['Anxiety']
y3_test = df1.iloc[lower:,:]['Anxiety']

y4_train = df1.iloc[:lower,:]['Bipolar']
y4_test = df1.iloc[lower:,:]['Bipolar']

y5_train = df1.iloc[:lower,:]['Eating']
y5_test = df1.iloc[lower:,:]['Eating']


# print(len(x_test),len(y1_test))

regr = LinearRegression() 
  
regr.fit(x_train, y1_train)

joblib.dump(regr,'LRM1SP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y1_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y1_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")


regr = LinearRegression() 
  
regr.fit(x_train, y2_train)

joblib.dump(regr,'LRM1DP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y2_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y2_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y3_train)

joblib.dump(regr,'LRM1AP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y3_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y3_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y4_train)

joblib.dump(regr,'LRM1BP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y4_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y4_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y5_train)

joblib.dump(regr,'LRM1EP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y5_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y5_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")


# Dataset 1_2_feaures



lower = int(0.6*len(df2))



categorical_columns = ['Decade']

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df2[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_1 = pd.concat([df2, one_hot_df], axis=1)

df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]

x_train = X1.iloc[:lower,:]
x_test = X1.iloc[lower:,:]


y1_train = df2.iloc[:lower,:]['Schizophrenia']
y1_test = df2.iloc[lower:,:]['Schizophrenia']

y2_train = df2.iloc[:lower,:]['Depressive']
y2_test = df2.iloc[lower:,:]['Depressive']

y3_train = df2.iloc[:lower,:]['Anxiety']
y3_test = df2.iloc[lower:,:]['Anxiety']

y4_train = df2.iloc[:lower,:]['Bipolar']
y4_test = df2.iloc[lower:,:]['Bipolar']

y5_train = df2.iloc[:lower,:]['Eating']
y5_test = df2.iloc[lower:,:]['Eating']


# print(len(x_test),len(y1_test))

regr = LinearRegression() 
  
regr.fit(x_train, y1_train)

joblib.dump(regr,'LRM1SD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y1_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y1_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")


regr = LinearRegression() 
  
regr.fit(x_train, y2_train)

joblib.dump(regr,'LRM1DD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y2_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y2_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y3_train)

joblib.dump(regr,'LRM1AD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y3_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y3_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y4_train)

joblib.dump(regr,'LRM1BD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y4_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y4_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y5_train)

joblib.dump(regr,'LRM1ED.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y5_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y5_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")



#Dataset 1 Averages
lower = int(0.6*len(df3))


categorical_columns = ['Decade']

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df3[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_1 = pd.concat([df3, one_hot_df], axis=1)

df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]

x_train = X1.iloc[:lower,:]
x_test = X1.iloc[lower:,:]



y1_train = df3.iloc[:lower,:]['Schizophrenia Avg DALYs']
y1_test = df3.iloc[lower:,:]['Schizophrenia Avg DALYs']

y2_train = df3.iloc[:lower,:]['Depressive Avg DALYs']
y2_test = df3.iloc[lower:,:]['Depressive Avg DALYs']

y3_train = df3.iloc[:lower,:]['Anxiety Avg DALYs']
y3_test = df3.iloc[lower:,:]['Anxiety Avg DALYs']

y4_train = df3.iloc[:lower,:]['Bipolar Avg DALYs']
y4_test = df3.iloc[lower:,:]['Bipolar Avg DALYs']

y5_train = df3.iloc[:lower,:]['Eating Avg DALYs']
y5_test = df3.iloc[lower:,:]['Eating Avg DALYs']


# print(x1_train,y1_train)

# exit()

# print()

# exit()

# exit()

regr = LinearRegression() 
  
regr.fit(x_train, y1_train)

joblib.dump(regr,'LRM1ASD.pkl')

y_pred = regr.predict(x_test)

# exit()

# exit()
# print(y_pred)

mae = mean_absolute_error(y_true=y1_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y1_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

# exit()


regr = LinearRegression() 
  
regr.fit(x_train, y2_train)

joblib.dump(regr,'LRM1ADD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y2_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y2_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y3_train)

joblib.dump(regr,'LRM1AAD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y3_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y3_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y4_train)


joblib.dump(regr,'LRM1ABD.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y4_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y4_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y5_train)

joblib.dump(regr,'LRM1AED.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y5_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y5_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")






y1_train = df3.iloc[:lower,:]['Schizophrenia Avg Percent']
y1_test = df3.iloc[lower:,:]['Schizophrenia Avg Percent']

y2_train = df3.iloc[:lower,:]['Depressive Avg Percent']
y2_test = df3.iloc[lower:,:]['Depressive Avg Percent']

y3_train = df3.iloc[:lower,:]['Anxiety Avg Percent']
y3_test = df3.iloc[lower:,:]['Anxiety Avg Percent']

y4_train = df3.iloc[:lower,:]['Bipolar Avg Percent']
y4_test = df3.iloc[lower:,:]['Bipolar Avg Percent']

y5_train = df3.iloc[:lower,:]['Eating Avg Percent']
y5_test = df3.iloc[lower:,:]['Eating Avg Percent']


# print(x1_train,y1_train)

# exit()

# print()

# exit()

# exit()

regr = LinearRegression() 
  
regr.fit(x_train, y1_train)
joblib.dump(regr,'LRM1ASP.pkl')

y_pred = regr.predict(x_test)

# exit()

# exit()
# print(y_pred)

mae = mean_absolute_error(y_true=y1_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y1_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

# exit()


regr = LinearRegression() 
  
regr.fit(x_train, y2_train)

joblib.dump(regr,'LRM1ADP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y2_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y2_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y3_train)

joblib.dump(regr,'LRM1AAP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y3_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y3_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y4_train)

joblib.dump(regr,'LRM1ABP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y4_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y4_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")

regr = LinearRegression() 
  
regr.fit(x_train, y5_train)

joblib.dump(regr,'LRM1AEP.pkl')

y_pred = regr.predict(x_test)

# print(y_pred)

mae = mean_absolute_error(y_true=y5_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y5_test,y_pred=y_pred)

print(f"Mean Absolute Error {mae}")
print(f"Mean Squared Error {mse}")
print("========================")


