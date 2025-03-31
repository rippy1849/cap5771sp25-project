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


data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


csv1 = 'Data_2_Pre_One_Hot.csv'



csv1_path = Path(data_path + csv1)

df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv1_path)


y1 = df1['Gender'].map({'Male':0, 'Female':1})
y2 = df1['treatment'].map({'No':0, 'Yes' : 1})

# print(y1)

# split_1 = int(len(y1)*.6)
lower = 6000
upper = 10000

# y1_train = df1.iloc[:lower,:]['Gender']

y1_train_female = df1.iloc[:int(lower/2),:]['Gender']
y1_train_male = df1.iloc[100000:100000+int(lower/2),:]['Gender']

y1_train = pd.concat([y1_train_female, y1_train_male], axis=0)

# print(y1_train)


y1_test_female = df1.iloc[lower:upper,:]['Gender']
y1_test_male = df1.iloc[100000+lower:100000 + upper,:]['Gender']

y1_test = pd.concat([y1_test_female, y1_test_male], axis=0)


# y1_test = df1.iloc[lower:upper,:]['Gender']

y2_train_female = df1.iloc[:int(lower/2),:]['treatment']
y2_train_male = df1.iloc[100000:100000+int(lower/2),:]['treatment']

y2_train = pd.concat([y2_train_female, y2_train_male], axis=0)


y2_test_female = df1.iloc[lower:upper,:]['treatment']
y2_test_male = df1.iloc[100000+lower:100000 + upper,:]['treatment']

y2_test = pd.concat([y2_test_female, y2_test_male], axis=0)


df1 = df1.drop(['Gender'], axis=1)
df2 = df2.drop(['treatment'], axis=1)


categorical_columns = df1.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df1[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_1 = pd.concat([df1, one_hot_df], axis=1)

df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)


# df1_train = df_encoded_1.iloc[:lower,:]
# df1_test = df_encoded_1.iloc[lower:upper,:]


df1_train_female = df_encoded_1.iloc[:int(lower/2),:]
df1_train_male = df_encoded_1.iloc[100000:100000+int(lower/2),:]

df1_train = pd.concat([df1_train_female, df1_train_male], axis=0)


df1_test_female = df_encoded_1.iloc[lower:upper,:]
df1_test_male = df_encoded_1.iloc[100000+lower:100000 + upper,:]

df1_test = pd.concat([df1_test_female, df1_test_male], axis=0)






categorical_columns = df1.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df1[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded_2 = pd.concat([df1, one_hot_df], axis=1)

df_encoded_2 = df_encoded_2.drop(categorical_columns, axis=1)


df2_train = df_encoded_2.iloc[:lower,:]
df2_test = df_encoded_2.iloc[lower:upper,:]

# print(y1_train)

df2_train_female = df_encoded_2.iloc[:int(lower/2),:]
df2_train_male = df_encoded_2.iloc[100000:100000+int(lower/2),:]

df2_train = pd.concat([df2_train_female, df2_train_male], axis=0)


df2_test_female = df_encoded_2.iloc[lower:upper,:]
df2_test_male = df_encoded_2.iloc[100000+lower:100000 + upper,:]

df2_test = pd.concat([df2_test_female, df2_test_male], axis=0)


#SVM MODEL 1

svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
# Trained the model
svm.fit(df1_train, y1_train)


y_pred = svm.predict(df1_test)

accuracy = accuracy_score(y1_test, y_pred)

y1_test = pd.DataFrame(y1_test)

# print(y_pred[0])

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

y1_test.reset_index(inplace = True)

# exit()
# print(y1_test)

y1_ROC_tpr_1 = []
y1_ROC_fpr_1 = []


for index, row in y1_test.iterrows():
    test = row['Gender']
    pred = y_pred[index]
    
    
    if (pred == 'Female') and pred == test:
        truePositive += 1
    
    if (pred == 'Female') and pred != test:
        falsePositive += 1
        
    if (pred == 'Male') and pred == test:
        trueNegative += 1

    if (pred == 'Male') and pred != test:
        falseNegative +=1
        
    if index % 100 == 0:
        y1_ROC_tpr_1.append(truePositive)
        y1_ROC_fpr_1.append(falsePositive)
        
y1_ROC_tpr_1.append(truePositive)
y1_ROC_fpr_1.append(falsePositive)

y1_ROC_tpr_1_yaxis = []
y1_ROC_fpr_1_xaxis = []

for tp,fp in zip(y1_ROC_tpr_1,y1_ROC_fpr_1):
    # print(tp,fp)
    y1_ROC_tpr_1_yaxis.append(1-(tp/truePositive))
    y1_ROC_fpr_1_xaxis.append(1-(fp/falsePositive))


# print(y1_ROC_fpr_1_xaxis)
# print(y1_ROC_tpr_1_yaxis)

# plt.scatter(y1_ROC_fpr_1_xaxis, y1_ROC_tpr_1_yaxis)
# plt.show()



# exit()


# print(y1_ROC_fpr_1)

# print(truePositive,falsePositive)

precision = truePositive/(truePositive+falsePositive)
recall = truePositive/(truePositive + falseNegative)
f1score = 2*precision*recall/(precision+recall)



# precision = precision_score(y1_test,y_pred,pos_label="Female")
# f1score = f1_score(y1_test,y_pred,pos_label="Female")
# recallscore = recall_score(y1_test,y_pred,pos_label="Female")

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'f1score: {f1score}')
print(f'recallscore: {recall}')
print("==============")



#Second SVM


svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
# Trained the model
svm.fit(df2_train, y2_train)

# print(svm.predict(df2_test))

y_pred = svm.predict(df2_test)


y2_test = pd.DataFrame(y2_test)

y2_test.reset_index(inplace = True)


truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

y2_ROC_tpr_1 = []
y2_ROC_fpr_1 = []

for index, row in y2_test.iterrows():
    test = row['treatment']
    pred = y_pred[index]
    
    
    if (pred == 'Yes') and pred == test:
        truePositive += 1
    
    if (pred == 'Yes') and pred != test:
        falsePositive += 1
        
    if (pred == 'No') and pred == test:
        trueNegative += 1

    if (pred == 'No') and pred != test:
        falseNegative +=1
        
    if index % 100 == 0:
        y2_ROC_tpr_1.append(truePositive)
        y2_ROC_fpr_1.append(falsePositive)
        
y2_ROC_tpr_1.append(truePositive)
y2_ROC_fpr_1.append(falsePositive)

y2_ROC_tpr_1_yaxis = []
y2_ROC_fpr_1_xaxis = []

for tp,fp in zip(y2_ROC_tpr_1,y2_ROC_fpr_1):
    # print(tp,fp)
    y2_ROC_tpr_1_yaxis.append(1-(tp/truePositive))
    y2_ROC_fpr_1_xaxis.append(1-(fp/falsePositive))
        
# print(truePositive,falsePositive)

# print(y2_ROC_tpr_1_yaxis)

# plt.scatter(y2_ROC_fpr_1_xaxis, y2_ROC_tpr_1_yaxis)
# plt.show()


precision = truePositive/(truePositive+falsePositive)
recall = truePositive/(truePositive + falseNegative)
f1score = 2*precision*recall/(precision+recall)
accuracy = (truePositive + trueNegative)/(trueNegative+truePositive+falseNegative+falsePositive)

# accuracy = accuracy_score(y2_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'f1score: {f1score}')
print(f'recallscore: {recall}')
print("==============")


# exit()



clf = DecisionTreeClassifier(random_state=1)
clf.fit(df1_train, y1_train)

y_pred = clf.predict(df1_test)



y1_test = pd.DataFrame(y1_test)

# print(y_pred[0])

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

y1_test.reset_index(inplace = True)

# exit()
# print(y1_test)

y1_ROC_tpr_2 = []
y1_ROC_fpr_2 = []


for index, row in y1_test.iterrows():
    test = row['Gender']
    pred = y_pred[index]
    
    
    if (pred == 'Female') and pred == test:
        truePositive += 1
    
    if (pred == 'Female') and pred != test:
        falsePositive += 1
        
    if (pred == 'Male') and pred == test:
        trueNegative += 1

    if (pred == 'Male') and pred != test:
        falseNegative +=1
        
    if index % 100 == 0:
        y1_ROC_tpr_2.append(truePositive)
        y1_ROC_fpr_2.append(falsePositive)
        
y1_ROC_tpr_2.append(truePositive)
y1_ROC_fpr_2.append(falsePositive)

y1_ROC_tpr_2_yaxis = []
y1_ROC_fpr_2_xaxis = []

for tp,fp in zip(y1_ROC_tpr_2,y1_ROC_fpr_2):
    # print(tp,fp)
    y1_ROC_tpr_2_yaxis.append(1-(tp/truePositive))
    y1_ROC_fpr_2_xaxis.append(1-(fp/falsePositive))


precision = truePositive/(truePositive+falsePositive)
recall = truePositive/(truePositive + falseNegative)
f1score = 2*precision*recall/(precision+recall)
accuracy = (truePositive + trueNegative)/(trueNegative+truePositive+falseNegative+falsePositive)

# accuracy = accuracy_score(y2_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'f1score: {f1score}')
print(f'recallscore: {recall}')
print("==============")


# plt.scatter(y1_ROC_fpr_2_xaxis, y1_ROC_tpr_2_yaxis)
# plt.show()

# exit()


#Train2

# clf = DecisionTreeClassifier(random_state=1)
# clf.fit(df2_train, y2_train)

# y_pred = clf.predict(df2_test)




clf = DecisionTreeClassifier(random_state=1)
clf.fit(df2_train, y2_train)

y_pred = clf.predict(df2_test)



y2_test = pd.DataFrame(y2_test)

# print(y_pred[0])

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

y2_test.reset_index(inplace = True)

# exit()
# print(y1_test)

y2_ROC_tpr_2 = []
y2_ROC_fpr_2 = []


for index, row in y2_test.iterrows():
    test = row['treatment']
    pred = y_pred[index]
    
    
    if (pred == 'Yes') and pred == test:
        truePositive += 1
    
    if (pred == 'Yes') and pred != test:
        falsePositive += 1
        
    if (pred == 'No') and pred == test:
        trueNegative += 1

    if (pred == 'No') and pred != test:
        falseNegative +=1
        
    if index % 100 == 0:
        y2_ROC_tpr_2.append(truePositive)
        y2_ROC_fpr_2.append(falsePositive)
        
y2_ROC_tpr_2.append(truePositive)
y2_ROC_fpr_2.append(falsePositive)

y2_ROC_tpr_2_yaxis = []
y2_ROC_fpr_2_xaxis = []

# print(y2_ROC_tpr_2)

# exit()

for tp,fp in zip(y2_ROC_tpr_2,y2_ROC_fpr_2):
    # print(tp,fp)
    y2_ROC_tpr_2_yaxis.append(1-(tp/truePositive))
    if fp != 0:
        y2_ROC_fpr_2_xaxis.append(1-(fp/falsePositive))
    else:
        y2_ROC_fpr_2_xaxis.append(1)


precision = truePositive/(truePositive+falsePositive)
recall = truePositive/(truePositive + falseNegative)
f1score = 2*precision*recall/(precision+recall)
accuracy = (truePositive + trueNegative)/(trueNegative+truePositive+falseNegative+falsePositive)

# accuracy = accuracy_score(y2_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'f1score: {f1score}')
print(f'recallscore: {recall}')
print("==============")


# print(y_pred)

# accuracy = accuracy_score(y2_test, y_pred)

# print(f'Accuracy: {accuracy}')



rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(df1_train, y1_train)

# # Make predictions
y_pred = rf_classifier.predict(df1_test)



y1_test = pd.DataFrame(y1_test)

# exit()

# print(y_pred[0])

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0


y1_ROC_tpr_3 = []
y1_ROC_fpr_3 = []


for index, row in y1_test.iterrows():
    test = row['Gender']
    pred = y_pred[index]
    
    
    if (pred == 'Female') and pred == test:
        truePositive += 1
    
    if (pred == 'Female') and pred != test:
        falsePositive += 1
        
    if (pred == 'Male') and pred == test:
        trueNegative += 1

    if (pred == 'Male') and pred != test:
        falseNegative +=1
        
    if index % 100 == 0:
        y1_ROC_tpr_3.append(truePositive)
        y1_ROC_fpr_3.append(falsePositive)
        
y1_ROC_tpr_3.append(truePositive)
y1_ROC_fpr_3.append(falsePositive)

# exit()

y1_ROC_tpr_3_yaxis = []
y1_ROC_fpr_3_xaxis = []

for tp,fp in zip(y1_ROC_tpr_3,y1_ROC_fpr_3):
    # print(tp,fp)
    y1_ROC_tpr_3_yaxis.append(1-(tp/truePositive))
    y1_ROC_fpr_3_xaxis.append(1-(fp/falsePositive))


precision = truePositive/(truePositive+falsePositive)
recall = truePositive/(truePositive + falseNegative)
f1score = 2*precision*recall/(precision+recall)
accuracy = (truePositive + trueNegative)/(trueNegative+truePositive+falseNegative+falsePositive)

# accuracy = accuracy_score(y2_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'f1score: {f1score}')
print(f'recallscore: {recall}')
print("==============")



rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(df2_train, y2_train)

# # Make predictions
y_pred = rf_classifier.predict(df2_test)



y2_test = pd.DataFrame(y2_test)

# exit()

# print(y_pred[0])

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0


y2_ROC_tpr_3 = []
y2_ROC_fpr_3 = []


for index, row in y2_test.iterrows():
    test = row['treatment']
    pred = y_pred[index]
    
    
    if (pred == 'Yes') and pred == test:
        truePositive += 1
    
    if (pred == 'Yes') and pred != test:
        falsePositive += 1
        
    if (pred == 'No') and pred == test:
        trueNegative += 1

    if (pred == 'No') and pred != test:
        falseNegative +=1
        
    if index % 100 == 0:
        y2_ROC_tpr_3.append(truePositive)
        y2_ROC_fpr_3.append(falsePositive)
        
y2_ROC_tpr_3.append(truePositive)
y2_ROC_fpr_3.append(falsePositive)

# exit()

y2_ROC_tpr_3_yaxis = []
y2_ROC_fpr_3_xaxis = []

for tp,fp in zip(y2_ROC_tpr_3,y2_ROC_fpr_3):
    # print(tp,fp)
    y2_ROC_tpr_3_yaxis.append(1-(tp/truePositive))
    if fp != 0:
        y2_ROC_fpr_3_xaxis.append(1-(fp/falsePositive))
    else:
        y2_ROC_fpr_3_xaxis.append(1)


precision = truePositive/(truePositive+falsePositive)
recall = truePositive/(truePositive + falseNegative)
f1score = 2*precision*recall/(precision+recall)
accuracy = (truePositive + trueNegative)/(trueNegative+truePositive+falseNegative+falsePositive)

# accuracy = accuracy_score(y2_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'f1score: {f1score}')
print(f'recallscore: {recall}')
print("==============")




plt.plot(y1_ROC_tpr_1_yaxis, y1_ROC_fpr_1_xaxis, label='SVM Model', color='red')
plt.plot(y1_ROC_tpr_2_yaxis, y1_ROC_fpr_2_xaxis, label='Decision Tree', color='green')
plt.plot(y1_ROC_tpr_3_yaxis, y1_ROC_fpr_3_xaxis, label='Random Forest', color='blue')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Gender')
plt.legend()

# plt.show()
plt.savefig('Data_2_ROC_curve_y1.png')

plt.clf()

plt.plot(y2_ROC_tpr_1_yaxis, y2_ROC_fpr_1_xaxis, label='SVM Model', color='red')
# plt.plot(y2_ROC_tpr_2_yaxis, y2_ROC_fpr_2_xaxis, label='Decision Tree', color='green')
# plt.plot(y2_ROC_tpr_3_yaxis, y2_ROC_fpr_3_xaxis, label='Random Forest', color='blue')

plt.scatter([0], [1], label='Decision Tree', color='green')
plt.scatter([0], [1], label='Random Forest', color='blue')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Treatment')
plt.legend()

# plt.show()

plt.savefig('Data_2_ROC_curve_y2.png')