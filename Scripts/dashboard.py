import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

from urllib.error import URLError
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'
model_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\Models\\'


csv1 = 'Data_1_1_Schizophrenia.csv'
csv2 = 'Data_1_1_Depression.csv'
csv3 = 'Data_1_1_Bipolar.csv'
csv4 = 'Data_1_1_Eating.csv'
csv5 = 'Data_1_1_Anxiety.csv'

csv6 = 'Data_1_2_Schizophrenia.csv'
csv7 = 'Data_1_2_Depression.csv'
csv8 = 'Data_1_2_Bipolar.csv'
csv9 = 'Data_1_2_Eating.csv'
csv10 = 'Data_1_2_Anxiety.csv'

csv11 = 'data_1_4-adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'
csv12 = 'data_1_5-anxiety-disorders-treatment-gap.csv'
csv13 = 'data_1_6-depressive-symptoms-across-us-population.csv'
csv14 = 'Data_2_Pre_One_Hot.csv'
csv15 = 'Data_3_Pre_One_Hot.csv'

csv16 = 'Data_1_1_Features.csv'
csv17 = 'Data_1_2_Features.csv'
csv18 = 'MLModelAvgPercentDALY.csv'



model1 = 'DT_Gender_Data_2.pkl'
model2 = 'DT_Treatment_Data_2.pkl'
model3 = 'RF_Gender_Data_2.pkl'
model4 = 'RF_Treatment_Data_2.pkl'
model5 = 'SVM_Gender_Data_2.pkl'
model6 = 'SVM_Treatment_Data_2.pkl'

model7 = 'DT_Depression_Data_3.pkl'
model8 = 'DT_Gender_Data_3.pkl'
model9 = 'RF_Depression_Data_3.pkl'
model10 = 'RF_Gender_Data_3.pkl'
model11 = 'SVM_Depression_Data_3.pkl'
model12 = 'SVM_Gender_Data_3.pkl'

model13 = 'LRM1SP.pkl'
model14 = 'LRM1DP.pkl'
model15 = 'LRM1AP.pkl'
model16 = 'LRM1BP.pkl'
model17 = 'LRM1EP.pkl'

model18 = 'LRM1SD.pkl'
model19 = 'LRM1DD.pkl'
model20 = 'LRM1AD.pkl'
model21 = 'LRM1BD.pkl'
model22 = 'LRM1ED.pkl'

model23 = 'LRM1ASP.pkl'
model24 = 'LRM1ADP.pkl'
model25 = 'LRM1AAP.pkl'
model26 = 'LRM1ABP.pkl'
model27 = 'LRM1AEP.pkl'

model28 = 'LRM1ASD.pkl'
model29 = 'LRM1ADD.pkl'
model30 = 'LRM1AAD.pkl'
model31 = 'LRM1ABD.pkl'
model32 = 'LRM1AED.pkl'





csv1_path = Path(data_path + csv1)
csv2_path = Path(data_path + csv2)
csv3_path = Path(data_path + csv3)
csv4_path = Path(data_path + csv4)
csv5_path = Path(data_path + csv5)

csv6_path = Path(data_path + csv6)
csv7_path = Path(data_path + csv7)
csv8_path = Path(data_path + csv8)
csv9_path = Path(data_path + csv9)
csv10_path = Path(data_path + csv10)
csv11_path = Path(data_path + csv11)
csv12_path = Path(data_path + csv12)
csv13_path = Path(data_path + csv13)
csv14_path = Path(data_path + csv14)
csv15_path = Path(data_path + csv15)

csv16_path = Path(data_path + csv16)
csv17_path = Path(data_path + csv17)
csv18_path = Path(data_path + csv18)




model1_path = Path(model_path + model1)
model2_path = Path(model_path + model2)
model3_path = Path(model_path + model3)
model4_path = Path(model_path + model4)
model5_path = Path(model_path + model5)
model6_path = Path(model_path + model6)

model7_path = Path(model_path + model7)
model8_path = Path(model_path + model8)
model9_path = Path(model_path + model9)
model10_path = Path(model_path + model10)
model11_path = Path(model_path + model11)
model12_path = Path(model_path + model12)

model13_path = Path(model_path + model13)
model14_path = Path(model_path + model14)
model15_path = Path(model_path + model15)
model16_path = Path(model_path + model16)
model17_path = Path(model_path + model17)

model18_path = Path(model_path + model18)
model19_path = Path(model_path + model19)
model20_path = Path(model_path + model20)
model21_path = Path(model_path + model21)
model22_path = Path(model_path + model22)

model23_path = Path(model_path + model23)
model24_path = Path(model_path + model24)
model25_path = Path(model_path + model25)
model26_path = Path(model_path + model26)
model27_path = Path(model_path + model27)

model28_path = Path(model_path + model28)
model29_path = Path(model_path + model29)
model30_path = Path(model_path + model30)
model31_path = Path(model_path + model31)
model32_path = Path(model_path + model32)




df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
df3 = pd.read_csv(csv3_path)
df4 = pd.read_csv(csv4_path)
df5 = pd.read_csv(csv5_path)
df6 = pd.read_csv(csv6_path)
df7 = pd.read_csv(csv7_path)
df8 = pd.read_csv(csv8_path)
df9 = pd.read_csv(csv9_path)
df10 = pd.read_csv(csv10_path)
df11 = pd.read_csv(csv11_path)
df12 = pd.read_csv(csv12_path)
df13 = pd.read_csv(csv13_path)
df14 = pd.read_csv(csv14_path)
df15 = pd.read_csv(csv15_path)

df16 = pd.read_csv(csv16_path)
df17 = pd.read_csv(csv17_path)
df18 = pd.read_csv(csv18_path)



data_2_DT_Gender = joblib.load(model1_path)
data_2_DT_Treatment = joblib.load(model2_path)
data_2_RF_Gender = joblib.load(model3_path)
data_2_RF_Treatment = joblib.load(model4_path)
data_2_SVM_Gender = joblib.load(model5_path)
data_2_SVM_Treatment = joblib.load(model6_path)

data_3_DT_Depression = joblib.load(model7_path)
data_3_DT_Gender = joblib.load(model8_path)
data_3_RF_Depression = joblib.load(model9_path)
data_3_RF_Gender = joblib.load(model10_path)
data_3_SVM_Depression = joblib.load(model11_path)
data_3_SVM_Gender = joblib.load(model12_path)

data_1_LR_Schizophrenia_Percent = joblib.load(model13_path)
data_1_LR_Depression_Percent = joblib.load(model14_path)
data_1_LR_Anxiety_Percent = joblib.load(model15_path)
data_1_LR_Bipolar_Percent = joblib.load(model16_path)
data_1_LR_Eating_Percent = joblib.load(model17_path)

data_1_LR_Schizophrenia_DALY = joblib.load(model18_path)
data_1_LR_Depression_DALY = joblib.load(model19_path)
data_1_LR_Anxiety_DALY = joblib.load(model20_path)
data_1_LR_Bipolar_DALY = joblib.load(model21_path)
data_1_LR_Eating_DALY = joblib.load(model22_path)

data_1_LR_Average_Schizophrenia_Percent = joblib.load(model23_path)
data_1_LR_Average_Depression_Percent = joblib.load(model24_path)
data_1_LR_Average_Anxiety_Percent = joblib.load(model25_path)
data_1_LR_Average_Bipolar_Percent = joblib.load(model26_path)
data_1_LR_Average_Eating_Percent = joblib.load(model27_path)

data_1_LR_Average_Schizophrenia_DALY = joblib.load(model28_path)
data_1_LR_Average_Depression_DALY = joblib.load(model29_path)
data_1_LR_Average_Anxiety_DALY = joblib.load(model30_path)
data_1_LR_Average_Bipolar_DALY = joblib.load(model31_path)
data_1_LR_Average_Eating_DALY = joblib.load(model32_path)




df14 = df14.drop('self_employed', axis=1)




def one_hot(df):
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    df_encoded_2 = pd.concat([df, one_hot_df], axis=1)

    df_encoded_2 = df_encoded_2.drop(categorical_columns, axis=1)
    
    return df_encoded_2

def intro():
    import streamlit as st

    st.write("# Welcome to my Mental Health Data Dashboard Tool! 👋")
    st.write("By Andrew Rippy")
    st.sidebar.success("Select a page.")

    st.markdown(
        """
        Mental health is important topic to many people, and is just as important as physical health, yet many disregard it. BUT NOT TODAY!

        **👈 Select a page from the dropdown on the left** to see some interactive data with mental health,
        utilizing Streamlit!

        ### What is there to see?

        - Interactive Mental Health Data Visualizations
        - Interactive Machine Learning Models such as, Linear Regression, Random Forest, Decision Tree, SVM
        

        ### User Defined Data Points
        
        - Generate your own data from machine learning models
        - See how your data compares with real data!


    """
    )


def pca_dataset_1_avg():


    st.markdown(f"# {list(page_names_to_funcs.keys())[9]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1 with PCA.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health
"""

    )
    
    df_avg_daly = df18
    df_avg_percent = df18
        
    
    df_avg_daly = df_avg_daly.drop(['Schizophrenia Avg Percent','Depressive Avg Percent','Anxiety Avg Percent','Bipolar Avg Percent','Eating Avg Percent','Country'],axis=1)
    df_avg_percent = df_avg_percent.drop(['Schizophrenia Avg DALYs','Depressive Avg DALYs','Anxiety Avg DALYs','Bipolar Avg DALYs','Eating Avg DALYs','Country'],axis=1)


    decade_DALY = df_avg_daly['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    decade_percent = df_avg_percent['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})


    
    # st.write("### World Mental Health Stats")
    type_list = ['Average Percent','Average DALY']
    indicator_list = ['No Label', 'Decade']


    stat_types = st.multiselect(
        "Choose an Stat", type_list, ['Average Percent'], key="stats"
    )
    if not stat_types:
        st.error("Please select at least one type of Label")
    else:

        # df13.rename(columns={'Entity': 'Symptom'}, inplace=True)
        indicators = st.multiselect(
            "Choose an Label", indicator_list, ['No Label'], key="lables_percent"
        )
        if not indicators:
            st.error("Please select at least one type of Label")
        else:
            # data = df13.loc[df13['Symptom'].isin(symptoms)]
            
            st.write("### World Average Mental Health Diseases")
        
        
        for stat in stat_types:
            if stat == 'Average Percent':
                for indi in indicators:
                    if indi == 'No Label':
                        df = df_avg_percent
                        
                        df = df.drop(['Decade'],axis=1)
                    if indi == 'Decade':
                        df = df_avg_percent
                        df = df.drop(['Decade'],axis=1)
                        color_indicator = decade_percent
                    
        
                    pca = PCA(n_components=2)

                    X_train = pca.fit_transform(df)

                    x = []
                    y = []
                    for x_comp in X_train:
                        x.append(x_comp[0])
                        y.append(x_comp[1])
                    
                    
                    
                    data = {'Principle Component 1': x,
                            'Principle Component 2': y}
                
                    df_scatter = pd.DataFrame(data)
                    
                    if indi != 'No Label':
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {indi} as Label for Average World Percent")
                
                        st.plotly_chart(fig)
                    else:
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2', title=f"PCA with No Label for Average World Percent")
                    
                        st.plotly_chart(fig)
            if stat == 'Average DALY':
                for indi in indicators:
                    if indi == 'No Label':
                        df = df_avg_daly
                        df = df.drop(['Decade'],axis=1)
                    if indi == 'Decade':
                        df = df_avg_daly
                        df = df.drop(['Decade'],axis=1)
                        color_indicator = decade_DALY
                    
                    
        
                    pca = PCA(n_components=2)

                    X_train = pca.fit_transform(df)

                    x = []
                    y = []
                    for x_comp in X_train:
                        x.append(x_comp[0])
                        y.append(x_comp[1])
                    
                    
                    
                    data = {'Principle Component 1': x,
                            'Principle Component 2': y}
                
                    df_scatter = pd.DataFrame(data)
                    
                    if indi != 'No Label':
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {indi} as Label for Average World Percent")
                
                        st.plotly_chart(fig)
                    else:
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2', title=f"PCA with No Label for Average World DALY")
                    
                        st.plotly_chart(fig)
    st.write("### Work with models")
    
    
    t1 = st.selectbox(
    "What is the input decade?",
    (df_avg_percent['Decade'].unique()),
    )
    
    #PERCENT
    
    y_decade_data_1_1 = df_avg_percent['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    y_decade_data_1_1.iloc[len(y_decade_data_1_1)-1] = 'Test Point'
    
    # data_1_1['Decade'].iloc[len(data_1_1)-1] = t1
    df_avg_percent.at[len(df_avg_percent)-1, 'Decade'] = t1
    
    categorical_columns = ['Decade']

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(df_avg_percent[categorical_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    df_encoded_1 = pd.concat([df_avg_percent, one_hot_df], axis=1)

    df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

    X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]
    
    test_input_percent = pd.DataFrame((X1.iloc[len(df_avg_percent)-1])).transpose()
    
    # print(test_point_percent)
    
    
    p1 = data_1_LR_Average_Schizophrenia_Percent.predict(test_input_percent)[0]
    p2 = data_1_LR_Average_Depression_Percent.predict(test_input_percent)[0]
    p3 = data_1_LR_Average_Anxiety_Percent.predict(test_input_percent)[0]
    p4 = data_1_LR_Average_Bipolar_Percent.predict(test_input_percent)[0]
    p5 = data_1_LR_Average_Eating_Percent.predict(test_input_percent)[0]
    
    percent_arr = [p1,p2,p3,p4,p5]
    
    
    df = df_avg_percent
    df = df.drop(['Decade'],axis=1)
    df.iloc[len(df)-1] = percent_arr
    
    
    
    pca = PCA(n_components=2)

    X_train = pca.fit_transform(df)

    x = []
    y = []
    for x_comp in X_train:
        x.append(x_comp[0])
        y.append(x_comp[1])
    
    
    
    data = {'Principle Component 1': x,
            'Principle Component 2': y}

    df_scatter = pd.DataFrame(data)
    
    fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=y_decade_data_1_1, title=f"PCA with Decade as Label for Average World Percent")

    st.plotly_chart(fig)
    
    
    y_decade_data_1_2 = df_avg_daly['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    y_decade_data_1_2.iloc[len(y_decade_data_1_2)-1] = 'Test Point'
    
    # data_1_1['Decade'].iloc[len(data_1_1)-1] = t1
    df_avg_daly.at[len(df_avg_daly)-1, 'Decade'] = t1
    
    categorical_columns = ['Decade']

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(df_avg_daly[categorical_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    df_encoded_1 = pd.concat([df_avg_daly, one_hot_df], axis=1)

    df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

    X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]
    
    test_input_daly = pd.DataFrame((X1.iloc[len(df_avg_daly)-1])).transpose()
    
    # print(test_point_percent)
    


    
    p1 = data_1_LR_Average_Schizophrenia_DALY.predict(test_input_daly)[0]
    p2 = data_1_LR_Average_Depression_DALY.predict(test_input_daly)[0]
    p3 = data_1_LR_Average_Anxiety_DALY.predict(test_input_daly)[0]
    p4 = data_1_LR_Average_Bipolar_DALY.predict(test_input_daly)[0]
    p5 = data_1_LR_Average_Eating_DALY.predict(test_input_daly)[0]
    
    percent_arr = [p1,p2,p3,p4,p5]
    
    
    df = df_avg_daly
    df = df.drop(['Decade'],axis=1)
    df.iloc[len(df)-1] = percent_arr
    
    
    
    pca = PCA(n_components=2)

    X_train = pca.fit_transform(df)

    x = []
    y = []
    for x_comp in X_train:
        x.append(x_comp[0])
        y.append(x_comp[1])
    
    
    
    data = {'Principle Component 1': x,
            'Principle Component 2': y}

    df_scatter = pd.DataFrame(data)
    
    fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=y_decade_data_1_2, title=f"PCA with Decade as Label for Average World DALY")

    st.plotly_chart(fig)
    
    
    
    

def pca_dataset_1():


    st.markdown(f"# {list(page_names_to_funcs.keys())[8]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1 with PCA.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health
"""

    )
    
    df_avg_daly = df18
    df_avg_percent = df18
    
    data_1_1 = df16
    data_1_2 = df17
    
    decade_percent = data_1_1['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    year_percent = data_1_1['Year']
    
    decade_DALY = data_1_2['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    year_DALY = data_1_2['Year']
    
    
    
    df_avg_daly = df_avg_daly.drop(['Schizophrenia Avg Percent','Depressive Avg Percent','Anxiety Avg Percent','Bipolar Avg Percent','Eating Avg Percent','Country'],axis=1)
    df_avg_percent = df_avg_percent.drop(['Schizophrenia Avg DALYs','Depressive Avg DALYs','Anxiety Avg DALYs','Bipolar Avg DALYs','Eating Avg DALYs','Country'],axis=1)

    
    # st.write("### World Mental Health Stats")
    type_list = ['Percent','DALY']
    indicator_list = ['No Label', 'Decade','Year']


    stat_types = st.multiselect(
        "Choose an Stat", type_list, ['Percent'], key="stats"
    )
    if not stat_types:
        st.error("Please select at least one type of Label")
    else:

        # df13.rename(columns={'Entity': 'Symptom'}, inplace=True)
        indicators = st.multiselect(
            "Choose an Label", indicator_list, ['No Label'], key="lables_percent"
        )
        if not indicators:
            st.error("Please select at least one type of Label")
        else:
            # data = df13.loc[df13['Symptom'].isin(symptoms)]
            
            st.write("### World Mental Health Diseases")
        
        
        for stat in stat_types:
            if stat == 'Percent':
                for indi in indicators:
                    if indi == 'No Label':
                        df = data_1_1
                        df = df.drop(['Country','Decade','Year'],axis=1)
                    if indi == 'Decade':
                        df = data_1_1
                        df = df.drop(['Country','Decade','Year'],axis=1)
                        color_indicator = decade_percent
                    if indi == 'Year':
                        # df = df15.drop('Depression',axis=1)
                        df = data_1_1
                        df = df.drop(['Country','Decade','Year'],axis=1)
                        color_indicator = year_percent
                    
                    
                    # print(df)
                    # one_hot_new = one_hot(df)
        
                    pca = PCA(n_components=2)

                    X_train = pca.fit_transform(df)

                    x = []
                    y = []
                    for x_comp in X_train:
                        x.append(x_comp[0])
                        y.append(x_comp[1])
                    
                    
                    
                    data = {'Principle Component 1': x,
                            'Principle Component 2': y}
                
                    df_scatter = pd.DataFrame(data)
                    
                    if indi != 'No Label':
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {indi} as Label for World Percent")
                
                        st.plotly_chart(fig)
                    else:
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2', title=f"PCA with No Label for World Percent")
                    
                        st.plotly_chart(fig)
            if stat == 'DALY':
                for indi in indicators:
                    if indi == 'No Label':
                        df = data_1_2
                        df = df.drop(['Country','Decade','Year'],axis=1)
                    if indi == 'Decade':
                        df = data_1_2
                        df = df.drop(['Country','Decade','Year'],axis=1)
                        color_indicator = decade_DALY
                    if indi == 'Year':
                        # df = df15.drop('Depression',axis=1)
                        df = data_1_2
                        df = df.drop(['Country','Decade','Year'],axis=1)
                        color_indicator = year_DALY
                    
                    
                    # print(df)
                    # one_hot_new = one_hot(df)
        
                    pca = PCA(n_components=2)

                    X_train = pca.fit_transform(df)

                    x = []
                    y = []
                    for x_comp in X_train:
                        x.append(x_comp[0])
                        y.append(x_comp[1])
                    
                    
                    
                    data = {'Principle Component 1': x,
                            'Principle Component 2': y}
                
                    df_scatter = pd.DataFrame(data)
                    
                    if indi != 'No Label':
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {indi} as Label for World Percent")
                
                        st.plotly_chart(fig)
                    else:
                        fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2', title=f"PCA with No Label for World DALY")
                    
                        st.plotly_chart(fig)
                        
    st.write("### Work with models")
    
    
    t1 = st.selectbox(
    "What is the input decade?",
    (data_1_1['Decade'].unique()),
    )
    
    #PERCENT
    
    y_decade_data_1_1 = data_1_1['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    y_decade_data_1_1.iloc[len(y_decade_data_1_1)-1] = 'Test Point'
    
    # data_1_1['Decade'].iloc[len(data_1_1)-1] = t1
    data_1_1.at[len(data_1_1)-1, 'Decade'] = t1
    
    categorical_columns = ['Decade']

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(data_1_1[categorical_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    df_encoded_1 = pd.concat([data_1_1, one_hot_df], axis=1)

    df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

    X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]
    
    test_input_percent = pd.DataFrame((X1.iloc[len(data_1_1)-1])).transpose()
    
    # print(test_point_percent)
    

    
    p1 = data_1_LR_Schizophrenia_Percent.predict(test_input_percent)[0]
    p2 = data_1_LR_Depression_Percent.predict(test_input_percent)[0]
    p3 = data_1_LR_Anxiety_Percent.predict(test_input_percent)[0]
    p4 = data_1_LR_Bipolar_Percent.predict(test_input_percent)[0]
    p5 = data_1_LR_Eating_Percent.predict(test_input_percent)[0]
    
    percent_arr = [p1,p2,p3,p4,p5]
    
    
    df = data_1_1
    df = df.drop(['Country','Decade','Year'],axis=1)
    df.iloc[len(df)-1] = percent_arr
    
    
    
    pca = PCA(n_components=2)

    X_train = pca.fit_transform(df)

    x = []
    y = []
    for x_comp in X_train:
        x.append(x_comp[0])
        y.append(x_comp[1])
    
    
    
    data = {'Principle Component 1': x,
            'Principle Component 2': y}

    df_scatter = pd.DataFrame(data)
    
    fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=y_decade_data_1_1, title=f"PCA with Decade as Label for World Percent")

    st.plotly_chart(fig)
    
    
    
    #DALY
    
    y_decade_data_1_2 = data_1_2['Decade'].map({1:'1990-1999', 2:'2000-2009',3:'2010-2019'})
    y_decade_data_1_2.iloc[len(y_decade_data_1_2)-1] = 'Test Point'
    
    # data_1_1['Decade'].iloc[len(data_1_1)-1] = t1
    data_1_2.at[len(data_1_2)-1, 'Decade'] = t1
    
    categorical_columns = ['Decade']

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(data_1_2[categorical_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    df_encoded_1 = pd.concat([data_1_2, one_hot_df], axis=1)

    df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

    X1 = df_encoded_1[['Decade_1','Decade_2','Decade_3']]
    
    test_input_daly = pd.DataFrame((X1.iloc[len(data_1_2)-1])).transpose()
    
    # print(test_point_percent)
    

    
    p1 = data_1_LR_Schizophrenia_DALY.predict(test_input_daly)[0]
    p2 = data_1_LR_Depression_DALY.predict(test_input_daly)[0]
    p3 = data_1_LR_Anxiety_DALY.predict(test_input_daly)[0]
    p4 = data_1_LR_Bipolar_DALY.predict(test_input_daly)[0]
    p5 = data_1_LR_Eating_DALY.predict(test_input_daly)[0]
    
    percent_arr = [p1,p2,p3,p4,p5]
    
    
    df = data_1_2
    df = df.drop(['Country','Decade','Year'],axis=1)
    df.iloc[len(df)-1] = percent_arr
    
    
    
    pca = PCA(n_components=2)

    X_train = pca.fit_transform(df)

    x = []
    y = []
    for x_comp in X_train:
        x.append(x_comp[0])
        y.append(x_comp[1])
    
    
    
    data = {'Principle Component 1': x,
            'Principle Component 2': y}

    df_scatter = pd.DataFrame(data)
    
    fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=y_decade_data_1_2, title=f"PCA with Decade as Label for World DALY")

    st.plotly_chart(fig)

    
    
    
                      
                        
                        
            
 
 
def pca_dataset_3():


    st.markdown(f"# {list(page_names_to_funcs.keys())[7]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 3 with PCA.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/shariful07/student-mental-health/data
"""

    )
    df15 = pd.read_csv(csv15_path)
    
    # y1 = df1['Gender'].map({'Male':0, 'Female':1})
    # y2 = df1['treatment'].map({'No':0, 'Yes' : 1})
    
    y1 = df15['Depression']
    y2 = df15['Gender'] 
    
    # df15 = df15.drop('Study Year',axis=1)
    # df15 = df15.drop(['Age'], axis=1)
    df15 = df15.drop(['Study Year'], axis=1)
    
    # df15['Age'] = df15['Age'].map({ np.nan :18})

 
    indicator_list = ['No Label', 'Gender','Depression']

    # df13.rename(columns={'Entity': 'Symptom'}, inplace=True)
    indicators = st.multiselect(
        "Choose an Label", indicator_list, ['No Label'], key="illness1"
    )
    if not indicators:
        st.error("Please select at least one type of Label")
    else:
        # data = df13.loc[df13['Symptom'].isin(symptoms)]
        
        st.write("### Student Mental Health")
    
    
    
        for indi in indicators:
            if indi == 'No Label':
                df = df15
            if indi == 'Gender':
                df = df15.drop('Gender',axis=1)
                
                color_indicator = y2
                # color_indicator.iloc[len(color_indicator)-1] = 'Test Point'
                
            if indi == 'Depression':
                df = df15.drop('Depression',axis=1)
                color_indicator = y1
            
            
            one_hot_new = one_hot(df)
 
            pca = PCA(n_components=2)

            X_train = pca.fit_transform(one_hot_new)

            x = []
            y = []
            for x_comp in X_train:
                x.append(x_comp[0])
                y.append(x_comp[1])
            
            
            
            data = {'Principle Component 1': x,
                    'Principle Component 2': y}
        
            df_scatter = pd.DataFrame(data)
            
            if indi != 'No Label':
                fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {indi} as Label")
           
                st.plotly_chart(fig)
            else:
                fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2', title=f"PCA with No Label")
            
                st.plotly_chart(fig)
                
        st.markdown(f"# Work with Trained Models")
            
        st.write(
            """
            Use a user-defined test data point and work with 3 different models with two separate outputs to predict and see where they lie on a PCA graph.
            """
        )      
    
    
    
    t1 = st.selectbox(
    "What is your gender?",
    (df15['Gender'].unique()),
    )  
    
    
    t2 = st.selectbox(
    "What is your Age?",
    (df15['Age'].unique()),
    )
    
    t3 = st.selectbox(
    "What Major are you in?",
    (df15['Course'].unique()),
    )
    
    t4 = st.selectbox(
    "What is your GPA?",
    (df15['GPA'].unique()),
    )
    
    t5 = st.selectbox(
    "Are you Married?",
    (df15['Married'].unique()),
    )
    
    t6 = st.selectbox(
    "Do you have Depression?",
    (df15['Depression'].unique()),
    )
    
    t7 = st.selectbox(
    "Do you have Anxiety",
    (df15['Anxiety'].unique()),
    )
    
    t8 = st.selectbox(
    "Do you Panic?",
    (df15['Panic'].unique()),
    )
    
    t9 = st.selectbox(
    "Are you in treatment?",
    (df15['Treatment'].unique()),
    )
    
    t10 = 0
    
    if t6 == 'Yes':
        t10 += 1
    if t7 == 'Yes':
        t10 += 1
    if t8 == 'Yes':
        t10 += 1
    
    
    test_point_arr = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10]
   
    df15.iloc[len(df15)-1] = test_point_arr
    # y1.iloc[len(df14)-1] = 'Test Point'
    # y2.iloc[len(df14)-1] = 'Test Point'
    
    
    
    model_list = ['SVM', 'Decision Tree','Random Forest']
    
    chosen_models = st.multiselect(
        "Choose a Model", model_list, ['SVM'], key="models"
    )
    if not indicators:
        st.error("Please select at least one Model")
    else:
        output_list = ['Gender', 'Depression']
    
        outputs = st.multiselect(
            "Choose an Output", output_list, ['Gender'], key="outputs"
        )
        if not indicators:
            st.error("Please select at least one Output")
        else:
            # df = df14.drop('treatment',axis=1)
            # out = one_hot(df)
            # print(out)
            # color_indicator = y2
                
            
            
            for output in outputs:
                
                if output == 'Gender':
                    df = df15.drop('Gender',axis=1)
                
                
                if output == 'Depression':
                    df = df15.drop('Depression',axis=1)
                    
            
                
                one_hot_new = one_hot(df)
                # print(one_hot_new)
                test_point = one_hot_new.iloc[len(df)-1]
                # y1.iloc[len(df14)-1] = 'Test Point'
                # y2.iloc[len(df14)-1] = 'Test Point'
                # print()
                y_gender = pd.DataFrame(y2)['Gender']
                y_gender.iloc[len(df15)-1] = 'Test Point'
                
                y_depression = pd.DataFrame(y1)['Depression']
                y_depression.iloc[len(df15)-1] = 'Test Point'
                 
                
                # print(y1)
                
                if output == 'Gender':
                    color_indicator = y_gender
                if output == 'Depression':
                    color_indicator = y_depression
                
                pca = PCA(n_components=2)

                X_train = pca.fit_transform(one_hot_new)

                x = []
                y = []
                for x_comp in X_train:
                    x.append(x_comp[0])
                    y.append(x_comp[1])
        
                
                
                data = {'Principle Component 1': x,
                        'Principle Component 2': y}
            
                df_scatter = pd.DataFrame(data)
                
                # if output != 'No Label':
                fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {output} as Label")
            
                st.plotly_chart(fig)  
                
                
                test_input = pd.DataFrame(test_point.T)
                test_input = test_input.transpose()

                if output == 'Gender':
                    for m in chosen_models:
                        if m == 'SVM':
                            y_pred = data_3_SVM_Gender.predict(test_input)
                            st.write(f"SVM predicts you are {y_pred}")
                            # print(y_pred)
                        if m == 'Decision Tree':
                            y_pred = data_3_DT_Gender.predict(test_input)
                            st.write(f"Decision Tree predicts you are {y_pred}")
                        if m == 'Random Forest':
                            y_pred = data_3_RF_Gender.predict(test_input)
                            st.write(f"Random Forest predicts you are {y_pred}")
                            
                if output == 'Depression':
                    for m in chosen_models:
                        if m == 'SVM':
                            y_pred = data_3_SVM_Depression.predict(test_input)
                            st.write(f"SVM predicts you said {y_pred} to Depression")
                            # print(y_pred)
                        if m == 'Decision Tree':
                            y_pred = data_3_DT_Depression.predict(test_input)
                            st.write(f"Decision Tree predicts you said {y_pred} to Depression")
                        if m == 'Random Forest':
                            y_pred = data_3_RF_Depression.predict(test_input)
                            st.write(f"Random Forest predicts you said {y_pred} to Depression")              
        
    
def pca_dataset_2():


    st.markdown(f"# {list(page_names_to_funcs.keys())[6]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 2 with PCA.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset
"""

    )
    # y1 = df1['Gender'].map({'Male':0, 'Female':1})
    # y2 = df1['treatment'].map({'No':0, 'Yes' : 1})
    
    y1 = df14['Gender']
    y2 = df14['treatment']
    
    


    # illness_scores = ['Major depression','Bipolar disorder','Eating disorders','Dysthymia','Schizophrenia','Anxiety disorders']
    indicator_list = ['No Label', 'Gender','Treatment']

    # df13.rename(columns={'Entity': 'Symptom'}, inplace=True)
    indicators = st.multiselect(
        "Choose an Label", indicator_list, ['No Label'], key="illness1"
    )
    if not indicators:
        st.error("Please select at least one type of Label")
    else:
        # data = df13.loc[df13['Symptom'].isin(symptoms)]
        
        st.write("### Worldwide Mental Health")
    
    
    
        for indi in indicators:
            if indi == 'No Label':
                df = df14
            if indi == 'Gender':
                df = df14.drop('Gender',axis=1)
                
                color_indicator = y1
                # color_indicator.iloc[len(color_indicator)-1] = 'Test Point'
                
            if indi == 'Treatment':
                df = df14.drop('treatment',axis=1)
                color_indicator = y2
                
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            encoder = OneHotEncoder(sparse_output=False)

            one_hot_encoded = encoder.fit_transform(df[categorical_columns])

            one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

            df_encoded_1 = pd.concat([df, one_hot_df], axis=1)

            df_encoded_1 = df_encoded_1.drop(categorical_columns, axis=1)

            # print(df_encoded_1)

            pca = PCA(n_components=2)

            X_train = pca.fit_transform(df_encoded_1)

            x = []
            y = []
            for x_comp in X_train:
                x.append(x_comp[0])
                y.append(x_comp[1])
            # x.append(x[0])
            # y.append(y[0])
            # print(len(x),len(y),len(color_indicator))
            
            
            data = {'Principle Component 1': x,
                    'Principle Component 2': y}
        
            df_scatter = pd.DataFrame(data)
            
            if indi != 'No Label':
                fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {indi} as Label")
           
                st.plotly_chart(fig)
            else:
                fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2', title=f"PCA with No Label")
            
                st.plotly_chart(fig)
    st.markdown(f"# Work with Trained Models")
            
    st.write(
        """
        Use a user-defined test data point and work with 3 different models with two separate outputs to predict and see where they lie on a PCA graph.
        """
    )      
    
    
    
    t1 = st.selectbox(
    "What is your Gender",
    (df14['Gender'].unique()),
    )
    
    t2 = st.selectbox(
    "What Country are you from?",
    (df14['Country'].unique()),
    )
    
    t3 = st.selectbox(
    "What is your Occupation?",
    (df14['Occupation'].unique()),
    )
    
    t4 = st.selectbox(
    "Do you have family history with Mental Health?",
    (df14['family_history'].unique()),
    )
    
    t5 = st.selectbox(
    "Are you in treatment?",
    (df14['treatment'].unique()),
    )
    
    t6 = st.selectbox(
    "How many days indoors do you spend?",
    (df14['Days_Indoors'].unique()),
    )
    
    t7 = st.selectbox(
    "Do you have growing Stress?",
    (df14['Growing_Stress'].unique()),
    )
    
    t8 = st.selectbox(
    "Do you have Changes in Habits?",
    (df14['Changes_Habits'].unique()),
    )
    t9 = st.selectbox(
    "Do you have mental health history?",
    (df14['Mental_Health_History'].unique()),
    )
    
    t10 = st.selectbox(
    "Do you have mood swings?",
    (df14['Mood_Swings'].unique()),
    )
    
    t11 = st.selectbox(
    "Do you have coping struggles?",
    (df14['Coping_Struggles'].unique()),
    )
    
    t12 = st.selectbox(
    "Are you interested in work?",
    (df14['Work_Interest'].unique()),
    )
    
    t13 = st.selectbox(
    "Do you have social weakness",
    (df14['Social_Weakness'].unique()),
    )
    
    t14 = st.selectbox(
    "Do you have care options?",
    (df14['care_options'].unique()),
    )
    
    t15 = 0
    
    if t7 == 'Yes':
        t15 += 1
    if t8 == 'Yes':
        t15 += 1
    if t9 == 'Yes':
        t15 += 1
    if t10 == 'Yes':
        t15 += 1
    if t11 == 'Yes':
        t15 += 1
    if t12 == 'Yes':
        t15 += 1
    if t13 == 'Yes':
        t15 += 1
    
    
    # one_hot_columns = []
    test_point_arr = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15]
   
    df14.iloc[len(df14)-1] = test_point_arr
    # y1.iloc[len(df14)-1] = 'Test Point'
    # y2.iloc[len(df14)-1] = 'Test Point'
    
    
    
    model_list = ['SVM', 'Decision Tree','Random Forest']
    
    chosen_models = st.multiselect(
        "Choose a Model", model_list, ['SVM'], key="models"
    )
    if not indicators:
        st.error("Please select at least one Model")
    else:
        output_list = ['Gender', 'Treatment']
    
        outputs = st.multiselect(
            "Choose an Output", output_list, ['Gender'], key="outputs"
        )
        if not indicators:
            st.error("Please select at least one Output")
        else:
            # df = df14.drop('treatment',axis=1)
            # out = one_hot(df)
            # print(out)
            # color_indicator = y2
                
            
            
            for output in outputs:
                
                if output == 'Gender':
                    df = df14.drop('Gender',axis=1)
                
                
                if output == 'Treatment':
                    df = df14.drop('treatment',axis=1)
                    
            
                
                one_hot_new = one_hot(df)
                # print(one_hot_new)
                test_point = one_hot_new.iloc[len(df)-1]
                # y1.iloc[len(df14)-1] = 'Test Point'
                # y2.iloc[len(df14)-1] = 'Test Point'
                # print()
                y_gender = pd.DataFrame(y1)['Gender']
                y_gender.iloc[len(df14)-1] = 'Test Point'
                
                y_treatment = pd.DataFrame(y2)['treatment']
                y_treatment.iloc[len(df14)-1] = 'Test Point'
                 
                
                # print(y1)
                
                if output == 'Gender':
                    color_indicator = y_gender
                if output == 'Treatment':
                    color_indicator = y_treatment
                
                pca = PCA(n_components=2)

                X_train = pca.fit_transform(one_hot_new)

                x = []
                y = []
                for x_comp in X_train:
                    x.append(x_comp[0])
                    y.append(x_comp[1])
        
                
                
                data = {'Principle Component 1': x,
                        'Principle Component 2': y}
            
                df_scatter = pd.DataFrame(data)
                
                # if output != 'No Label':
                fig = px.scatter(df_scatter, x='Principle Component 1',y='Principle Component 2',color=color_indicator, title=f"PCA with {output} as Label")
            
                st.plotly_chart(fig)
                
                
                
                
                
                
                test_input = pd.DataFrame(test_point.T)
                test_input = test_input.transpose()

                if output == 'Gender':
                    for m in chosen_models:
                        if m == 'SVM':
                            y_pred = data_2_SVM_Gender.predict(test_input)
                            st.write(f"SVM predicts you are {y_pred}")
                            # print(y_pred)
                        if m == 'Decision Tree':
                            y_pred = data_2_DT_Gender.predict(test_input)
                            st.write(f"Decision Tree predicts you are {y_pred}")
                        if m == 'Random Forest':
                            y_pred = data_2_RF_Gender.predict(test_input)
                            st.write(f"Random Forest predicts you are {y_pred}")
                            
                if output == 'Treatment':
                    for m in chosen_models:
                        if m == 'SVM':
                            y_pred = data_2_SVM_Treatment.predict(test_input)
                            st.write(f"SVM predicts you said {y_pred} to Treatment")
                            # print(y_pred)
                        if m == 'Decision Tree':
                            y_pred = data_2_DT_Treatment.predict(test_input)
                            st.write(f"Decision Tree predicts you said {y_pred} to Treatment")
                        if m == 'Random Forest':
                            y_pred = data_2_RF_Treatment.predict(test_input)
                            st.write(f"Random Forest predicts you said {y_pred} to Treatment")
                # if output == 'Treatment':
                    
            
            
            
        
            
            
        
        
        
        
        
    
    
    
    
    
    
def depression_symptom_breakdown():


    st.markdown(f"# {list(page_names_to_funcs.keys())[5]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/
"""



    )

    # illness_scores = ['Major depression','Bipolar disorder','Eating disorders','Dysthymia','Schizophrenia','Anxiety disorders']


    df13.rename(columns={'Entity': 'Symptom'}, inplace=True)
    symptoms = st.multiselect(
        "Choose Symptom", df13.Symptom, ['Appetite change'], key="illness1"
    )
    if not symptoms:
        st.error("Please select at least one Symptom")
    else:
        data = df13.loc[df13['Symptom'].isin(symptoms)]
        
    st.write("### United States Depression Symptoms")
    
    
    # print(len(data))
    data_transposed = data[['Nearly every day','More than half the days','Several days','Not at all']].T.reset_index()
    # T.reset_index()
    # data = df13[symptoms]
    # print(data)
    
    for index, row in data.iterrows(): 
        symptom = row['Symptom']
   
   
   
        fig = px.pie(data_transposed, values = index, names = 'index',title=symptom)
        st.plotly_chart(fig)





def anxiety_treatment():


    st.markdown(f"# {list(page_names_to_funcs.keys())[4]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/
"""



    )

    # illness_scores = ['Major depression','Bipolar disorder','Eating disorders','Dysthymia','Schizophrenia','Anxiety disorders']


    df12.rename(columns={'Entity': 'Country'}, inplace=True)
    countries = st.multiselect(
        "Choose countries", df12.Country, ['Argentina'], key="illness1"
    )
    if not countries:
        st.error("Please select at least one Country")
    else:
        data = df12.loc[df12['Country'].isin(countries)]
        
    st.write("### Country Anxiety Treatment Gap")
    
    
    # print(len(data))
    data_transposed = data[['Potentially adequate treatment, conditional','Other treatments, conditional','Untreated, conditional']].T.reset_index()
    # T.reset_index()
    # data = df12[countries]
    # print(data)
    
    for index, row in data.iterrows(): 
        country = row['Country']
   
   
   
        fig = px.pie(data_transposed, values = index, names = 'index',title=country)
        st.plotly_chart(fig)
        
   
    # df = px.data.tips()
    # print(df)
   
    # group_labels = []
    # hist_data = []
    # data_arr = data.to_numpy()
    # for i in range(0, len(data.columns)):
    #     arr = []
    #     for value in data_arr:
    #         arr.append(value[i])
    #     hist_data.append(arr)
    
    
    # for column in data.columns:
    #     group_labels.append(column)
    
    
    
    # fig = ff.create_distplot(
    #         hist_data, group_labels)

    # # Plot!
    # st.plotly_chart(fig)
 
    
def illness_score():


    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/
"""



    )

    illness_scores = ['Major depression','Bipolar disorder','Eating disorders','Dysthymia','Schizophrenia','Anxiety disorders']

    illnesses = st.multiselect(
        "Choose Illness", illness_scores, ['Major depression'], key="illness1"
    )
    if not illnesses:
        st.error("Please select at least one illness score.")
    # else:
        # data = df1.loc[df1['Country'].isin(illnesses)]
        
    st.write("### Distribution of Illness Scores Across World Regions")
    
    data = df11[illnesses]
   
    group_labels = []
    hist_data = []
    data_arr = data.to_numpy()
    for i in range(0, len(data.columns)):
        arr = []
        for value in data_arr:
            arr.append(value[i])
        hist_data.append(arr)
    
    
    for column in data.columns:
        group_labels.append(column)
    
    
    
    fig = ff.create_distplot(
            hist_data, group_labels)

    # Plot!
    st.plotly_chart(fig)

        # print(data)
        # country_map = {}
        
        # for index, row in data.iterrows():
        #     country = row['Country']
        #     country_map[index] = country
        
        # data = data.T.reset_index()
        # # print(data)
        
        
        # data = pd.melt(data, id_vars=["index"]).rename(
        #     columns={"index": "year", "value": "Percent of Population with Schizophrenia", 'variable' : 'Country'}
        # )
        
        # # print(data)
        
        # data['Country'] = data['Country'].map(country_map)
        
        # chart = (
        #     alt.Chart(data)
        #     .mark_area(opacity=0.3)
        #     .encode(
        #         x="year:T",
        #         y=alt.Y("Percent of Population with Schizophrenia:Q", stack=None),
        #         color='Country'
        #     )
        # )
        
        
        # st.altair_chart(chart, use_container_width=True)    
    
    
def percent_pop():


    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/
"""
    )


    countries = st.multiselect(
        "Choose countries", list(df1.Country), ['Afghanistan'], key="SPercent"
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df1.loc[df1['Country'].isin(countries)]
        
        st.write("### Percent of Population with Schizophrenia", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        data = data.T.reset_index()
        # print(data)
        
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Percent of Population with Schizophrenia", 'variable' : 'Country'}
        )
        
        # print(data)
        
        data['Country'] = data['Country'].map(country_map)
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Percent of Population with Schizophrenia:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
        
        
        countries1 = st.multiselect(
        "Choose countries", list(df2.Country), ['Afghanistan'],key="DepressionPercent"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df2.loc[df2['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Percent of Population with Depression", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Percent of Population with Depression", 'variable' : 'Country'}
        )
        
        # print(country_map)
        # print(data)
        # data = data.drop(index=0)
        
        data['Country'] = data['Country'].map(country_map)
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Percent of Population with Depression:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
    countries1 = st.multiselect(
        "Choose countries", list(df3.Country), ['Afghanistan'],key="BipolarPercent"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df3.loc[df3['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Percent of Population with Bipolar", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Percent of Population with Bipolar", 'variable' : 'Country'}
        )
        # data = data.drop(index=0)
        # print(data)
        
        # print(country_map)
        
        data['Country'] = data['Country'].map(country_map)
        
        
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Percent of Population with Bipolar:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
        
    countries1 = st.multiselect(
        "Choose countries", list(df4.Country), ['Afghanistan'],key="EatingPercent"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df4.loc[df4['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Percent of Population with Eating Disorder", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Percent of Population with Eating Disorder", 'variable' : 'Country'}
        )
        # data = data.drop(index=0)
        # print(data)
        
        # print(country_map)
        
        data['Country'] = data['Country'].map(country_map)
        
        
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Percent of Population with Eating Disorder:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
        
    countries1 = st.multiselect(
        "Choose countries", list(df5.Country), ['Afghanistan'],key="AnxietyPercent"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df5.loc[df5['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Percent of Population with Anxiety", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Percent of Population with Anxiety", 'variable' : 'Country'}
        )
        # data = data.drop(index=0)
        # print(data)
        
        # print(country_map)
        
        data['Country'] = data['Country'].map(country_map)
        
        
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Percent of Population with Anxiety:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
 


def daly_pop():

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/
"""
    )


    countries = st.multiselect(
        "Choose countries", list(df6.Country), ['Afghanistan'], key="SDALY"
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df6.loc[df6['Country'].isin(countries)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Schizophrenia DALYs", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        data = data.T.reset_index()
        # print(data)
        
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Schizophrenia DALYs", 'variable' : 'Country'}
        )
        
        # print(data)
        
        data['Country'] = data['Country'].map(country_map)
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Schizophrenia DALYs:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
        
        
        countries1 = st.multiselect(
        "Choose countries", list(df7.Country), ['Afghanistan'],key="DepressionDALY"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df7.loc[df7['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Depression DALYs", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Depression DALYs", 'variable' : 'Country'}
        )
        
        # print(country_map)
        # print(data)
        # data = data.drop(index=0)
        
        data['Country'] = data['Country'].map(country_map)
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Depression DALYs:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
    countries1 = st.multiselect(
        "Choose countries", list(df8.Country), ['Afghanistan'],key="BipolarDALY"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df8.loc[df8['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Bipolar DALYs", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Bipolar DALYs", 'variable' : 'Country'}
        )
        # data = data.drop(index=0)
        # print(data)
        
        # print(country_map)
        
        data['Country'] = data['Country'].map(country_map)
        
        
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Bipolar DALYs:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
        
    countries1 = st.multiselect(
        "Choose countries", list(df9.Country), ['Afghanistan'],key="EatingDALYs"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df9.loc[df9['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Eating Disorder DALYs", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Eating Disorder DALYs", 'variable' : 'Country'}
        )
        # data = data.drop(index=0)
        # print(data)
        
        # print(country_map)
        
        data['Country'] = data['Country'].map(country_map)
        
        
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Eating Disorder DALYs:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)
        
        
        
    countries1 = st.multiselect(
        "Choose countries", list(df10.Country), ['Afghanistan'],key="AnxietyDALYs"
    )
    if not countries1:
        st.error("Please select at least one country.")
    else:
        data = df10.loc[df10['Country'].isin(countries1)]
        data = data.drop("Unnamed: 0", axis=1)
        
        st.write("### Anxiety DALYs", data.sort_index())

        # print(data)
        country_map = {}
        
        for index, row in data.iterrows():
            country = row['Country']
            country_map[index] = country
        
        
        
        
        
        data = data.T.reset_index()
        # print(data)
        
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Anxiety DALYs", 'variable' : 'Country'}
        )
        # data = data.drop(index=0)
        # print(data)
        
        # print(country_map)
        
        data['Country'] = data['Country'].map(country_map)
        
        
        
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Anxiety DALYs:Q", stack=None),
                color='Country'
            )
        )
        
        
        st.altair_chart(chart, use_container_width=True)






page_names_to_funcs = {
    "—": intro,
    "Percent Population Afflicted": percent_pop,
    "DALYs of Population" : daly_pop,
    "Mental Illness Score" : illness_score,
    "Anxiety Treatment" : anxiety_treatment,
    "Depression Symptoms Breakdown" : depression_symptom_breakdown,
    "World Mental Health with Models" : pca_dataset_2,
    "Student Mental Health with Models" : pca_dataset_3,
    "World Stats with Models" : pca_dataset_1,
    "Average World Stats with Models" : pca_dataset_1_avg
}

demo_name = st.sidebar.selectbox("Choose a Page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()