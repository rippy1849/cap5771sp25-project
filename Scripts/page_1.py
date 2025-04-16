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


data_path = 'C:\\Users\\fweep\\OneDrive\\Documents\\Code\\cap5771sp25-project\\Data\\'


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




def intro():
    import streamlit as st

    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )
    
    
    
    
def illness_score():


    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This page shows off the Data collected from Dataset 1.

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/data?select=5-+anxiety-disorders-treatment-gap.csv
"""



    )

    illness_scores = ['Major depression','Bipolar disorder','Eating disorders','Dysthymia','Schizophrenia','Anxiety disorders']

    illnesses = st.multiselect(
        "Choose countries", illness_scores, ['Major depression'], key="illness1"
    )
    if not illnesses:
        st.error("Please select at least one illness score.")
    # else:
        # data = df1.loc[df1['Country'].isin(illnesses)]
        
    st.write("### Distribution of Illness Scores Across World Regions")
    # st.write("### Distribution of Illness Scores Across World Regions")
    
    data = df11[illnesses]
    print(data)

    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']
    # group_labels = ['Group 1']
    
    

    # Create distplot with custom bin_size
    # fig = ff.create_distplot(
    #         hist_data, group_labels)
    
    fig = ff.create_distplot(
            data, group_labels)

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

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/data?select=5-+anxiety-disorders-treatment-gap.csv
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

(Data courtesy of the [Kaggle] https://www.kaggle.com/datasets/imtkaggleteam/mental-health/data?select=5-+anxiety-disorders-treatment-gap.csv
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
    "Mental Illness Score" : illness_score
}

demo_name = st.sidebar.selectbox("Choose a Page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()