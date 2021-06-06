# Importing libraries
import pandas as pd
import numpy as np

import math
import statistics

from datetime import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from data_prepration import feature_engineering

@st.cache()
def get_data():
    df,df_weekly_product_pivot,df_weekly_department_pivot,df_weekly_index_pivot,total,avg_markdown1,avg_markdown2,total_uplift1,total_uplift2,evaluation=feature_engineering()
    return df,df_weekly_product_pivot,df_weekly_department_pivot,df_weekly_index_pivot,total,avg_markdown1,avg_markdown2,total_uplift1,total_uplift2,evaluation

def feature_descriptions(df):
    feat_desc = pd.DataFrame({'Description': df.columns,
                              'Data Types': df.dtypes,
                              'Values': [df[i].unique() for i in df.columns],
                              'Number of unique values': [len(df[i].unique()) for i in df.columns]}).reset_index()
    return feat_desc[['Description','Data Types','Values','Number of unique values']]


def missing_data(df, top_x):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data.head(top_x)


def graph_sales_and_changes(df, column_group_by, column_agg):
    # group by date and get sum of sales, and precent change
    total_sales = df.groupby(column_group_by)[column_agg].sum()
    pct_change_sales = df.groupby(column_group_by)[column_agg].sum().pct_change(periods=7)
    fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
    # plot sum of sales over time(day)

    ax1 = total_sales.plot(legend=True, ax=axis1, marker='o')
    ax1.set_title("Total Sales", fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax1.set_xticks(range(len(total_sales)))
    ax1.set_xticklabels(total_sales.index.tolist(), rotation=90)
    ax1.axvline(x="2017-10-09", color='r', linestyle='--')
    ax1.axvline(x="2017-10-16", color='r', linestyle='--')

    # plot precent change for sales over time(day)
    ax2 = pct_change_sales.plot(legend=True, ax=axis2, marker='o', rot=90, colormap="summer")
    ax2.set_title("Sales Percent Change vs Last Week", fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax2.axvline(x="2017-10-09", color='r', linestyle='--')
    ax2.axvline(x="2017-10-16", color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')
    return fig

def uplifts_markdown_company():
    # Pivotting Uplifts and Markdowns in Company Level
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplot(1, 2, 1).set_title('Total Uplifts', fontsize=16)
    sns.barplot(total.weeks, total.uplifts)
    plt.subplot(1, 2, 2).set_title('Average Markdowns', fontsize=16)
    sns.barplot(total.weeks, total.markdowns)
    return fig

def index_level_graph1():
    # Visualizing uplift and markdown rates for week 1 and week 2 in index group level

    # setting the dimensions of the plot
    fig, ax = plt.subplots(figsize=(15, 12))

    plt.subplot(2, 2, 3).set_title('1st week markdown', fontsize=16)
    sns.barplot(df_weekly_index_pivot.INDEX_GROUP_NAME, df_weekly_index_pivot.markdown1)
    plt.axhline(y=avg_markdown1, color='y', linestyle='--')

    plt.subplot(2, 2, 1).set_title('1st week uplift', fontsize=16)
    sns.barplot(df_weekly_index_pivot.INDEX_GROUP_NAME, df_weekly_index_pivot.uplift1)
    plt.axhline(y=total_uplift1, color='y', linestyle='--')

    plt.subplot(2, 2, 4).set_title('2nd week markdown', fontsize=16)
    sns.barplot(df_weekly_index_pivot.INDEX_GROUP_NAME, df_weekly_index_pivot.markdown2)
    plt.axhline(y=avg_markdown2, color='y', linestyle='--')

    plt.subplot(2, 2, 2).set_title('2nd week uplift', fontsize=16)
    sns.barplot(df_weekly_index_pivot.INDEX_GROUP_NAME, df_weekly_index_pivot.uplift2)
    plt.axhline(y=total_uplift2, color='y', linestyle='--')
    return fig

def index_level_graph2():
    # Visualizing top and worst 10 uplifts in Department level for "Home" index group

    fig, ax = plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1).set_title('Best 10 Departments with highest uplift', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot.groupby('DEPARTMENT_NAME')[
        'uplift1'].sum().nlargest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("Uplift 1")

    plt.subplot(2, 2, 2).set_title('Worst 10 Departments with lowest uplift', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot.groupby('DEPARTMENT_NAME')[
        'uplift1'].sum().nsmallest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("Uplift 1")
    return fig

def index_level_graph3(index_group):
    # Visualizing top and worst 10 uplifts in Department level for "Home" index group

    fig, ax = plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1).set_title('Best 10 Departments with highest uplift', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot[df_weekly_department_pivot['INDEX_GROUP_NAME']==index_group].groupby('DEPARTMENT_NAME')[
        'uplift1'].sum().nlargest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("Uplift 1")

    plt.subplot(2, 2, 2).set_title('Worst 10 Departments with lowest uplift', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot[df_weekly_department_pivot['INDEX_GROUP_NAME']==index_group].groupby('DEPARTMENT_NAME')[
        'uplift1'].sum().nsmallest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("Uplift 1")
    return fig

def index_level_graph4():
    # Visualizing top and worst 10 markdowns in Department level for "Home" index group

    fig, ax = plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1).set_title('Top 10 Home Departments with highest markdown', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot.groupby('DEPARTMENT_NAME')[
        'markdown1'].mean().nlargest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("markdown1")

    plt.subplot(2, 2, 2).set_title('Worst 10 Home Departments with lowest markdown', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot.groupby('DEPARTMENT_NAME')[
        'markdown1'].mean().nsmallest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("markdown1")
    return fig

def index_level_graph5(index_group):
    # Visualizing top and worst 10 markdowns in Department level for "Home" index group

    fig, ax = plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1).set_title('Top 10 Departments with highest markdown', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot[df_weekly_department_pivot.INDEX_GROUP_NAME == index_group].groupby('DEPARTMENT_NAME')[
        'markdown1'].mean().nlargest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("markdown1")

    plt.subplot(2, 2, 2).set_title('Worst 10 Departments with lowest markdown', fontsize=16)
    ProductReturns = \
    df_weekly_department_pivot[df_weekly_department_pivot.INDEX_GROUP_NAME == index_group].groupby('DEPARTMENT_NAME')[
        'markdown1'].mean().nsmallest(10).plot(kind='bar')
    ProductReturns.set_xlabel("Department Name")
    ProductReturns.set_ylabel("markdown1")
    return fig

def dep_level_graph1(index_group,department_name):
    # Visualizing top and worst 10 uplifts in Department level

    fig, ax = plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1).set_title('Best 10 Products', fontsize=16)
    ProductReturns = df_weekly_product_pivot[df_weekly_product_pivot.INDEX_GROUP_NAME == index_group][df_weekly_product_pivot.DEPARTMENT_NAME == department_name].groupby('PRODUCT_CODE')['uplift1'].sum().nlargest(10).plot(
        kind='bar')
    ProductReturns.set_xlabel("Product Code")
    ProductReturns.set_ylabel("Uplift 1")

    plt.subplot(2, 2, 2).set_title('Worst 10 Products', fontsize=16)
    ProductReturns = df_weekly_product_pivot[df_weekly_product_pivot.INDEX_GROUP_NAME == index_group][df_weekly_product_pivot.DEPARTMENT_NAME == department_name].groupby('PRODUCT_CODE')['uplift1'].sum().nsmallest(10).plot(
        kind='bar')
    ProductReturns.set_xlabel("Product Code")
    ProductReturns.set_ylabel("Uplift 1")
    return fig

def product_level_graph1():
    # Visualizing uplift and markdown stats in produdct level

    fig, ax = plt.subplots(figsize=(15, 6))

    plt.subplot(3, 2, 1).set_title('1st week uplifts', fontsize=16)
    sns.boxplot(df_weekly_product_pivot.uplift1)

    plt.subplot(3, 2, 3).set_title('2nd week uplifts', fontsize=16)
    sns.boxplot(df_weekly_product_pivot.uplift2)

    plt.subplot(3, 2, 2).set_title('1st week markdowns', fontsize=16)
    sns.boxplot(df_weekly_product_pivot.markdown1)

    plt.subplot(3, 2, 4).set_title('2nd week markdowns', fontsize=16)
    sns.boxplot(df_weekly_product_pivot.markdown2)

    plt.subplot(3, 1, 3).set_title('uplift_change_2nd_week', fontsize=16)
    sns.boxplot(df_weekly_product_pivot.markdown1)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)
    return fig

def prediction_graph1():
    # Checking how good prediction model fitting with data

    # setting the dimensions of the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Product Level
    plt.subplot(3, 1, 1).set_title('Product Level Uplift-Forecast 1st Week', fontsize=16)
    sns.scatterplot(data=df_weekly_product_pivot, x="markdown1", y="uplift1")
    X_plot = np.linspace(0, 1, 100)
    Y_plot = np.exp(3 * X_plot)
    plt.plot(X_plot, Y_plot, color='r')

    # Department Level
    plt.subplot(3, 1, 2).set_title('Department Level Uplift-Forecast 1st Week', fontsize=16)
    sns.scatterplot(data=df_weekly_department_pivot, x="markdown1", y="uplift1")
    X_plot = np.linspace(0, 1, 100)
    Y_plot = np.exp(3 * X_plot)
    plt.plot(X_plot, Y_plot, color='r')

    # Index Level
    plt.subplot(3, 1, 3).set_title('Index Level Uplift-Forecast 1st Week', fontsize=16)
    sns.scatterplot(data=df_weekly_index_pivot, x="markdown1", y="uplift1")
    X_plot = np.linspace(0, 1, 100)
    Y_plot = np.exp(3 * X_plot)
    plt.plot(X_plot, Y_plot, color='r')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.5)
    return fig

def predictions_graph2():
    # Evaluation Metrics

    fig, ax = plt.subplots(figsize=(15, 6))

    plt.subplot(1, 3, 1).set_title('Product Level Evaluation', fontsize=16)
    sns.barplot(evaluation.Metric, evaluation.Product)

    plt.subplot(1, 3, 2).set_title('Department Level Evaluation', fontsize=16)
    sns.barplot(evaluation.Metric, evaluation.Department)

    plt.subplot(1, 3, 3).set_title('Index Level Evaluation', fontsize=16)
    sns.barplot(evaluation.Metric, evaluation.Index)



#### Application Part ####
# Getting Dataframes
df,df_weekly_product_pivot,df_weekly_department_pivot,df_weekly_index_pivot,total,avg_markdown1,avg_markdown2,total_uplift1,total_uplift2,evaluation=get_data()

select_status = st.sidebar.radio("Select the Level of Analyze", ('Home Page','Index Level','Department Level', 'Product Level','Predictions'))
st.set_option('deprecation.showPyplotGlobalUse', False)

if select_status=='Home Page':
    st.title('MARKDOWN ANALYSIS')
    st.write("This app is created to analysis markdown with details. "
             "Data includes daily sales of fashion retail products for 3 weeks, 1st week (week-0) is the week without any markdown. "
             "Other 2 weeks, week-1 and week-2 are the first and second week of markdown session.")

    st.header("Sales by day")
    st.pyplot(graph_sales_and_changes(df, 'date', 'purchases'))

    st.write("As we can see, sales increased almost 3 times in 1st markdown week. "
             "Markdown effect continues in the 2nd markdown week also.")

    st.header("Uplifts and Markdowns")
    st.pyplot(uplifts_markdown_company())


elif select_status=='Index Level':
    st.title('Index Level Analysis')

    st.header("Index groups Uplifts and Markdowns")
    st.pyplot(index_level_graph1())

    # Filters
    index_groups = df['INDEX_GROUP_NAME'].drop_duplicates().reset_index()

    index_group_name = st.sidebar.selectbox('Select an Index Group', index_groups.INDEX_GROUP_NAME)
    #department_name = st.sidebar.selectbox('Select a Department Group', df['DEPARTMENT_NAME'].drop_duplicates())

    st.title('Filtered Index Analysis ({})'.format(index_group_name))

    filtered_data = df[df['INDEX_GROUP_NAME']==index_group_name]
    #filtered_data = filtered_data[filtered_data['DEPARTMENT_NAME'] == department_name]

    st.header("Sales by day of {}".format(index_group_name))
    st.pyplot(graph_sales_and_changes(filtered_data, 'date', 'purchases'))

    st.header("Best and worst 10 uplift Departments of {}".format(index_group_name))
    st.pyplot(index_level_graph3(index_group_name))

    st.header("Best and worst 10 Markdown Departments of {}".format(index_group_name))
    st.pyplot(index_level_graph5(index_group_name))


elif select_status=='Department Level':
    st.title('Department Level Analysis')

    index_groups = df['INDEX_GROUP_NAME'].drop_duplicates().reset_index()

    index_group_name = st.sidebar.selectbox('Select an Index Group', index_groups.INDEX_GROUP_NAME)
    filtered_data = df[df['INDEX_GROUP_NAME']==index_group_name]

    department_name=filtered_data.DEPARTMENT_NAME.drop_duplicates().reset_index()

    department_name = st.sidebar.selectbox('Select a Department Group', department_name.DEPARTMENT_NAME)
    filtered_data = filtered_data[filtered_data['DEPARTMENT_NAME']==department_name]

    st.header("Best and worst 10 uplift Departments")
    st.pyplot(index_level_graph2())

    st.header("Best and worst 10 Markdown Departments")
    st.pyplot(index_level_graph4())

    st.title('Filtered Department Analysis ({},{})'.format(index_group_name,department_name))

    st.header("Sales by day of {} and {}".format(index_group_name,department_name))
    st.pyplot(graph_sales_and_changes(filtered_data, 'date', 'purchases'))

    st.header("Best and worst 10 uplift products of {} and {}".format(index_group_name,department_name))
    st.pyplot(dep_level_graph1(index_group_name,department_name))

elif select_status=='Product Level':
    st.title('Product Level Analysis')

    index_groups = df['INDEX_GROUP_NAME'].drop_duplicates().reset_index()

    index_group_name = st.sidebar.selectbox('Select an Index Group', index_groups.INDEX_GROUP_NAME)
    filtered_data = df[df['INDEX_GROUP_NAME']==index_group_name]

    department_name=filtered_data.DEPARTMENT_NAME.drop_duplicates().reset_index()

    department_name = st.sidebar.selectbox('Select a Department Group', department_name.DEPARTMENT_NAME)
    filtered_data = filtered_data[filtered_data['DEPARTMENT_NAME']==department_name]

    products=filtered_data.PRODUCT_CODE.drop_duplicates().reset_index()

    product_code = st.sidebar.selectbox('Select a Department Group', products.PRODUCT_CODE)
    filtered_data = filtered_data[filtered_data['PRODUCT_CODE']==product_code]

    st.header("Uplifts and Markdowns boxplots")
    st.pyplot(product_level_graph1())

    st.title('Filtered Product Analysis ({},{},{})'.format(index_group_name,department_name,product_code))

    st.header("Sales by day of {} and {} and {}".format(index_group_name,department_name,product_code))
    st.pyplot(graph_sales_and_changes(filtered_data, 'date', 'purchases'))


else:
    st.title('Predictions Analysis')

    st.header("Predictions fitting graph")
    st.pyplot(prediction_graph1())

    st.header('Predictions Evaluation')
    st.pyplot(predictions_graph2())