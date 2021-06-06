import pandas as pd
import numpy as np
import statistics

def load_data():
    product_df=pd.read_csv('product_table.csv')
    sales_df=pd.read_csv('sales_master.csv')
    return product_df,sales_df

def calculate_markdown_and_lift(df):
    df['markdown1'] = (df['gross_amount1'] - df['net_amount1']) / df['gross_amount1']
    df['markdown2'] = (df['gross_amount2'] - df['net_amount2']) / df['gross_amount2']

    df['uplift1'] = (df['purchases1'] - df['purchases0']) / df['purchases0']
    df['uplift2'] = (df['purchases2'] - df['purchases0']) / df['purchases0']

    df['uplift_change_2nd_week'] = df['uplift2'] / df['uplift1']
    return df

def df_to_pivot(df, values, index, columns):
    new_df = pd.pivot_table(df, values=values, index=index, columns=columns).reset_index()
    new_df = pd.DataFrame(new_df.to_records())
    new_df.columns = [hdr.replace("(", "").replace(")", "").replace("'", "").replace(",", "").replace(" ", "") for hdr
                      in new_df.columns]
    return new_df

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def feature_engineering():
    # Importing Datasets
    product_df,sales_df=load_data()

    # Separating variant into article and size
    sales_df['variant'] = sales_df['variant'].apply(str)
    sales_df['article'] = sales_df['variant'].str[:9]
    sales_df['size'] = sales_df['variant'].str[9:]

    # First, if purchases is zero, them amounts should be zero also. Applying that.
    sales_df.loc[sales_df['purchases'].eq(0), 'net_amount'] = 0
    sales_df.loc[sales_df['purchases'].eq(0), 'gross_amount'] = 0

    # Filling missing values
    sales_df['gross_amount'] = sales_df.groupby(['date','purchases','article'])['gross_amount'].transform(lambda x: x.fillna(x.mean()))
    sales_df['net_amount'] = sales_df.groupby(['date','purchases','article'])['net_amount'].transform(lambda x: x.fillna(x.mean()))
    sales_df['gross_amount'] = sales_df.groupby(['date','article'])['gross_amount'].transform(lambda x: x.fillna(x.mean()))
    sales_df['net_amount'] = sales_df.groupby(['date','article'])['net_amount'].transform(lambda x: x.fillna(x.mean()))
    sales_df['gross_amount'] = sales_df.groupby('article')['gross_amount'].transform(lambda x: x.fillna(x.mean()))
    sales_df['net_amount'] = sales_df.groupby('article')['net_amount'].transform(lambda x: x.fillna(x.mean()))

    # Merging products and sales datasets into one dataframe
    product_df['article'] = product_df['article'].apply(str)
    df = sales_df.merge(product_df, how='left', on='article')

    # Adding Week feature to dataset
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.strftime("%V")
    df['week'] = [int(w) for w in df['week']]
    df['week'] = df['week'] - 40

    # Creating a dataframe in week-product level
    df_weekly_product = df.groupby(
        ['week', 'PRODUCT_CODE', 'PRODUCT_TYPE_NAME', 'DEPARTMENT_NAME', 'INDEX_GROUP_NAME', 'SECTION_NAME']
    ).agg(
        purchases=('purchases', 'sum'),
        net_amount=('net_amount', 'sum'),
        gross_amount=('gross_amount', 'sum')
    ).reset_index()

    # Creating prodduct level, department level and index level datasets to use in next steps
    df_weekly_product_pivot = df_to_pivot(df_weekly_product, ['purchases', 'net_amount', 'gross_amount'],
                                          ['PRODUCT_CODE', 'PRODUCT_TYPE_NAME', 'DEPARTMENT_NAME', 'INDEX_GROUP_NAME',
                                           'SECTION_NAME'], 'week')
    df_weekly_department_pivot = df_to_pivot(df_weekly_product, ['purchases', 'net_amount', 'gross_amount'],
                                             ['DEPARTMENT_NAME', 'INDEX_GROUP_NAME'], 'week')
    df_weekly_index_pivot = df_to_pivot(df_weekly_product, ['purchases', 'net_amount', 'gross_amount'],
                                        ['INDEX_GROUP_NAME'], 'week')
    df_weekly_product_pivot = calculate_markdown_and_lift(df_weekly_product_pivot)
    df_weekly_department_pivot = calculate_markdown_and_lift(df_weekly_department_pivot)
    df_weekly_index_pivot = calculate_markdown_and_lift(df_weekly_index_pivot)

    # Filling missing values with zero
    df_weekly_product_pivot = df_weekly_product_pivot.fillna(0)
    df_weekly_department_pivot = df_weekly_department_pivot.fillna(0)

    # Creating company based metrics for EDA part
    total_uplift1 = (df_weekly_product_pivot.purchases1.sum() - df_weekly_product_pivot.purchases0.sum()) / df_weekly_product_pivot.purchases0.sum()
    total_uplift2 = (df_weekly_product_pivot.purchases2.sum() - df_weekly_product_pivot.purchases0.sum()) / df_weekly_product_pivot.purchases0.sum()
    avg_markdown1 = (df_weekly_product_pivot.gross_amount1.sum() - df_weekly_product_pivot.net_amount1.sum()) / df_weekly_product_pivot.gross_amount1.sum()
    avg_markdown2 = (df_weekly_product_pivot.gross_amount2.sum() - df_weekly_product_pivot.net_amount2.sum()) / df_weekly_product_pivot.gross_amount2.sum()

    uplifts = [total_uplift1, total_uplift2]
    markdowns = [avg_markdown1, avg_markdown2]

    total = pd.DataFrame(data={'weeks': ['week1', 'week2'], 'uplifts': uplifts, 'markdowns': markdowns})

    # Preparing evaluating dataframe to use on visualization

    # Making predictions
    df_weekly_product_pivot['prediction1'] = np.exp(3 * df_weekly_product_pivot['markdown1'])
    df_weekly_department_pivot['prediction1'] = np.exp(3 * df_weekly_department_pivot['markdown1'])
    df_weekly_index_pivot['prediction1'] = np.exp(3 * df_weekly_index_pivot['markdown1'])

    # To avoid divide by zero error, added uplift very little value
    df_weekly_product_pivot['uplift1'] = np.where(df_weekly_product_pivot['uplift1'] == 0, 0.000001,
                                                  df_weekly_product_pivot['uplift1'])
    df_weekly_department_pivot['uplift1'] = np.where(df_weekly_department_pivot['uplift1'] == 0, 0.000001,
                                                     df_weekly_department_pivot['uplift1'])

    # Calculating evaluation metrics for 3 different level.

    prod_rmse = rmse(df_weekly_product_pivot.prediction1, df_weekly_product_pivot.uplift1)
    prod_std = statistics.stdev(df_weekly_product_pivot.uplift1)

    dep_rmse = rmse(df_weekly_department_pivot.prediction1, df_weekly_department_pivot.uplift1)
    dep_std = statistics.stdev(df_weekly_department_pivot.uplift1)

    index_rmse = rmse(df_weekly_index_pivot.prediction1, df_weekly_index_pivot.uplift1)
    index_std = statistics.stdev(df_weekly_index_pivot.uplift1)

    RMSE = [prod_rmse, dep_rmse, index_rmse]
    STD = [prod_std, dep_std, index_std]

    evaluation = pd.DataFrame(data={'Metric': ['RMSE', 'STD'], 'Product': [prod_rmse, prod_std], 'Department': [dep_rmse, dep_std],
              'Index': [index_rmse, index_std]})

    return df,df_weekly_product_pivot,df_weekly_department_pivot,df_weekly_index_pivot,total,avg_markdown1,avg_markdown2,total_uplift1,total_uplift2,evaluation