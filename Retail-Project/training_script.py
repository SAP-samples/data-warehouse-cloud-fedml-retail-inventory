import os
import time
import pandas as pd
import argparse
from sklearn import __version__ as sklearnver
import joblib

import lightgbm as lgb
import numpy as np
from fedml_azure import DbConnection
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json

############################## Helper Functions ##############################
def get_dwc_data(table_name,table_size):
    db = DbConnection()
    start_time = time.time()
    data = db.get_data_with_headers(table_name=table_name, size=table_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    data = pd.DataFrame(data[0], columns=data[1])
    return data

def create_lags_year(df, lag_variables):
    df = df.sort_values(['retailer', 'productsku',
                            'calendar_year', 'calendar_month'])
    lag_per = [1]
    for lag_col in lag_variables:
        for lag in lag_per:
            df[lag_col + '_lag_' + str(lag)] = calc_lags(
                df, ['retailer', 'productsku', 'calendar_year'], lag_col, lag)
    return df

def calc_cum_mean(df, groupby_cols, col):
    cum_mean = df.groupby(groupby_cols)[col].apply(
        lambda x: x.expanding().mean().fillna(value=0))
    return np.log1p(cum_mean.values)

def calc_lags(df, groupby_cols, on_col, lag):
    lags = df.groupby(groupby_cols)[on_col].shift(periods=lag).fillna(value=0)
#     return lags.values
    return np.log1p(lags.values)

def encode_cats(df, cat_cols):
    for cat in cat_cols:
        #         print(cat)
        # le = preprocessing.LabelEncoder()
        # le.fit(df[cat].unique())
        # df[cat] = le.transform(df[cat])
        df[cat] = df[cat].astype('category')

def train_test_split(df):
    df_test = df[(df['calendar_year'] == 2020) & (df['calendar_month'].isin([10, 11, 12]))]
    df_train = df[~(((df['calendar_year'] == 2020)) & (df['calendar_month'].isin([10, 11, 12])))]
    results_df = df_test[['retailer', 'productsku', 'calendar_year',
                            'calendar_month', 'mtd_consumption']]
    return df_train, df_test, results_df

############################## Required Functions ##############################
def store_model(model_file_name,model_output):
    print('storing the model....')
    os.makedirs('./outputs', exist_ok=True)
    with open(model_file_name, 'wb') as file:
        joblib.dump(value=model_output, filename='outputs/' + model_file_name)

############################## Training Script Part ##############################

parser = argparse.ArgumentParser()

# Sagemaker specific arguments. Defaults are set in the environment variables.
parser.add_argument('--model_file_name', type=str)
parser.add_argument('--dist_table', type=str)
parser.add_argument('--dist_size', type=str)
parser.add_argument('--product_table', type=str)
parser.add_argument('--product_size', type=str)
parser.add_argument('--retailer_table', type=str)
parser.add_argument('--retailer_size', type=str)

parser.add_argument('--retailer3_table', type=str)
parser.add_argument('--retailer3_size', type=str)

parser.add_argument('--retailer2_table', type=str)
parser.add_argument('--retailer2_size', type=str)

parser.add_argument('--retailer1_table', type=str)
parser.add_argument('--retailer1_size', type=str)

args = parser.parse_args()

print('\n\n********* Handling Data - Splitting into Train and Test *********n\n')
print(args.dist_table)
dist_data = get_dwc_data(args.dist_table, float(args.dist_size))
print('got dist_data')
product_data = get_dwc_data(args.product_table, float(args.product_size))
print('got product_data')
retailer_data = get_dwc_data(args.retailer_table, float(args.retailer_size))
print('got retailer_data')
    
retailer3_data = get_dwc_data(args.retailer3_table, float(args.retailer3_size))
print('got retailer3_data')
retailer2_data = get_dwc_data(args.retailer2_table, float(args.retailer2_size))
print('got retailer2_data')
retailer1_data = get_dwc_data(args.retailer1_table, float(args.retailer1_size))
print('got retailer1_data')

master_dist = dist_data.merge(product_data, how='left', left_on='productsku', right_on='Product').drop('Product', axis=1)
master_dist = master_dist.merge(retailer_data, how='left', left_on='retailer', right_on='RetailID').drop('RetailID', axis=1)
    
master_dist = master_dist[['productsku', 'retailer', 'Retailer', 'Type', 'calendar_year', 'calendar_month', 'max_allocation',
    'inventory_requested', 'mtd_consumption',  'Color',
    'Collection', 'Style', 'Season', 'Demographic', 'Fit', 'Material',
    ]]
    
retail_master = pd.concat([retailer2_data, retailer3_data, retailer1_data])
    
retail_master['previous_mo'] = retail_master['calendar_month'] - 1
master_dist['previous_mo'] = master_dist['calendar_month'] - 1
    
master_dist = master_dist.merge(retail_master, left_on=['productsku', 'retailer', 'calendar_year', 'previous_mo'], right_on=['productsku', 'retailer', 'calendar_year', 'previous_mo'] )
    
master_dist = master_dist[['productsku', 'retailer', 'Retailer', 'Type', 'calendar_year',
       'calendar_month_x','inventory', 'sales', 'max_allocation', 'inventory_requested',
       'mtd_consumption', 'Color', 'Collection', 'Style', 'Season',
       'Demographic', 'Fit', 'Material']]
    
master_dist = master_dist.rename(columns={'calendar_month_x': 'calendar_month', 'inventory': 'last_mo_inventory', 'sales': 'last_mo_sales' })
    
data = create_lags_year(df = master_dist, lag_variables = ['mtd_consumption'])
data['Last_Mo_Sales_Avg'] = calc_cum_mean(data, ['productsku', 'calendar_year', 'calendar_month'], 'last_mo_sales')
    
model_data = data[['productsku', 'retailer', 'Retailer', 'Type', 'calendar_year',
    'calendar_month', 'last_mo_inventory', 'last_mo_sales', 'Last_Mo_Sales_Avg',
    'mtd_consumption_lag_1',
    'max_allocation', 'inventory_requested', 'mtd_consumption', 'Color',
    'Collection', 'Style', 'Season', 'Demographic', 'Fit', 'Material']]

cat_cols = ['productsku', 'retailer','Type', 'calendar_year',
    'calendar_month', 'Color',
    'Collection', 'Style', 'Season', 'Demographic', 'Fit', 'Material']
    
    
log_cols = ['last_mo_inventory', 'last_mo_sales', 'mtd_consumption', 'max_allocation' ]
for x in log_cols:
    model_data[x] = np.log1p(model_data[x])
        
encode_cats(model_data, cat_cols)
    

estimator = lgb.LGBMRegressor(objective='regression')

train, test, results = train_test_split(model_data)
print('successfully data split into train and test.')
train_y = train['mtd_consumption']
test_y = test['mtd_consumption']

del train['mtd_consumption']
del train['inventory_requested']
del train['Retailer']

del test['mtd_consumption']
del test['inventory_requested']
del test['Retailer']

print('fitting the model...')

estimator.fit(train, train_y)
#     estimator = estimator.fit(X_train, y_train)
y_pred = np.expm1(estimator.predict(test))
# y_pred = np.round(y_pred)
y_pred = [0 if i < 0 else i for i in y_pred]

results['Prediction'] = y_pred

results['mtd_consumption'] = np.expm1(results['mtd_consumption'])

#Save the model to the location specified by args.model_dir
store_model(args.model_file_name,estimator)

print("saved model!")
