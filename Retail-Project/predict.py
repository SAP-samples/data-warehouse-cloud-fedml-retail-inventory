import joblib
import numpy as np
import pandas as pd
import os
import json
# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'retailmodel.pkl'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)
    model = joblib.load(model_path)

def calc_lags(df, groupby_cols, on_col, lag):
    lags = df.groupby(groupby_cols)[on_col].shift(periods=lag).fillna(value=0)
    return np.log1p(lags.values)

def create_lags_year(df, lag_variables):
    print('in create_lag_year')
    df = df.sort_values(['retailer', 'productsku', 'calendar_year', 'calendar_month'])
    lag_per = [1]
    for lag_col in lag_variables:
        for lag in lag_per:
            df[lag_col + '_lag_' + str(lag)] = calc_lags(df, ['retailer', 'productsku', 'calendar_year'], lag_col, lag)
    return df

def calc_cum_mean(df, groupby_cols, col):
    print('in calc_cum_mean')
    cum_mean = df.groupby(groupby_cols)[col].apply(lambda x: x.expanding().mean().fillna(value=0))
    return np.log1p(cum_mean.values)

def encode_cats(df, cat_cols):
    for cat in cat_cols:
        df[cat] = df[cat].astype('category')   

def test_split(df):
    df_test = df[pd.isnull(df['inventory_requested'])]
    results_df = df_test[['retailer', 'productsku', 'calendar_year',
                          'calendar_month', 'mtd_consumption']]
    return df_test, results_df

# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.

def run(input_data):
    try:
        distributor_data = json.loads(input_data)['dist_data']
        print('distributor_data loaded')
        distributor_data = pd.DataFrame(distributor_data, columns=['productsku', 'calendar_year', 'calendar_month', 'max_allocation', 'inventory_requested', 'mtd_consumption', 'retailer'])
        print('distributor_data df')

        product_data = json.loads(input_data)['product_data']
        print('product_data loaded')
        product_data = pd.DataFrame(product_data, columns=['Product', 'Color', 'Collection', 'Style', 'Season', 'Demographic',
       'Fit', 'Material'])
        print('product_data df')

        retailer_data = json.loads(input_data)['retailer_data']
        print('retailer_data loaded')
        retailer_data = pd.DataFrame(retailer_data, columns=['RetailID', 'Retailer', 'Type'])
        print('retailer_data df')

        retailer3_data = json.loads(input_data)['retailer3_data']
        print('retailer3_data loaded')
        retailer3_data = pd.DataFrame(retailer3_data, columns=['productsku', 'calendar_year', 'calendar_month', 'inventory', 'sales',
       'retailer'])
        print('retailer3_data df')

        retailer2_data = json.loads(input_data)['retailer2_data']
        print('retailer2_data loaded')
        retailer2_data = pd.DataFrame(retailer2_data, columns=['productsku', 'calendar_year', 'calendar_month', 'inventory', 'sales',
       'retailer'])
        print('retailer2_data df')

        retailer1_data = json.loads(input_data)['retailer1_data']
        print('retailer1_data loaded')
        retailer1_data = pd.DataFrame(retailer1_data, columns=['productsku', 'calendar_year', 'calendar_month', 'inventory', 'sales',
       'retailer'])
        print('retailer1_data df')

        master_dist = distributor_data.merge(product_data, how='left', left_on='productsku', right_on='Product').drop('Product', axis=1)
        master_dist = master_dist.merge(retailer_data, how='left', left_on='retailer', right_on='RetailID').drop('RetailID', axis=1)

        master_dist = master_dist[['productsku', 'retailer', 'Retailer', 'Type', 'calendar_year', 'calendar_month', 'max_allocation',
                           'inventory_requested', 'mtd_consumption',  'Color',
                           'Collection', 'Style', 'Season', 'Demographic', 'Fit', 'Material',
                           ]]
        retail_master = pd.concat([retailer2_data, retailer3_data, retailer1_data])
        retail_master['previous_mo'] = retail_master['calendar_month'] - 1
        master_dist['previous_mo'] = master_dist['calendar_month'] - 1
        
        master_dist = master_dist.merge(retail_master, left_on=['productsku', 'retailer', 'calendar_year', 'previous_mo'], 
                                        right_on=['productsku', 'retailer', 'calendar_year', 'previous_mo'])
        print('master_dist merged')

        master_dist = master_dist[['productsku', 'retailer', 'Retailer', 'Type', 'calendar_year',
                                'calendar_month_x', 'inventory', 'sales', 'max_allocation', 'inventory_requested',
                                'mtd_consumption', 'Color', 'Collection', 'Style', 'Season',
                                'Demographic', 'Fit', 'Material']]
        master_dist = master_dist.rename(columns={'calendar_month_x': 'calendar_month', 'inventory': 'last_mo_inventory', 'sales': 'last_mo_sales'})
        print('master_dist renamed columns')

        data = create_lags_year(df=master_dist, lag_variables=['mtd_consumption'])
        data['Last_Mo_Sales_Avg'] = calc_cum_mean(data, ['productsku', 'calendar_year', 'calendar_month'], 'last_mo_sales')
        model_data = data[['productsku', 'retailer', 'Retailer', 'Type', 'calendar_year',
                        'calendar_month', 'last_mo_inventory', 'last_mo_sales', 'Last_Mo_Sales_Avg',
                        'mtd_consumption_lag_1',
                        'max_allocation', 'inventory_requested', 'mtd_consumption', 'Color',
                        'Collection', 'Style', 'Season', 'Demographic', 'Fit', 'Material']]
        cat_cols = ['productsku', 'retailer', 'Type', 'calendar_year',
                    'calendar_month', 'Color',
                    'Collection', 'Style', 'Season', 'Demographic', 'Fit', 'Material']
        log_cols = ['last_mo_inventory', 'last_mo_sales',
                    'mtd_consumption', 'max_allocation']
        for x in log_cols:
            model_data[x] = np.log1p(model_data[x])
        encode_cats(model_data, cat_cols)
        print('encoded_cat columns')
        test, results = test_split(model_data)
        print('data split into test and results')
        print('test data')
        print('results data')

        del test['mtd_consumption']
        del test['inventory_requested']
        del test['Retailer']
        print('deleted 3 columns from test data')
        print('test data')

        print('predicting....')
        y_pred = np.expm1(model.predict(test))
        y_pred = [0 if i < 0 else i for i in y_pred]
        y_pred = [max_allocation if yp >= max_allocation else yp for yp, max_allocation in zip(y_pred, np.expm1(test['max_allocation']))]
        results['Prediction'] = y_pred
        print("assigned results['Prediction']")
        del results['mtd_consumption']
        print('deleted mtd_consumption from results')

        print('about to json.dumps()')
        return json.dumps({"result": results.values.tolist()})
    except Exception as e:
        error_str = str(e)
        # return error message back to the client
        return json.dumps({"error": error_str})