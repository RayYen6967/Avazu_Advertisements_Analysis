import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Read data
df_all = pd.read_csv("train.csv")

#Down sampling
df_0 = df_all[df_all.click == 0].sample(n=6865066, random_state=123)

#Get the balanced dataset
df_down = pd.concat([df_0, df_all[df_all.click == 1]])

#Partition the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df_down.drop('click', axis=1), df_down['click'], test_size=0.2, random_state=123)
df_train = pd.concat([y_train, X_train], axis=1)
df_test = pd.concat([y_test, X_test], axis=1)

#Function for mapping time series data (By daytime intervals)
def hour_imputer2(hour):
    if str(hour)[-2:] == '00' or str(hour)[-2:] == '01':
        return '00-01'
    elif str(hour)[-2:] == '02' or str(hour)[-2:] == '03':
        return '02-03'
    elif str(hour)[-2:] == '04' or str(hour)[-2:] == '05':
        return '04-05'
    elif str(hour)[-2:] == '06' or str(hour)[-2:] == '07':
        return '06-07'
    elif str(hour)[-2:] == '08' or str(hour)[-2:] == '09':
        return '08-09'
    elif str(hour)[-2:] == '10' or str(hour)[-2:] == '11':
        return '10-11'
    elif str(hour)[-2:] == '12' or str(hour)[-2:] == '13':
        return '12-13'
    elif str(hour)[-2:] == '14' or str(hour)[-2:] == '15':
        return '14-15'
    elif str(hour)[-2:] == '16' or str(hour)[-2:] == '17':
        return '16-17'
    elif str(hour)[-2:] == '18' or str(hour)[-2:] == '19':
        return '18-19'
    elif str(hour)[-2:] == '20' or str(hour)[-2:] == '21':
        return '20-21'
    else:
        return '22-23'

# Function for generating date from feature hour
def get_date(hour):
    y = '20' + str(hour)[:2]
    m = str(hour)[2:4]
    d = str(hour)[4:6]
    return y + '-' + m + '-' + d

#Get the mean click rate for each category for features with too many categories
#Get the mean click rate for each category for features with too many categories
df_site_id = df_all[['site_id', 'click']].groupby('site_id').mean()['click']
df_site_domain = df_all[['site_domain', 'click']].groupby('site_domain').mean()['click']
df_app_id = df_all[['app_id', 'click']].groupby('app_id').mean()['click']
df_app_domain = df_all[['app_domain', 'click']].groupby('app_domain').mean()['click']
df_site_category = df_all[['site_category', 'click']].groupby('site_category').mean()['click']
df_app_category = df_all[['app_category', 'click']].groupby('app_category').mean()['click']
df_device_model = df_all[['device_model', 'click']].groupby('device_model').mean()['click']

#Get the overall click rate
df_mean = df_all['click'].mean()

# Functions for mapping categories with corresponding mean click rate
def mapper1(X):
    if X not in df_site_id.index:
        return df_mean
    else:
        return df_site_id[df_site_id.index == X].values[0]
def mapper2(X):
    if X not in df_site_domain.index:
        return df_mean
    else:
        return df_site_domain[df_site_domain.index == X].values[0]
def mapper3(X):
    if X not in df_app_id.index:
        return df_mean
    else:
        return df_app_id[df_app_id.index == X].values[0]
def mapper4(X):
    if X not in df_app_domain.index:
        return df_mean
    else:
        return df_app_domain[df_app_domain.index == X].values[0]
def mapper5(X):
    if X not in df_site_category.index:
        return df_mean
    else:
        return df_site_category[df_site_category.index == X].values[0]
def mapper6(X):
    if X not in df_app_category.index:
        return df_mean
    else:
        return df_app_category[df_app_category.index == X].values[0]
def mapper7(X):
    if X not in df_device_model.index:
        return df_mean
    else:
        return df_device_model[df_device_model.index == X].values[0]

target_list = ['site_id', 'site_domain', 'app_id', 'app_domain', 'site_category',
               'app_category', 'device_model']
mapper_list = [mapper1, mapper2, mapper3, mapper4, mapper5, mapper6,
               mapper7]

#Function for re-categorize data points by their mean click rates
def ranker(rate):
    if rate <= df_mean - 0.04:
        return "very low"
    elif rate <= df_mean - 0.02:
        return "low"
    elif rate <= df_mean + 0.02:
        return "middle"
    elif rate <= df_mean + 0.04:
        return "high"
    else:
        return "very high"

print('First part done')

for x in range(125):
    #Random subset a sample from train or test data 
    df = df_train.sample(n=100000, random_state=(x+1))
    #df = df_test.sample(n=100000, random_state=1)

    # Drop features with only one category
    df.drop(['C14', 'C17', 'C19', 'C20', 'C21'], axis=1, inplace=True)
    
    #Drop unuseful feature (All or most unique categories)
    df.drop('id', inplace=True, axis=1)
    df.drop('device_id', inplace=True, axis=1)
    df.drop('device_ip', inplace=True, axis=1)

    # Create new feature weekday
    df['weekday'] = pd.to_datetime(df.hour.apply(get_date)).dt.dayofweek.astype(str)

    #Map time series data
    df['hour'] = df['hour'].apply(hour_imputer2)

    #Convert categorical data to correct type
    cat_list = ['C1', 'banner_pos', 'site_id', 'site_domain',
           'site_category', 'app_id', 'app_domain', 'app_category', 
           'device_model', 'device_type', 'device_conn_type',
           'C15', 'C16', 'C18']
    for col in cat_list:
        df[col] = df[col].astype(str, copy=False)

     #Map categories with corresponding mean click rate
    for col, mapper in zip(target_list, mapper_list):
        df[col] = df[col].apply(mapper)
    
    #Re-categorize data points by their mean click rates 
    for col in target_list:
        df[col] = df[col].apply(ranker)

    path = str("df_train_%s.csv" % (x+1))
    df.to_csv(path, index=False)
    print(x+1)
