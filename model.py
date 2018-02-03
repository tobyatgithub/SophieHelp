#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 22:18:26 2018

@author: sophie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn import preprocessing


# add four variables: 'artist_age', 'auction_year', 
# 'auction_year minus artist_death_year', 'whehter artist is alive'
def FeatureEngineering(df):
    df['artist_death_year'] = df['artist_death_year'].replace(np.NaN, 2018)
    df['artist_age'] = df['artist_death_year'] - df['artist_birth_year']
    df['auction_year'] = pd.to_datetime(df['auction_date']).dt.year
    df['auction_year-artist_death_year'] = df['auction_year'] - df['artist_death_year']
    df['alive_indicator'] = df['artist_death_year'].map(lambda x: 1 if (x == 2018) else 0)
    return df

def traintestsplit(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop('hammer_price', axis = 1), dataset['hammer_price'], test_size = 0.33, random_state = 6)
    return X_train, X_test, y_train, y_test

def LinearRegressionModel(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_test_pred = regr.predict(X_test)
    coef = regr.coef_
    mse = mean_squared_error(y_test, y_test_pred)
    squareroot_mse = np.sqrt(mse)
    return squareroot_mse, coef

def predict(test_csv):
    # Read Data
    df = pd.read_csv(test_csv, encoding = 'latin-1')
    
    # Drop NaN in estimate_high, estimate_low, hammer_price
    df_drop_nan = df[df['hammer_price'].notnull() & df['estimate_high'].notnull() & df['estimate_low'].notnull()]
    
    # Drop -1 in hammer_price
    df_drop_nan = df_drop_nan.loc[df_drop_nan['hammer_price'] > 0]

    # Add feature
    df_added_feature = FeatureEngineering(df_drop_nan)
    
    # Remove those non-number column or ones not important
    drop_column_names = ['artist_birth_year',
                         'artist_death_year',
                         'artist_name',
                         'artist_nationality',
                         'auction_date',
                         'category',
                         'currency',
                         'edition',
                         'location',
                         'materials',
                         'title',
                         'year_of_execution',
                         'measurement_depth_cm',
                         'measurement_height_cm', 
                         'measurement_width_cm',]
    df_2 = df_added_feature.drop(drop_column_names, axis = 1)
    
    # Save column names
    names = list(df_2.columns)
    # Normalize each columns
    min_max_scaler = preprocessing.MinMaxScaler()
    df_drop_nan_scaled = min_max_scaler.fit_transform(df_2.values)
    df_drop_nan = pd.DataFrame(df_drop_nan_scaled)
    df_drop_nan.columns = names

    # Set y_test frame
    y_test = df_drop_nan['hammer_price']        
    
    # Set x_test frame
    df_3 = df_drop_nan.drop('hammer_price',axis=1)

    
    # Read model saved
    regr = joblib.load('filename.pkl') 
    y_test_pred = regr.predict(df_3)
    mse = mean_squared_error(y_test, y_test_pred)
    result = np.sqrt(mse)
    print('RSME = ', result)
    
    # Output prediction into a csv file
    np.savetxt("prediction.csv", y_test_pred, delimiter=",")
    return result
    