# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:30:51 2025

@author: EPGE903150
"""

# Import libraries
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import statistics as st
# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from nfisis.genetic import GEN_NTSK, GEN_NMR
from nfisis.ensemble import R_NTSK, R_NMR

# Definitions
n_runs = 100

Series = ["Alice_91_Site_1A_Trina_Power", "Alice_59_Site_38_QCELLS_Power", "Yulara_5_Site_1_Power", "Yulara_8_Site_5_Power"]

#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "Alice_91_Site_1A_Trina_Power"

# Print DL training information
disp_DL = 0

# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
X = Data[Data.columns[1:13]].values
y = Data[Data.columns[13]].values

# Spliting the data into train and test
n = Data.shape[0]
training_size = round(n*0.6)
validation_size = round(n*0.8)
X_train, X_val, X_test = X[:training_size,:], X[training_size:validation_size,:], X[validation_size:,:]
y_train, y_val, y_test = y[:training_size], y[training_size:validation_size], y[validation_size:]

# Fixing y shape
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# GEN-NMR
# -----------------------------------------------------------------------------


Model_Name = "GEN-NMR"

# Set hyperparameters range
parameters = {'rules':range(1,20, 2), 'fuzzy_operator':["prod","min","max","minmax"],
              'num_generations':[10], 'num_parents_mating':[5], 'sol_per_pop':[10], 'error_metric':["RMSE","MAE","CPPM"], 'parallel_processing':[10]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NMR(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GEN_NMR_params = param
        
l_NRMSE = []
l_NDEI = []
l_MAPE = []
for it in range(n_runs):

    # Initialize the model
    model = GEN_NMR(**best_GEN_NMR_params)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("\nRMSE:", RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test.max() - y_test.min())
    l_NRMSE.append(NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asarray(y_test, dtype=float))
    l_NDEI.append(NDEI)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    l_MAPE.append(MAPE)

GEN_NMR_ = f'{Model_Name} & {st.mean(l_NRMSE):.2f} $\pm$ {st.stdev(l_NRMSE):.2f} & {st.mean(l_NDEI):.2f} $\pm$ {st.stdev(l_NDEI):.2f} & {st.mean(l_MAPE):.2f} $\pm$ {st.stdev(l_MAPE):.2f}'




# -----------------------------------------------------------------------------
# GEN-NTSK-RLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-RLS"

# Set hyperparameters range
parameters = {'rules':[1], 'lambda1':[0.95,0.96,0.97,0.98,0.99], 'adaptive_filter':["RLS"], 'fuzzy_operator':["prod","min","max","minmax"],
              'num_generations':[5], 'num_parents_mating':[5], 'sol_per_pop':[5], 'error_metric':["RMSE","MAE","CPPM"], 'parallel_processing':[10]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GEN_NTSK_RLS_params = param


l_NRMSE = []
l_NDEI = []
l_MAPE = []
for it in range(n_runs):
    
    # Initialize the model
    model = GEN_NTSK(**best_GEN_NTSK_RLS_params)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("\nRMSE:", RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test.max() - y_test.min())
    l_NRMSE.append(NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asarray(y_test, dtype=float))
    l_NDEI.append(NDEI)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    l_MAPE.append(MAPE)                                                                

GEN_NTSK_RLS_ = f'{Model_Name} & {st.mean(l_NRMSE):.2f} $\pm$ {st.stdev(l_NRMSE):.2f} & {st.mean(l_NDEI):.2f} $\pm$ {st.stdev(l_NDEI):.2f} & {st.mean(l_MAPE):.2f} $\pm$ {st.stdev(l_MAPE):.2f}'

# -----------------------------------------------------------------------------
# GEN-NTSK-wRLS
# -----------------------------------------------------------------------------

Model_Name = "GEN-NTSK-wRLS"

# Set hyperparameters range
parameters = {'rules':range(1,20,2), 'adaptive_filter':["wRLS"], 'fuzzy_operator':["prod","min","max","minmax"],
              'num_generations':[5], 'num_parents_mating':[5], 'sol_per_pop':[5], 'error_metric':["RMSE","MAE","CPPM"], 'parallel_processing':[10]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = GEN_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_GEN_NTSK_wRLS_params = param

l_NRMSE = []
l_NDEI = []
l_MAPE = []
for it in range(n_runs):
    
    # Initialize the model
    model = GEN_NTSK(**best_GEN_NTSK_wRLS_params)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("\nRMSE:", RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test.max() - y_test.min())
    l_NRMSE.append(NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asarray(y_test, dtype=float))
    l_NDEI.append(NDEI)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    l_MAPE.append(MAPE)


GEN_NTSK_wRLS_ = f'{Model_Name} & {st.mean(l_NRMSE):.2f} $\pm$ {st.stdev(l_NRMSE):.2f} & {st.mean(l_NDEI):.2f} $\pm$ {st.stdev(l_NDEI):.2f} & {st.mean(l_MAPE):.2f} $\pm$ {st.stdev(l_MAPE):.2f}'

# -----------------------------------------------------------------------------
# R-NMR
# -----------------------------------------------------------------------------


Model_Name = "R-NMR"

# Set hyperparameters range
parameters = {'n_estimators':[50], 'combination':["mean","median","weighted_average"]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = R_NMR(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_R_NMR_params = param

l_NRMSE = []
l_NDEI = []
l_MAPE = []
for it in range(n_runs):
    
    # Initialize the model
    model = R_NMR(**best_R_NMR_params)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("\nRMSE:", RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test.max() - y_test.min())
    l_NRMSE.append(NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asarray(y_test, dtype=float))
    l_NDEI.append(NDEI)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    l_MAPE.append(MAPE)

R_NMR_ = f'{Model_Name} & {st.mean(l_NRMSE):.2f} $\pm$ {st.stdev(l_NRMSE):.2f} & {st.mean(l_NDEI):.2f} $\pm$ {st.stdev(l_NDEI):.2f} & {st.mean(l_MAPE):.2f} $\pm$ {st.stdev(l_MAPE):.2f}'


# -----------------------------------------------------------------------------
# R-NTSK
# -----------------------------------------------------------------------------


Model_Name = "R-NTSK"

# Set hyperparameters range
parameters = {'n_estimators':[50], 'combination':["mean","median","weighted_average"]}

grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    #print(param)

    # Optimize parameters
    model = R_NTSK(**param)
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_R_NTSK_params = param

l_NRMSE = []
l_NDEI = []
l_MAPE = []
for it in range(n_runs):
    
    # Initialize the model
    model = R_NTSK(**best_R_NTSK_params)
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("\nRMSE:", RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test.max() - y_test.min())
    l_NRMSE.append(NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(np.asarray(y_test, dtype=float))
    l_NDEI.append(NDEI)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    l_MAPE.append(MAPE)

R_NTSK_ = f'{Model_Name} & {st.mean(l_NRMSE):.2f} $\pm$ {st.stdev(l_NRMSE):.2f} & {st.mean(l_NDEI):.2f} $\pm$ {st.stdev(l_NDEI):.2f} & {st.mean(l_MAPE):.2f} $\pm$ {st.stdev(l_MAPE):.2f}' 
    

#-----------------------------------------------------------------------------
# Print results
#-----------------------------------------------------------------------------


GEN_NMR_
GEN_NTSK_RLS_
GEN_NTSK_wRLS_
R_NMR_
R_NTSK_