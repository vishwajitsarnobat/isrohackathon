import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, LSTM, Bidirectional, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xarray as xr
import netCDF4 as nc

'''
accd contains attributes, go to google and type accd
**cf contains the climate and forecast conventions

'''
# Load your radar data
def read_radar_data(file_path):
    dataset = nc.Dataset(file_path, 'r')
    print('Type of dataset', type(dataset))
    
    # Dictionary to store all data
    data = {}

    for var_name in dataset.variables:
        data[var_name] = dataset.variables[var_name][:]
        print(f'size of {data[var_name]}', len(data[var_name]))

    dataset.close()

    return data

def mask_fill_value_read_data(file_path):
    # Open the NetCDF file
    dataset = nc.Dataset(file_path, 'r')

    # Create a dictionary to store all data
    data = {}

    # Loop through all variables in the dataset
    for var_name in dataset.variables:
        variable = dataset.variables[var_name]
        
        # Check for missing or fill values
        if hasattr(variable, '_FillValue'):
            fill_value = variable._FillValue
            var_data = variable[:]
            var_data = np.ma.masked_equal(var_data, fill_value)
        elif hasattr(variable, 'missing_value'):
            missing_value = variable.missing_value
            var_data = variable[:]
            var_data = np.ma.masked_equal(var_data, missing_value)
        else:
            var_data = variable[:]
        
        data[var_name] = var_data
    
    # Close the NetCDF file
    dataset.close()

    return data

# Convert to Cartesian coordinates
def convert_to_cartesian(data):
    # Implement conversion logic here
    return cartesian_data

# Normalize data
def normalize(data):
    normalized_data = (data - np.mean(data)) / np.std(data)
    return normalized_data

# Temporal alignment
def align_temporally(data):
    # Implement temporal alignment here
    return aligned_data

# Split the data into training, validation, and testing sets
def split_data(data, labels, test_size=0.2, val_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Data augmentation
def augment_data(data):
    # Implement augmentation logic here
    return augmented_data

def build_model():
    model = Sequential()
    
    # 2D CNN for spatial features
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    
    # 1D CNN for temporal features
    model.add(Conv1D(64, 3, activation='relu', input_shape=(timesteps, features)))
    
    # Bidirectional LSTM
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    
    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    
    return mae, rmse

def k_fold_cross_validation(data, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    maes = []
    rmses = []
    
    for train_index, val_index in kf.split(data):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        
        model = build_model()
        train_model(model, X_train, y_train, X_val, y_val)
        
        mae, rmse = evaluate_model(model, X_val, y_val)
        maes.append(mae)
        rmses.append(rmse)
    
    print(f'Average MAE: {np.mean(maes)}')
    print(f'Average RMSE: {np.mean(rmses)}')




file_path = '/usr/src/app/RCTLS_01JUL2024_000543_L2C_STD.nc'
ds = xr.open_dataset(file_path)

# Specify the variable name
variable_name = 'DBZ'

# Check if the variable exists in the dataset
if variable_name in ds:
    # Access the variable and convert to NumPy array
    variable_data = ds[variable_name].values
    
    # Compute and print the maximum value
    max_value = np.max(variable_data)
    print(f"Maximum value of variable '{variable_name}': {max_value}")
else:
    print(f"Variable '{variable_name}' not found in the dataset.")
