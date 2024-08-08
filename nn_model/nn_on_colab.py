from google.colab import files
import xarray as xr
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, ConvLSTM2D, BatchNormalization, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Upload files
uploaded = files.upload()

# Preprocessing functions
def extract_variables(file_path, variable):
    ds = xr.open_dataset(file_path)
    var = ds[variable].values
    ds.close()
    return var

def standardize(data):
    scaler = StandardScaler()
    reshaped_data = data.reshape(-1, 1)
    standardized_data = scaler.fit_transform(reshaped_data).reshape(data.shape)
    return standardized_data

def load_and_preprocess_data(uploaded_files):
    velocity_list = []
    reflectivity_list = []
    timestamps = []
    
    for file_name in uploaded_files.keys():
        file_path = file_name
        radial_velocity = extract_variables(file_path, 'VEL')
        reflectivity = extract_variables(file_path, 'DBZ')
        
        standardized_velocity = standardize(radial_velocity)
        standardized_reflectivity = standardize(reflectivity)
        
        velocity_list.append(standardized_velocity)
        reflectivity_list.append(standardized_reflectivity)
        
        ds = xr.open_dataset(file_path)
        timestamps.append(ds['time'].values)
        ds.close()
    
    return np.array(velocity_list), np.array(reflectivity_list), np.array(timestamps)

# Model building functions
def create_spatial_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_temporal_model(input_shape):
    inputs = Input(shape=input_shape)
    x = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_radarcast_net(spatial_input_shape, temporal_input_shape):
    spatial_model = create_spatial_model(spatial_input_shape)
    temporal_model = create_temporal_model(temporal_input_shape)
    
    combined_input = Concatenate()([spatial_model.output, temporal_model.output])
    
    x = Dense(64, activation='relu')(combined_input)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[spatial_model.input, temporal_model.input], outputs=output)
    return model

# Loading and preprocessing data
velocity_data, reflectivity_data, timestamps = load_and_preprocess_data(uploaded)

# Prepare input shapes
spatial_input_shape = (81, 481, 481, 1)  # Height, Width, Depth, Channels
temporal_input_shape = (None, 481, 481, 1)  # Sequence, Height, Width, Channels

# Split data into training and testing sets
train_size = int(0.7 * len(velocity_data))
val_size = int(0.15 * len(velocity_data))

train_velocity = velocity_data[:train_size]
val_velocity = velocity_data[train_size:train_size + val_size]
test_velocity = velocity_data[train_size + val_size:]

train_reflectivity = reflectivity_data[:train_size]
val_reflectivity = reflectivity_data[train_size:train_size + val_size]
test_reflectivity = reflectivity_data[train_size + val_size:]

# Dummy labels for the example (replace with actual labels if available)
train_labels = np.random.rand(train_velocity.shape[0], 1)
val_labels = np.random.rand(val_velocity.shape[0], 1)
test_labels = np.random.rand(test_velocity.shape[0], 1)

# Reshape data to match the input shape of the models
train_velocity = train_velocity[..., np.newaxis]
val_velocity = val_velocity[..., np.newaxis]
test_velocity = test_velocity[..., np.newaxis]

train_reflectivity = train_reflectivity[..., np.newaxis]
val_reflectivity = val_reflectivity[..., np.newaxis]
test_reflectivity = test_reflectivity[..., np.newaxis]

# Build and compile the model
model = create_radarcast_net(spatial_input_shape, temporal_input_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Training the model with a smaller batch size
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('radarcast_net_best_model.h5', save_best_only=True)

history = model.fit(
    [train_velocity, train_reflectivity],
    train_labels,
    validation_data=([val_velocity, val_reflectivity], val_labels),
    epochs=100,
    batch_size=8,  # Smaller batch size
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluating the model
test_loss, test_mae = model.evaluate([test_velocity, test_reflectivity], test_labels)

# Making predictions
def predict_and_correct(model, initial_data):
    predictions = model.predict(initial_data)
    return predictions

initial_velocity_data = test_velocity[:100]  # Example initial data with added channel dimension
initial_reflectivity_data = test_reflectivity[:100]  # Example initial data with added channel dimension
initial_data = [initial_velocity_data, initial_reflectivity_data]
predictions = predict_and_correct(model, initial_data)

print(predictions)
