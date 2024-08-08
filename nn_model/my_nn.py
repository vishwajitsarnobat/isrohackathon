'''
errors to work on:
make ot atleat work
use time difference as a parameter for accurate prediction
'''


import xarray as xr
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, ConvLSTM2D, BatchNormalization, Dense, Flatten, Concatenate, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence

# Preprocessing functions
def extract_variables(file_path, variable):
    ds = xr.open_dataset(file_path)
    var = ds[variable].values

    # Replace NaN values with zero
    var = np.nan_to_num(var, nan=0.0)

    # Extracting a 5x5x5 grid from the middle
    mid_height = var.shape[1] // 2
    mid_lat = var.shape[2] // 2
    mid_lon = var.shape[3] // 2

    var = var[:, 
              mid_height - 2:mid_height + 3, 
              mid_lat - 2:mid_lat + 3, 
              mid_lon - 2:mid_lon + 3]

    ds.close()
    return var

def standardize(data):
    # Replace NaN values with zero before standardizing
    data = np.nan_to_num(data, nan=0.0)
    scaler = StandardScaler()
    reshaped_data = data.reshape(-1).astype(np.float32)  # Flatten the data
    standardized_data = scaler.fit_transform(reshaped_data[:, np.newaxis]).reshape(data.shape)
    return standardized_data

def load_and_preprocess_data(folder_path):
    velocity_list = []
    reflectivity_list = []
    timestamps = []
    file_info = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.nc'):
            file_path = os.path.join(folder_path, file_name)
            ds = xr.open_dataset(file_path)
            timestamp = ds['time'].values[0]
            timestamps.append(timestamp)
            file_info.append((file_path, timestamp))
            ds.close()

    file_info.sort(key=lambda x: x[1])
    
    for file_path, _ in file_info: # to arrange data in timely order
        radial_velocity = extract_variables(file_path, 'VEL')
        reflectivity = extract_variables(file_path, 'DBZ')

        standardized_velocity = standardize(radial_velocity)
        standardized_reflectivity = standardize(reflectivity)
        
        velocity_list.append(standardized_velocity)
        reflectivity_list.append(standardized_reflectivity)

    return np.array(velocity_list), np.array(reflectivity_list), np.array(timestamps)

def create_sequences(velocity_data, reflectivity_data, seq_length=3):
    X_velocity, X_reflectivity, y_velocity, y_reflectivity = [], [], [], []
    
    for i in range(len(velocity_data) - seq_length):
        seq_X_velocity = velocity_data[i]
        seq_X_reflectivity = reflectivity_data[i]
        seq_y_velocity = velocity_data[i + seq_length]
        seq_y_reflectivity = reflectivity_data[i + seq_length]
        
        X_velocity.append(seq_X_velocity)
        X_reflectivity.append(seq_X_reflectivity)
        y_velocity.append(seq_y_velocity)
        y_reflectivity.append(seq_y_reflectivity)
    
    X_velocity = np.array(X_velocity)
    X_reflectivity = np.array(X_reflectivity)
    y_velocity = np.array(y_velocity)
    y_reflectivity = np.array(y_reflectivity)
    
    # Reshape to match the model's expected input
    X_velocity = X_velocity[..., np.newaxis]  # Adding the channel dimension
    X_reflectivity = X_reflectivity[..., np.newaxis]  # Adding the channel dimension
    y_velocity = y_velocity[..., np.newaxis]  # Adding the channel dimension
    y_reflectivity = y_reflectivity[..., np.newaxis]  # Adding the channel dimension
    
    return X_velocity, X_reflectivity, y_velocity, y_reflectivity

# Data generator class to avoid kind of overfitting using shuffle
class DataGenerator(Sequence):
    def __init__(self, X_velocity, X_reflectivity, y_velocity, y_reflectivity, batch_size=32, shuffle=True):
        self.X_velocity = X_velocity
        self.X_reflectivity = X_reflectivity
        self.y_velocity = y_velocity
        self.y_reflectivity = y_reflectivity
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X_velocity))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X_velocity) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X_velocity = tf.convert_to_tensor(self.X_velocity[indexes], dtype=tf.float32)
        batch_X_reflectivity = tf.convert_to_tensor(self.X_reflectivity[indexes], dtype=tf.float32)
        batch_y_velocity = tf.convert_to_tensor(self.y_velocity[indexes], dtype=tf.float32)
        batch_y_reflectivity = tf.convert_to_tensor(self.y_reflectivity[indexes], dtype=tf.float32)
        return (batch_X_velocity, batch_X_reflectivity), (batch_y_velocity, batch_y_reflectivity)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Model building functions
def create_spatial_model(input_shape):
    inputs = Input(shape=input_shape)  # (5, 5, 5, 1)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_temporal_model(input_shape):
    inputs = Input(shape=input_shape)  # (seq_length, 5, 5, 1)
    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_radarcast_net(spatial_input_shape, temporal_input_shape):
    # Velocity models
    spatial_model_velocity = create_spatial_model(spatial_input_shape)  # (5, 5, 5, 1)
    temporal_model_velocity = create_temporal_model(temporal_input_shape)  # (seq_length, 5, 5, 1)
    
    # Reflectivity models
    spatial_model_reflectivity = create_spatial_model(spatial_input_shape)  # (5, 5, 5, 1)
    temporal_model_reflectivity = create_temporal_model(temporal_input_shape)  # (seq_length, 5, 5, 1)
    
    # Combine inputs
    combined_input_velocity = Concatenate()([spatial_model_velocity.output, temporal_model_velocity.output])
    combined_input_reflectivity = Concatenate()([spatial_model_reflectivity.output, temporal_model_reflectivity.output])
    
    combined = Concatenate()([combined_input_velocity, combined_input_reflectivity])
    
    # Fully connected layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layers
    output_velocity = Dense(np.prod(spatial_input_shape[:-1]), activation='linear', name='velocity_output')(x)
    output_velocity = Reshape(spatial_input_shape[:-1])(output_velocity)
    
    output_reflectivity = Dense(np.prod(spatial_input_shape[:-1]), activation='linear', name='reflectivity_output')(x)
    output_reflectivity = Reshape(spatial_input_shape[:-1])(output_reflectivity)
    
    model = Model(inputs=[spatial_model_velocity.input, temporal_model_velocity.input, 
                          spatial_model_reflectivity.input, temporal_model_reflectivity.input], 
                  outputs=[output_velocity, output_reflectivity])
    
    return model

folder_path = '/home/vishwajitsarnobat/Downloads/isro_hackathon_data'
velocity_data, reflectivity_data, timestamps = load_and_preprocess_data(folder_path) # (28, 1, 5, 5, 5), (28, 1, 5, 5, 5), (28)

seq_length = 3 
X_velocity, X_reflectivity, y_velocity, y_reflectivity = create_sequences(velocity_data, reflectivity_data, seq_length=seq_length) # (25, 1, 5, 5, 5, 1)

train_size = int(0.7 * len(X_velocity))
val_size = int(0.15 * len(X_velocity))

# dividing into sets
# inputs
train_X_velocity = X_velocity[:train_size] # (17, 1, 5, 5, 5, 1)
val_X_velocity = X_velocity[train_size:train_size + val_size] # (3, 1, 5, 5, 5, 1)
test_X_velocity = X_velocity[train_size + val_size:] # (5, 1, 5, 5, 5, 1)

train_X_reflectivity = X_reflectivity[:train_size]
val_X_reflectivity = X_reflectivity[train_size:train_size + val_size]
test_X_reflectivity = X_reflectivity[train_size + val_size:]

# outputs
train_y_velocity = y_velocity[:train_size]
val_y_velocity = y_velocity[train_size:train_size + val_size]
test_y_velocity = y_velocity[train_size + val_size:]

train_y_reflectivity = y_reflectivity[:train_size]
val_y_reflectivity = y_reflectivity[train_size:train_size + val_size]
test_y_reflectivity = y_reflectivity[train_size + val_size:]
print(train_X_velocity.shape, val_X_velocity.shape, test_X_velocity.shape)

# data generation
batch_size = 4 # need to adjust batch size
train_generator = DataGenerator(train_X_velocity, train_X_reflectivity, train_y_velocity, train_y_reflectivity, batch_size=batch_size)
val_generator = DataGenerator(val_X_velocity, val_X_reflectivity, val_y_velocity, val_y_reflectivity, batch_size=batch_size)
test_generator = DataGenerator(test_X_velocity, test_X_reflectivity, test_y_velocity, test_y_reflectivity, batch_size=batch_size)

spatial_input_shape = (5, 5, 5, 1)
temporal_input_shape = (seq_length, 5, 5, 1)
model = create_radarcast_net(spatial_input_shape, temporal_input_shape)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('radarcast_net.keras', monitor='val_loss', save_best_only=True)

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')