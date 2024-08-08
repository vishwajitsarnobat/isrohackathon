import xarray as xr
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, ConvLSTM2D, BatchNormalization, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence

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

def load_and_preprocess_data(folder_path):
    velocity_list = []
    reflectivity_list = []
    timestamps = []
    
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.nc'):
            file_path = os.path.join(folder_path, file_name)
            radial_velocity = extract_variables(file_path, 'VEL')
            reflectivity = extract_variables(file_path, 'DBZ')
            
            standardized_velocity = standardize(radial_velocity)
            standardized_reflectivity = standardize(reflectivity)
            
            velocity_list.append(standardized_velocity)
            reflectivity_list.append(standardized_reflectivity)
            
            ds = xr.open_dataset(file_path)
            timestamps.append(ds['time'].values[0])
            ds.close()
    
    return np.array(velocity_list), np.array(reflectivity_list), np.array(timestamps)

def create_sequences(velocity_data, reflectivity_data, seq_length=3):
    X, y_velocity, y_reflectivity = [], [], []
    
    for i in range(len(velocity_data) - seq_length):
        X.append(velocity_data[i:i + seq_length - 1])
        y_velocity.append(velocity_data[i + seq_length - 1])
        y_reflectivity.append(reflectivity_data[i + seq_length - 1])
    
    return np.array(X), np.array(y_velocity), np.array(y_reflectivity)

# Data generator class
class DataGenerator(Sequence):
    def __init__(self, X_data, y_velocity, y_reflectivity, batch_size=32, shuffle=True):
        self.X_data = X_data
        self.y_velocity = y_velocity
        self.y_reflectivity = y_reflectivity
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X_data[indexes]
        batch_y_velocity = self.y_velocity[indexes]
        batch_y_reflectivity = self.y_reflectivity[indexes]
        return batch_X, [batch_y_velocity, batch_y_reflectivity]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Model building functions
def create_spatial_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_temporal_model(input_shape):
    inputs = Input(shape=input_shape)
    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_radarcast_net(spatial_input_shape, temporal_input_shape):
    spatial_model = create_spatial_model(spatial_input_shape)
    temporal_model = create_temporal_model(temporal_input_shape)
    
    combined_input = Concatenate()([spatial_model.output, temporal_model.output])
    
    x = Dense(128, activation='relu')(combined_input)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_velocity = Dense(1, activation='linear', name='velocity_output')(x)
    output_reflectivity = Dense(1, activation='linear', name='reflectivity_output')(x)
    
    model = Model(inputs=[spatial_model.input, temporal_model.input], outputs=[output_velocity, output_reflectivity])
    return model

# Loading and preprocessing data
folder_path = '/home/vishwajitsarnobat/Downloads/isro_hackathon_data'
velocity_data, reflectivity_data, timestamps = load_and_preprocess_data(folder_path)

# Create sequences for training
seq_length = 4
X_data, y_velocity, y_reflectivity = create_sequences(velocity_data, reflectivity_data, seq_length)

# Split data into training and testing sets based on files
train_size = int(0.7 * len(X_data))
val_size = int(0.15 * len(X_data))

train_X = X_data[:train_size]
val_X = X_data[train_size:train_size + val_size]
test_X = X_data[train_size + val_size:]

train_y_velocity = y_velocity[:train_size]
val_y_velocity = y_velocity[train_size:train_size + val_size]
test_y_velocity = y_velocity[train_size + val_size:]

train_y_reflectivity = y_reflectivity[:train_size]
val_y_reflectivity = y_reflectivity[train_size:train_size + val_size]
test_y_reflectivity = y_reflectivity[train_size + val_size:]

# Build and compile the model
spatial_input_shape = (81, 481, 481, 1)  # Height, Width, Depth, Channels
temporal_input_shape = (seq_length - 1, 481, 481, 1)  # Sequence, Height, Width, Channels

model = create_radarcast_net(spatial_input_shape, temporal_input_shape)
model.compile(optimizer='adam', loss={'velocity_output': 'mean_squared_error', 'reflectivity_output': 'mean_squared_error'},
              metrics={'velocity_output': 'mae', 'reflectivity_output': 'mae'})

# Create data generators
train_generator = DataGenerator(train_X, train_y_velocity, train_y_reflectivity, batch_size=32)
val_generator = DataGenerator(val_X, val_y_velocity, val_y_reflectivity, batch_size=32, shuffle=False)

# Training the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('radarcast_net_best_model.h5', save_best_only=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluating the model
test_X_sequences = test_X
test_y_velocity = test_y_velocity
test_y_reflectivity = test_y_reflectivity

test_predictions = model.predict(test_X_sequences)
test_r2_velocity = r2_score(test_y_velocity, test_predictions[0])
test_r2_reflectivity = r2_score(test_y_reflectivity, test_predictions[1])

print(f"R2 Score for Velocity: {test_r2_velocity:.4f}")
print(f"R2 Score for Reflectivity: {test_r2_reflectivity:.4f}")
