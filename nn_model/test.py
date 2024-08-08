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
    
    for file_path, _ in file_info:
        radial_velocity = extract_variables(file_path, 'VEL')
        reflectivity = extract_variables(file_path, 'DBZ')

        standardized_velocity = standardize(radial_velocity)
        standardized_reflectivity = standardize(reflectivity)
        
        velocity_list.append(standardized_velocity)
        reflectivity_list.append(standardized_reflectivity)

    return np.array(velocity_list), np.array(reflectivity_list), np.array(timestamps)

def create_sequences(velocity_data, reflectivity_data, timestamps, seq_length=3):
    X_velocity, X_reflectivity, y_velocity, y_reflectivity, time_diffs = [], [], [], [], []

    timestamps_in_seconds = np.array([ts.astype(np.int64) / 1e9 for ts in timestamps])
    
    for i in range(len(velocity_data) - seq_length):
        seq_X_velocity = velocity_data[i:i+seq_length]
        seq_X_reflectivity = reflectivity_data[i:i+seq_length]
        seq_y_velocity = velocity_data[i + seq_length]
        seq_y_reflectivity = reflectivity_data[i + seq_length]

        time_diff = (timestamps_in_seconds[i + seq_length] - timestamps_in_seconds[i]).astype(np.float32)
        
        X_velocity.append(seq_X_velocity)
        X_reflectivity.append(seq_X_reflectivity)
        y_velocity.append(seq_y_velocity)
        y_reflectivity.append(seq_y_reflectivity)
        time_diffs.append(time_diff)
    
    X_velocity = np.array(X_velocity)
    X_reflectivity = np.array(X_reflectivity)
    y_velocity = np.array(y_velocity)
    y_reflectivity = np.array(y_reflectivity)
    time_diffs = np.array(time_diffs, dtype=np.float32)  

    X_velocity = X_velocity[..., np.newaxis]  # Adding the channel dimension
    X_reflectivity = X_reflectivity[..., np.newaxis]  # Adding the channel dimension
    y_velocity = y_velocity[..., np.newaxis]  # Adding the channel dimension
    y_reflectivity = y_reflectivity[..., np.newaxis]  # Adding the channel dimension
    time_diffs = time_diffs.reshape(-1, 1)  
    
    return X_velocity, X_reflectivity, y_velocity, y_reflectivity, time_diffs

def create_tf_dataset(X_velocity, X_reflectivity, y_velocity, y_reflectivity, time_diffs, batch_size=32, shuffle=True):
    def generator():
        for i in range(len(X_velocity)):
            yield (
                {
                    'velocity_spatial_input': X_velocity[i][:5],  # Ensure this is (5, 5, 5, 1)
                    'velocity_temporal_input': X_velocity[i],  # Ensure this is (3, 5, 5, 1)
                    'reflectivity_spatial_input': X_reflectivity[i][:5],  # Ensure this is (5, 5, 5, 1)
                    'reflectivity_temporal_input': X_reflectivity[i],  # Ensure this is (3, 5, 5, 1)
                    'time_diff': time_diffs[i]  # Ensure this is (1,)
                },
                {
                    'velocity_output': y_velocity[i],  # Ensure this is (5, 5, 5, 1)
                    'reflectivity_output': y_reflectivity[i]  # Ensure this is (5, 5, 5, 1)
                }
            )

    output_signature = (
        {
            'velocity_spatial_input': tf.TensorSpec(shape=(5, 5, 5, 1), dtype=tf.float32),
            'velocity_temporal_input': tf.TensorSpec(shape=(3, 5, 5, 1), dtype=tf.float32),
            'reflectivity_spatial_input': tf.TensorSpec(shape=(5, 5, 5, 1), dtype=tf.float32),
            'reflectivity_temporal_input': tf.TensorSpec(shape=(3, 5, 5, 1), dtype=tf.float32),
            'time_diff': tf.TensorSpec(shape=(1,), dtype=tf.float32),
        },
        {
            'velocity_output': tf.TensorSpec(shape=(5, 5, 5, 1), dtype=tf.float32),
            'reflectivity_output': tf.TensorSpec(shape=(5, 5, 5, 1), dtype=tf.float32),
        }
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X_velocity))
    
    dataset = dataset.batch(batch_size)
    return dataset


# Model building functions
def create_spatial_model(input_shape, name_prefix):
    inputs = Input(shape=input_shape, name=f'{name_prefix}_spatial_input')  # Ensure input_shape is (5, 5, 5, 1)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_temporal_model(input_shape, name_prefix):
    inputs = Input(shape=input_shape, name=f'{name_prefix}_temporal_input')  # Ensure input_shape is (seq_length, 5, 5, 1)
    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_radarcast_net(spatial_input_shape, temporal_input_shape):
    # Velocity models
    spatial_model_velocity = create_spatial_model(spatial_input_shape, name_prefix='velocity')
    temporal_model_velocity = create_temporal_model(temporal_input_shape, name_prefix='velocity')
    
    # Reflectivity models
    spatial_model_reflectivity = create_spatial_model(spatial_input_shape, name_prefix='reflectivity')
    temporal_model_reflectivity = create_temporal_model(temporal_input_shape, name_prefix='reflectivity')
    
    # Combine inputs
    combined_input_velocity = Concatenate()([spatial_model_velocity.output, temporal_model_velocity.output])
    combined_input_reflectivity = Concatenate()([spatial_model_reflectivity.output, temporal_model_reflectivity.output])
    
    combined = Concatenate()([combined_input_velocity, combined_input_reflectivity])
    
    # Add the time_diff input
    time_diff_input = Input(shape=(1,), name='time_diff')  # Time difference input
    
    # Fully connected layers
    x = Concatenate()([combined, time_diff_input])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layers
    output_velocity = Dense(np.prod(spatial_input_shape[:-1]), activation='linear', name='velocity_output')(x)
    output_velocity = Reshape(spatial_input_shape[:-1])(output_velocity)
    
    output_reflectivity = Dense(np.prod(spatial_input_shape[:-1]), activation='linear', name='reflectivity_output')(x)
    output_reflectivity = Reshape(spatial_input_shape[:-1])(output_reflectivity)
    
    model = Model(inputs=[spatial_model_velocity.input, temporal_model_velocity.input, 
                          spatial_model_reflectivity.input, temporal_model_reflectivity.input, 
                          time_diff_input], 
                  outputs=[output_velocity, output_reflectivity])
    
    return model


# Main script
folder_path = '/home/vishwajitsarnobat/Downloads/isro_hackathon_data'
velocity_data, reflectivity_data, timestamps = load_and_preprocess_data(folder_path)

seq_length = 3 
X_velocity, X_reflectivity, y_velocity, y_reflectivity, time_diffs = create_sequences(velocity_data, reflectivity_data, timestamps, seq_length=seq_length)

train_size = int(0.7 * len(X_velocity))
val_size = int(0.15 * len(X_velocity))

# dividing into sets
train_X_velocity = X_velocity[:train_size]
val_X_velocity = X_velocity[train_size:train_size + val_size]
test_X_velocity = X_velocity[train_size + val_size:]

train_X_reflectivity = X_reflectivity[:train_size]
val_X_reflectivity = X_reflectivity[train_size:train_size + val_size]
test_X_reflectivity = X_reflectivity[train_size + val_size:]

train_y_velocity = y_velocity[:train_size]
val_y_velocity = y_velocity[train_size:train_size + val_size]
test_y_velocity = y_velocity[train_size + val_size:]

train_y_reflectivity = y_reflectivity[:train_size]
val_y_reflectivity = y_reflectivity[train_size:train_size + val_size]
test_y_reflectivity = y_reflectivity[train_size + val_size:]

train_time_diffs = time_diffs[:train_size]
val_time_diffs = time_diffs[train_size:train_size + val_size]
test_time_diffs = time_diffs[train_size + val_size:]

spatial_input_shape = (5, 5, 5, 1)
temporal_input_shape = (seq_length, 5, 5, 1)

batch_size = 4  # Adjust as needed

# Create the TensorFlow datasets
train_dataset = create_tf_dataset(
    train_X_velocity, train_X_reflectivity, train_y_velocity, train_y_reflectivity, train_time_diffs, batch_size=batch_size
)
val_dataset = create_tf_dataset(
    val_X_velocity, val_X_reflectivity, val_y_velocity, val_y_reflectivity, val_time_diffs, batch_size=batch_size
)
test_dataset = create_tf_dataset(
    test_X_velocity, test_X_reflectivity, test_y_velocity, test_y_reflectivity, test_time_diffs, batch_size=batch_size
)

# Create the model with correct input shapes
model = create_radarcast_net(spatial_input_shape, temporal_input_shape)

model.summary()

# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('radarcast_net.keras', monitor='val_loss', save_best_only=True)

# Compile the model with correct output layer names
model.compile(
    optimizer='adam', 
    loss={
        'velocity_output': 'mse',
        'reflectivity_output': 'mse'
    },
    metrics={
        'velocity_output': 'mae',
        'reflectivity_output': 'mae'
    }
)

# Train the model
model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=100, 
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model on the test set
test_loss, test_velocity_mae, test_reflectivity_mae = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test MAE (Velocity): {test_velocity_mae}, Test MAE (Reflectivity): {test_reflectivity_mae}')
