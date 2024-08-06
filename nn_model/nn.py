import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Bidirectional, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.measure import label, regionprops

# Preprocessing
def extract_variables(file_path, variable):
    ds = xr.open_dataset(file_path)
    var = ds[variable].values
    ds.close()
    return var

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def identify_precipitating_systems(reflectivity, threshold=20, min_area=50):
    precipitating_systems = []
    for t in range(reflectivity.shape[0]):
        mask = reflectivity[t] > threshold
        if np.sum(mask) > min_area:
            precipitating_systems.append(mask)
    return precipitating_systems

def track_precipitation_systems(reflectivity, threshold=20):
    tracks = []
    for t in range(reflectivity.shape[0]):
        labeled_array, _ = label(reflectivity[t] > threshold, return_num=True)
        regions = regionprops(labeled_array)
        system_tracks = [{'centroid': region.centroid, 'area': region.area} for region in regions]
        tracks.append(system_tracks)
    return tracks

file_path = '/usr/src/app/RCTLS_01JUL2024_000543_L2C_STD.nc'
file_path = '/home/vishwajitsarnobat/workspace/isrohackathon/RCTLS_01JUL2024_000543_L2C_STD.nc' # for my local
radial_velocity = extract_variables(file_path, 'VEL')
reflectivity = extract_variables(file_path, 'DBZ')

normalized_velocity = normalize(radial_velocity)
normalized_reflectivity = normalize(reflectivity)

# Identify and track precipitation systems
precipitating_systems = identify_precipitating_systems(normalized_reflectivity)
tracks = track_precipitation_systems(normalized_reflectivity)

# Prepare data for prediction
def prepare_data_for_prediction(reflectivity, sequence_length=10):
    X, y = [], []
    for t in range(sequence_length, reflectivity.shape[0]):
        X.append(reflectivity[t-sequence_length:t])
        y.append(reflectivity[t])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = prepare_data_for_prediction(normalized_reflectivity, sequence_length)

# Building the Model
def create_spatial_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), activation='relu')(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu')(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_temporal_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, activation='relu')(inputs)
    x = Conv1D(64, 3, activation='relu')(x)
    x = Flatten()(x)
    return Model(inputs, x)

def create_radarcast_net(spatial_input_shape, temporal_input_shape):
    spatial_model = create_spatial_model(spatial_input_shape)
    temporal_model = create_temporal_model(temporal_input_shape)
    
    combined_input = Concatenate()([spatial_model.output, temporal_model.output])
    
    x = Bidirectional(LSTM(64, return_sequences=True))(combined_input)
    x = Bidirectional(LSTM(32))(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[spatial_model.input, temporal_model.input], outputs=output)
    return model

spatial_input_shape = (sequence_length, 81, 481, 481, 1)  # Adjust according to data
temporal_input_shape = (sequence_length, 1)  # Adjust according to data

model = create_radarcast_net(spatial_input_shape, temporal_input_shape)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Training the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('radarcast_net_best_model.h5', save_best_only=True)

history = model.fit(
    [X[..., np.newaxis], y[..., np.newaxis]],  # Adding a new axis for channel dimension
    y,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluating the Model
test_loss, test_mae = model.evaluate([X[..., np.newaxis], y[..., np.newaxis]], y)

# Predict Future Path
def predict_future_path(model, recent_data):
    return model.predict(recent_data)

recent_data = X[-sequence_length:]  # Example data with the last sequence length
predicted_path = predict_future_path(model, np.expand_dims(recent_data, axis=0))

print(predicted_path)



