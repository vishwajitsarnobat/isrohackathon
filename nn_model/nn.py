import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv1D, Bidirectional, LSTM, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

# Preprocessing
def extract_variables(file_path, variable):
    ds = xr.open_dataset(file_path)
    var = ds[variable].values
    ds.close()
    return var

def standardize(data):
    # Flatten the data for fitting the scaler
    n_samples = data.size
    reshaped_data = data.reshape(n_samples, 1)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(reshaped_data).reshape(data.shape)
    return standardized_data

file_path = 'https://www.mosdac.gov.in/download/?r=/download&path=L09yZGVyL0F1ZzI0XzEwNTYwNy9SQ1RMU18wMUpVTDIwMjRfMDAwNTQzX0wyQ19TVEQubmM%3D'
print(extract_variables(file_path, 'VEL'))
sys.exit()
file_path = '/usr/src/app/RCTLS_01JUL2024_000543_L2C_STD.nc'
file_path = '/home/vishwajitsarnobat/workspace/isrohackathon/RCTLS_01JUL2024_000543_L2C_STD.nc' # for my local

# Extract and standardize data
radial_velocity = extract_variables(file_path, 'VEL')
reflectivity = extract_variables(file_path, 'DBZ')

standardized_velocity = standardize(radial_velocity)  # Standardizing
standardized_reflectivity = standardize(reflectivity)

# Use the entire snapshot for training and evaluation
train_velocity = standardized_velocity
train_reflectivity = standardized_reflectivity

# Dummy labels for the example (replace with actual labels if available)
train_labels = np.random.rand(train_velocity.shape[0], 1)

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

# Adjust the input shapes according to your data
spatial_input_shape = train_velocity.shape[1:] + (1,)  # Adding channel dimension
temporal_input_shape = (train_velocity.shape[1], 1)  # Assuming time dimension

model = create_radarcast_net(spatial_input_shape, temporal_input_shape)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Training the Model
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('radarcast_net_best_model.h5', save_best_only=True)

history = model.fit(
    [train_velocity[..., np.newaxis], train_reflectivity[..., np.newaxis]],  # Adding a new axis for channel dimension
    train_labels,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluating the Model
test_loss, test_mae = model.evaluate([train_velocity[..., np.newaxis], train_reflectivity[..., np.newaxis]], train_labels)

# Predicting and Correcting
def predict_and_correct(model, initial_data):
    predictions = model.predict(initial_data)
    # Logic for correcting predictions every 30 minutes
    return predictions

initial_velocity_data = train_velocity[:100, ..., np.newaxis]  # Example initial data with added channel dimension
initial_reflectivity_data = train_reflectivity[:100, ..., np.newaxis]  # Example initial data with added channel dimension
initial_data = [initial_velocity_data, initial_reflectivity_data]
predictions = predict_and_correct(model, initial_data)

print(predictions)



