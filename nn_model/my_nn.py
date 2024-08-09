import xarray as xr
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, LSTM, Dense, Flatten, Concatenate, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def load_and_preprocess_data(folder_path, grid_size=5):
    data = []
    timestamps = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.nc'):
            file_path = os.path.join(folder_path, file_name)
            with xr.open_dataset(file_path) as ds:
                timestamp = ds['time'].values[0]
                timestamps.append(timestamp)

                reflectivity = ds['DBZ'].values[0]
                velocity = ds['VEL'].values[0]

                # Replace NaN values with the mean of the non-NaN values
                reflectivity = np.nan_to_num(reflectivity, nan=np.nanmean(reflectivity))
                velocity = np.nan_to_num(velocity, nan=np.nanmean(velocity))

                center_h, center_lat, center_lon = [s // 2 for s in reflectivity.shape]
                half_grid = grid_size // 2

                slices = [slice(center - half_grid, center + half_grid + 1) 
                          for center in (center_h, center_lat, center_lon)]

                ref_grid = reflectivity[tuple(slices)]
                vel_grid = velocity[tuple(slices)]

                combined_data = np.stack([ref_grid, vel_grid], axis=-1)
                data.append(combined_data)

    return np.array(data), np.array(timestamps)

def create_sequences(data, timestamps, input_seq_length=3, forecast_horizon=3):
    X, y, time_diffs = [], [], []
    
    for i in range(len(data) - input_seq_length - forecast_horizon + 1):
        X.append(data[i:i+input_seq_length])
        y.append(data[i+input_seq_length+forecast_horizon-1])
        time_diff = (timestamps[i+input_seq_length+forecast_horizon-1] - timestamps[i]).astype('timedelta64[m]').astype(int)
        time_diffs.append(time_diff)

    return np.array(X), np.array(y), np.array(time_diffs)

def build_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    
    x = TimeDistributed(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))(x)
    
    x = TimeDistributed(Flatten())(x)
    
    x = LSTM(128, return_sequences=False)(x)
    
    time_input = Input(shape=(1,))
    
    x = Concatenate()([x, time_input])
    
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    
    output = Dense(np.prod(output_shape), activation='linear')(x)
    output = tf.keras.layers.Reshape(output_shape)(output)
    
    model = Model(inputs=[inputs, time_input], outputs=output)
    return model

def analyze_predictions(predictions, y_test, threshold=35):
    precipitation_detected = []
    movement_vectors = []

    for pred, true in zip(predictions, y_test):
        pred_reflectivity = pred[..., 0]
        true_reflectivity = true[..., 0]

        pred_precipitation = pred_reflectivity > threshold
        true_precipitation = true_reflectivity > threshold

        precipitation_detected.append({
            'predicted': pred_precipitation.any(),
            'actual': true_precipitation.any()
        })

        if pred_precipitation.any() and true_precipitation.any():
            pred_com = np.array(np.where(pred_precipitation)).mean(axis=1)
            true_com = np.array(np.where(true_precipitation)).mean(axis=1)
            movement = true_com - pred_com
            movement_vectors.append(movement)
        else:
            movement_vectors.append(None)

    return precipitation_detected, movement_vectors

def plot_reflectivity(actual, predicted, slice_index=2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(actual[slice_index, :, :], cmap='viridis')
    ax1.set_title('Actual Reflectivity')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(predicted[slice_index, :, :], cmap='viridis')
    ax2.set_title('Predicted Reflectivity')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def plot_temporal_evolution(actual_sequence, predicted_future, time_diff):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i in range(3):
        im = axes[i].imshow(actual_sequence[i, 2, :, :, 0], cmap='viridis')
        axes[i].set_title(f'T-{2-i}')
        plt.colorbar(im, ax=axes[i])
    
    im = axes[3].imshow(predicted_future[2, :, :, 0], cmap='viridis')
    axes[3].set_title(f'T+{time_diff} (Predicted)')
    plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.show()

def main():
    folder_path = '/home/vishwajitsarnobat/Downloads/isro_hackathon_data'
    grid_size = 5
    input_seq_length = 3
    forecast_horizon = 3

    data, timestamps = load_and_preprocess_data(folder_path, grid_size)
    X, y, time_diffs = create_sequences(data, timestamps, input_seq_length, forecast_horizon)

    X_train, X_test, y_train, y_test, time_diffs_train, time_diffs_test = train_test_split(
        X, y, time_diffs, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train_scaled = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_test_scaled = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    input_shape = (input_seq_length, grid_size, grid_size, grid_size, 2)
    output_shape = (grid_size, grid_size, grid_size, 2)
    model = build_model(input_shape, output_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(
        [X_train_scaled, time_diffs_train], y_train_scaled,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
    )

    test_loss, test_mae = model.evaluate([X_test_scaled, time_diffs_test], y_test_scaled)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    predictions = model.predict([X_test_scaled, time_diffs_test])
    predictions = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)

    precipitation_results, movements = analyze_predictions(predictions, y_test)

    correct_detections = sum(1 for res in precipitation_results if res['predicted'] == res['actual'])
    accuracy = correct_detections / len(precipitation_results)
    print(f"Precipitation Detection Accuracy: {accuracy:.2f}")

    valid_movements = [m for m in movements if m is not None]
    if valid_movements:
        avg_movement = np.mean(valid_movements, axis=0)
        print(f"Average Movement Vector: {avg_movement}")
        
        current_position = np.array([2, 2, 2])
        future_position = current_position + avg_movement
        print(f"Predicted future position: {future_position}")
    else:
        print("No valid movements detected")

    plot_reflectivity(y_test[0, ..., 0], predictions[0, ..., 0])
    plot_temporal_evolution(X_test[0], predictions[0], time_diffs_test[0])

    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.flatten(), predictions.flatten())

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    model.save('precipitation_model.keras')

if __name__ == "__main__":
    main()

