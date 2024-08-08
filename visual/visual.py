import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load the dataset
ds = xr.open_dataset('/home/vishwajitsarnobat/Downloads/isro_hackathon_data/RCTLS_01JUL2024_000543_L2C_STD.nc')

# Access the VEL and DBZ variables
vel = ds['VEL']
dbz = ds['DBZ']

# Print the shape of the variables
print("VEL shape:", vel.shape)
print("DBZ shape:", dbz.shape)

# Function to update the plot based on the height index
def update_plot(height_index):
    time_index = 0  # First (and only) time index

    # Extract the data for plotting
    reflectivity_slice = dbz[time_index, height_index, :, :]

    # Clear the previous plot
    clear_output(wait=True)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.contourf(reflectivity_slice, cmap='viridis')
    plt.colorbar(label='Reflectivity (dBZ)')
    plt.title(f'Reflectivity at height index {height_index}')
    plt.xlabel('Longitude index')
    plt.ylabel('Latitude index')
    plt.show()

# Create a slider for height index
height_slider = widgets.IntSlider(min=0, max=dbz.shape[1] - 1, step=1, value=0, description='Height Index')

# Link the slider to the update_plot function
interactive_plot = widgets.interactive(update_plot, height_index=height_slider)

# Display the slider and the interactive plot
display(interactive_plot)
