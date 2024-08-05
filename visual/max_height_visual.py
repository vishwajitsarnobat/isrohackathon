import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the dataset
dwr = xr.open_dataset('/usr/src/app/RCTLS_01JUL2024_000543_L2C_STD.nc')
print(f"DWR: {list(dwr.keys())}")

# Extract the DBZ variable
dbz = dwr['DBZ']
print(f"DWR: {dbz}")

# Get the maximum DBZ values
max_dbz = dbz.max('height')
print(max_dbz.lat, max_dbz.lon)

# Extract latitude and longitude
dwr_lat = dwr['latitude'].values
dwr_lon = dwr['longitude'].values

# Create a meshgrid for plotting
dwr_lon_grid, dwr_lat_grid = np.meshgrid(dwr_lon, dwr_lat)

# Extract a slice of the data
mdh = max_dbz[0, :, :].values
mdh = np.where(mdh > 0, mdh, np.nan)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)

# Add latitude and longitude labels
ax.set_xticks(np.arange(dwr_lon_grid.min(), dwr_lon_grid.max(), 1.), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(dwr_lat_grid.min(), dwr_lat_grid.max(), 1.), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.2f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{y:.2f}'))

# Plot radar location (example coordinates)
radar_lon = 76.85889  # Replace with the actual longitude
radar_lat = 8.54278    # Replace with the actual latitude
ax.plot(radar_lon, radar_lat, 'ro', markersize=5, transform=ccrs.PlateCarree(), label='TERLS Radar')
ax.legend()

# Contour plot
contour = ax.contourf(dwr_lon_grid, dwr_lat_grid, mdh, cmap='viridis', transform=ccrs.PlateCarree())
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('max_dbz')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('DBZ Max Plot 10JAN2024_152518')

plt.grid(True)
plt.show()
