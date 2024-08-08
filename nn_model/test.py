import xarray
import re

ds = xarray.open_dataset('/home/vishwajitsarnobat/Downloads/isro_hackathon_data/RCTLS_01JUL2024_000543_L2C_STD.nc')
# print(ds['history'])
history = ds.attrs['history']
timestamp = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', history).group(0)
print('Timestamp: ', timestamp)

