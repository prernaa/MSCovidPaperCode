## dtypes for sensors

DEF_SENSOR_DTYPES = {
    "locations": {'_id': 'int64', 'timestamp': 'int64', 'device_id': 'O', 'double_latitude': 'float64', 'double_longitude': 'float64', 'double_bearing': 'float64', 'double_speed': 'float64', 'double_altitude': 'float64', 'provider': 'O', 'accuracy': 'float64', 'label': 'float64'}
}

DEF_USE_COLS_ALWAYS = ["_id", 'timestamp', 'device_id']

DEF_USE_COLS = {
    "locations": DEF_USE_COLS_ALWAYS + ['double_latitude', 'double_longitude'], \
    "bluetooth": DEF_USE_COLS_ALWAYS + ['bt_address'], \
    "wifi": DEF_USE_COLS_ALWAYS + ['bssid']
}


## NOT USED IN THE FIXING DEVICE ID FILE
DEF_COLS_FOR_DUPLICATE_REMOVAL = {
    "survey": ['timestamp', 'device_id'],\
    "locations": ['timestamp', 'device_id'],\
    "applications_foreground": ['timestamp', 'device_id'],\
    "screen": ['timestamp', 'device_id'],\
    "wifi": ['timestamp', 'device_id'],
}