from flask import Flask, render_template, jsonify
import threading
import serial
import time
import pandas as pd
import joblib
import requests
from datetime import datetime
import folium

app = Flask(__name__)

SERIAL_PORT = 'COM5' 
BAUD_RATE = 9600
READ_COMMAND = b'\xFF\x01\x86\x00\x00\x00\x00\x00\x79'

try:
    model = joblib.load('xgb_model_prediction.pkl')
except Exception as e:
    model = None

sensor_state = {
    'count': 0,
    'current_raw': {'PM_1.0': 0, 'PM_2.5': 0, 'PM_10': 0, 'CO2': 0, 'TVOC': 0, 'CH2O': 0},
    'is_calibrated': False,
    'final_calibrated_data': None,
    'weekly_forecast': None
}

def decode_packet(packet):
    if len(packet) >= 26 and packet[0] == 0xFF and packet[1] == 0x86:
        return {
            'PM_1.0': (packet[2] << 8) | packet[3],
            'PM_2.5': (packet[4] << 8) | packet[5],
            'PM_10':  (packet[6] << 8) | packet[7],
            'CO2':    (packet[8] << 8) | packet[9],
            'TVOC':   (packet[10] << 8) | packet[11],
            'CH2O':   round(((packet[12] << 8) | packet[13]) / 1000.0, 3)
        }
    return None

def read_sensor_data():
    global sensor_state
    target_readings = []
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        ser.reset_input_buffer()
        
        while True:
            ser.write(READ_COMMAND)
            time.sleep(0.5)
            
            if ser.in_waiting >= 26:
                raw_bytes = ser.read(26)
                data = decode_packet(raw_bytes)
                ser.reset_input_buffer()
                
                if data:
                    sensor_state['current_raw'] = data
                    
                    if not sensor_state['is_calibrated']:
                        sensor_state['count'] += 1
                        
                        if 20 <= sensor_state['count'] <= 22:
                            target_readings.append(data)
                        
                        if sensor_state['count'] == 22:
                            final_data = {}
                            for key in target_readings[0].keys():
                                total = sum(reading[key] for reading in target_readings)
                                final_data[key] = round(total / 3.0, 2)
                                
                            sensor_state['final_calibrated_data'] = final_data
                            sensor_state['is_calibrated'] = True
                            generate_weekly_forecast()
                            
            time.sleep(2) 
    except serial.SerialException as e:
        pass

def generate_weekly_forecast():
    global sensor_state
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 27.3064,
        "longitude": 88.3643,
        "daily": ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"],
        "timezone": "Asia/Kolkata"
    }
    
    try:
        weather_data = requests.get(url, params=params).json()['daily']
        forecast_list = []
        
        base_pollution = sensor_state['final_calibrated_data']
        
        for i in range(7):
            current_date = pd.to_datetime(weather_data['time'][i])
            
            data_feature = pd.DataFrame([{
                'PM2_5': base_pollution['PM_2.5'],
                'PM10': base_pollution['PM_10'],
                'CO': 0.8,
                'NO2': 12.0,
                'temperature': weather_data['temperature_2m_max'][i],
                'precipitation': weather_data['precipitation_sum'][i],
                'relative_humidity': 75.0,
                'wind_speed_num': weather_data['windspeed_10m_max'][i],
                'hours': current_date.hour,
                'month': current_date.month,
                'day_of_week': current_date.dayofweek
            }])
            
            predicted_value = model.predict(data_feature)[0].item()
            
            forecast_list.append({
                'date': current_date.strftime('%A, %b %d'),
                'temp': weather_data['temperature_2m_max'][i],
                'aqi': round(predicted_value)
            })
            
        sensor_state['weekly_forecast'] = forecast_list
    except Exception as e:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_view():
    try:
        df_stations = pd.read_csv('../Station_Details_Historical.csv')
        df_aqi = pd.read_csv('../AQI_Category_Historical.csv')
        df_merged = df_aqi.merge(df_stations, on='Station_id', how='inner')
        df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
        df_recent = df_merged.sort_values('timestamp', ascending=False).drop_duplicates(subset=['Station_id'])
        df_recent[['Latitude', 'Longitude']] = df_recent['Latitude_Longitude'].str.split(', ', expand=True).astype(float)

        map_center = [26.2006, 92.9376]
        aqi_map = folium.Map(location=map_center, zoom_start=6)

        for _, row in df_recent.iterrows():
            if row['category_name'] == 'Good': color = 'darkgreen'
            elif row['category_name'] == 'Satisfactory': color = 'lightgreen'
            elif row['category_name'] == 'Moderate': color = 'orange'
            elif row['category_name'] == 'Poor': color = 'red'
            elif row['category_name'] == 'Very Poor': color = 'purple'
            else: color = 'darkred'
            
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=f"City: {row['City']}<br>AQI: {row['Exact_AQI']}<br>Category: {row['category_name']}",
                tooltip=f"{row['City']} - AQI: {row['Exact_AQI']}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8
            ).add_to(aqi_map)

        return aqi_map.get_root().render()
    except Exception as e:
        return str(e)

@app.route('/api/status')
def status():
    return jsonify(sensor_state)

if __name__ == '__main__':
    sensor_thread = threading.Thread(target=read_sensor_data, daemon=True)
    sensor_thread.start()
    app.run(debug=False, host='0.0.0.0', port=5000)