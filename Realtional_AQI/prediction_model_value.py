import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import requests
from datetime import datetime,timedelta




import serial
import time
import pandas as pd
from datetime import datetime

SERIAL_PORT = 'COM5' 
BAUD_RATE = 9600
READ_COMMAND = b'\xFF\x01\x86\x00\x00\x00\x00\x00\x79'

def decode_packet(packet):
    if len(packet) >= 26 and packet[0] == 0xFF and packet[1] == 0x86:
        pm1_0 = (packet[2] << 8) | packet[3]
        pm2_5 = (packet[4] << 8) | packet[5]
        pm10  = (packet[6] << 8) | packet[7]
        co2   = (packet[8] << 8) | packet[9]
        
        tvoc = (packet[10] << 8) | packet[11]
        ch2o = (packet[12] << 8) | packet[13]
        
        return {
            'PM_1.0': pm1_0,
            'PM_2.5': pm2_5,
            'PM_10': pm10,
            'CO2': co2,
            'TVOC': tvoc,
            'CH2O': ch2o
        }
    else:
        return None

def get_calibrated_sensor_data():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    reading_count = 0
    target_readings = []
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print("Connected! Sensor warming up. Discarding initial calibration readings...")
        ser.reset_input_buffer()
        
        while reading_count < 22:
            ser.write(READ_COMMAND)
            time.sleep(0.5)
            
            if ser.in_waiting >= 26:
                raw_bytes = ser.read(26)
                data = decode_packet(raw_bytes)
                ser.reset_input_buffer()
                
                if data is not None:
                    reading_count += 1
                    
                    if reading_count < 20:
                        print(f"[-] Discarding calibration reading {reading_count}/19...")
                    else:
                        print(f"[+] Capturing valid reading {reading_count}...")
                        target_readings.append(data)
                        
            time.sleep(2) 

    except serial.SerialException as e:
        print(f"\n[!] Connection Error: {e}")
        return None
    except KeyboardInterrupt:
        print("\n[!] Data collection stopped by user.")
        return None
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
            
    if len(target_readings) == 3:
        print("\nCalculating average of readings 20, 21, and 22...")
        current_sensor_data = {}
        
        for key in target_readings[0].keys():
            total = sum(reading[key] for reading in target_readings)
            current_sensor_data[key] = round(total / 3.0, 2)
            
        print("\n" + "="*40)
        print("FINAL CALIBRATED SENSOR DATA")
        print("="*40)
        for key, value in current_sensor_data.items():
            print(f"'{key}': {value},")
        print("="*40)
        
        return current_sensor_data
    else:
        print("Failed to capture enough readings for calibration.")
        return None

if __name__ == '__main__':
    current_sensor_data = get_calibrated_sensor_data()
    
    if current_sensor_data: 
        print("\nReady to pass 'current_sensor_data' to the XGBoost model!")



if not current_sensor_data:
    print("Connect the sensor!")
    exit()







model = joblib.load('xgb_model_prediction.pkl')

aqi_values = []

# current_sensor_data = {
#     'PM_1.0': 15.2,
#     'PM_2.5': 35.5,
#     'PM_10': 50.1,
#     'CO2': 410.0,
#     'TVOC': 120.0,
#     'CH2O': 0.02,
#     # If your model also expects NO2 or CO from the previous dataset, 
#     # and you don't have them on the new sensor, you must provide default/average values.
#     # 'CO': 0.5, 'NO2': 15.0  <-- Uncomment if your model needs these!
# }


def ravalgla_detail():
  url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": 27.3064,
    "longitude": 88.3643,
    "daily": ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"],
    "timezone": "Asia/Kolkata"
  }

  print("fething ravalgla data : ")
  data = requests.get(url,params=params).json()
  return data['daily']




def predict_aqi():
  weather_data = ravalgla_detail()

  date = weather_data['time']
  preciption = weather_data['precipitation_sum']
  wind_speed = weather_data['windspeed_10m_max']
  temp = weather_data['temperature_2m_max']
  

  for i in range(7):
    current_date = date[i]
    current_date = pd.to_datetime(current_date)
    current_temp = temp[i]
    current_wind_speed = wind_speed[i]
    current_preciption = preciption[i]
    current_humadity = 75.0
    
    data_feature = pd.DataFrame([{
      'PM2_5' : current_sensor_data['PM_2.5'],
      'PM10' : current_sensor_data['PM_10'],
      'CO' : 0.8,
      'NO2' : 12.0,
      'temperature' : current_temp,
      'precipitation' : current_preciption,
      'relative_humidity' : current_humadity,
      'wind_speed_num' : current_wind_speed,
      'hours' : current_date.hour,
      'month' : current_date.month,
      'day_of_week': current_date.dayofweek
    }])

    predict_aqi = model.predict(data_feature)
    print(predict_aqi.item())
    aqi_values.append(predict_aqi.item())


predict_aqi()
     

