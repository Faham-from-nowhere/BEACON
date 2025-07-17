import pandas as pd
import json
import math
from datetime import datetime, timezone
import numpy as np

# --- Feature Extraction Function ---
# This function is designed to extract features from a single behavior data entry
def extract_features(log_entry_data: dict) -> dict:
    features = {}

    # Access the 'data' and 'context' parts of the log_entry_data
    behavior_data = log_entry_data.get('data', {})
    context_data = log_entry_data.get('context', {})
    
    # --- 1. Typing Patterns Features ---
    typing_patterns = behavior_data.get('typingPatterns', [])
    if typing_patterns:
        # Ensure timestamps are numbers and sort them
        timestamps = sorted([kp['timestamp'] for kp in typing_patterns if 'timestamp' in kp and kp['timestamp'] is not None])
        dwell_times = [kp['dwellTime'] for kp in typing_patterns if 'dwellTime' in kp and kp['dwellTime'] is not None]

        if len(timestamps) > 1:
            typing_duration = (timestamps[-1] - timestamps[0]) / 1000 # in seconds
            features['typing_duration_sec'] = typing_duration
            if typing_duration > 0:
                features['char_per_sec'] = len(typing_patterns) / typing_duration
            else:
                features['char_per_sec'] = 0

            flight_times = []
            for i in range(len(timestamps) - 1):
                # Flight time is the duration between key up of current and key down of next
                # Assuming dwellTime is already available from tracker, otherwise needs keyUp timestamp
                if 'dwellTime' in typing_patterns[i] and typing_patterns[i]['dwellTime'] is not None:
                     flight_time_ms = typing_patterns[i+1]['timestamp'] - (typing_patterns[i]['timestamp'] + typing_patterns[i]['dwellTime'])
                     if flight_time_ms >= 0: # Ensure non-negative flight times
                         flight_times.append(flight_time_ms)

            if dwell_times:
                features['avg_dwell_time_ms'] = sum(dwell_times) / len(dwell_times)
                features['std_dev_dwell_time_ms'] = pd.Series(dwell_times).std() if len(dwell_times) > 1 else 0
            else:
                features['avg_dwell_time_ms'] = 0
                features['std_dev_dwell_time_ms'] = 0

            if flight_times:
                features['avg_flight_time_ms'] = sum(flight_times) / len(flight_times)
                features['std_dev_flight_time_ms'] = pd.Series(flight_times).std() if len(flight_times) > 1 else 0
            else:
                features['avg_flight_time_ms'] = 0
                features['std_dev_flight_time_ms'] = 0

        else: # Handle cases with 0 or 1 typing event
            features['typing_duration_sec'] = 0
            features['char_per_sec'] = 0
            features['avg_dwell_time_ms'] = dwell_times[0] if dwell_times else 0
            features['std_dev_dwell_time_ms'] = 0
            features['avg_flight_time_ms'] = 0
            features['std_dev_flight_time_ms'] = 0

        backspaces = sum(1 for kp in typing_patterns if kp['key'] == 'Backspace')
        features['backspace_ratio'] = backspaces / len(typing_patterns) if len(typing_patterns) > 0 else 0
    else:
        features['typing_duration_sec'] = 0
        features['char_per_sec'] = 0
        features['avg_dwell_time_ms'] = 0
        features['std_dev_dwell_time_ms'] = 0
        features['avg_flight_time_ms'] = 0
        features['std_dev_flight_time_ms'] = 0
        features['backspace_ratio'] = 0

    # --- 2. Mouse movements ---
    mouse_movements = behavior_data.get('mouseMovements', [])
    if mouse_movements and len(mouse_movements) > 1:
        mouse_movements.sort(key=lambda x: x['timestamp'])
        total_dist = 0
        speeds = []
        for i in range(len(mouse_movements) - 1):
            x1, y1, t1 = mouse_movements[i]['x'], mouse_movements[i]['y'], mouse_movements[i]['timestamp']
            x2, y2, t2 = mouse_movements[i+1]['x'], mouse_movements[i+1]['y'], mouse_movements[i+1]['timestamp']
            
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_dist += distance
            
            time_diff_sec = (t2 - t1) / 1000
            if time_diff_sec > 0:
                speeds.append(distance / time_diff_sec)

        features['total_mouse_distance_px'] = total_dist
        if speeds:
            features['avg_mouse_speed_px_sec'] = sum(speeds) / len(speeds)
            features['max_mouse_speed_px_sec'] = max(speeds)
            features['std_dev_mouse_speed_px_sec'] = pd.Series(speeds).std() if len(speeds) > 1 else 0
        else:
            features['avg_mouse_speed_px_sec'] = 0
            features['max_mouse_speed_px_sec'] = 0
            features['std_dev_mouse_speed_px_sec'] = 0

        xs = [m['x'] for m in mouse_movements]
        ys = [m['y'] for m in mouse_movements]
        if xs and ys:
            features['mouse_movement_width'] = max(xs) - min(xs)
            features['mouse_movement_height'] = max(ys) - min(ys)
            features['mouse_movement_area_px'] = features['mouse_movement_width'] * features['mouse_movement_height']
        else:
            features['mouse_movement_width'] = 0
            features['mouse_movement_height'] = 0
            features['mouse_movement_area_px'] = 0

    else:
        features['total_mouse_distance_px'] = 0
        features['avg_mouse_speed_px_sec'] = 0
        features['max_mouse_speed_px_sec'] = 0
        features['std_dev_mouse_speed_px_sec'] = 0
        features['mouse_movement_width'] = 0
        features['mouse_movement_height'] = 0
        features['mouse_movement_area_px'] = 0

    # --- 3. Click Events Features ---
    click_events = behavior_data.get('clickEvents', [])
    if click_events:
        features['num_clicks'] = len(click_events)
        
        hover_times = [ce['hoverTime'] for ce in click_events if 'hoverTime' in ce and ce['hoverTime'] is not None]
        if hover_times:
            features['avg_hover_time_ms'] = sum(hover_times) / len(hover_times)
            features['std_dev_hover_time_ms'] = pd.Series(hover_times).std() if len(hover_times) > 1 else 0
        else:
            features['avg_hover_time_ms'] = 0
            features['std_dev_hover_time_ms'] = 0

        off_target_clicks = sum(1 for ce in click_events if ce['target'] in ['HTML', 'BODY'])
        features['off_target_click_ratio'] = off_target_clicks / features['num_clicks'] if features['num_clicks'] > 0 else 0

        click_timestamps = sorted([ce['timestamp'] for ce in click_events if 'timestamp' in ce and ce['timestamp'] is not None])
        if len(click_timestamps) > 1:
            session_duration_clicks = (click_timestamps[-1] - click_timestamps[0]) / 1000
            if session_duration_clicks > 0:
                features['clicks_per_sec'] = features['num_clicks'] / session_duration_clicks
            else:
                features['clicks_per_sec'] = 0
            
            time_between_clicks = [(click_timestamps[i+1] - click_timestamps[i]) for i in range(len(click_timestamps) - 1)]
            if time_between_clicks:
                features['avg_time_between_clicks_ms'] = sum(time_between_clicks) / len(time_between_clicks)
                features['std_dev_time_between_clicks_ms'] = pd.Series(time_between_clicks).std() if len(time_between_clicks) > 1 else 0
            else:
                features['avg_time_between_clicks_ms'] = 0
                features['std_dev_time_between_clicks_ms'] = 0
        else:
            features['clicks_per_sec'] = 0
            features['avg_time_between_clicks_ms'] = 0
            features['std_dev_time_between_clicks_ms'] = 0

    else:
        features['num_clicks'] = 0
        features['avg_hover_time_ms'] = 0
        features['std_dev_hover_time_ms'] = 0
        features['off_target_click_ratio'] = 0
        features['clicks_per_sec'] = 0
        features['avg_time_between_clicks_ms'] = 0
        features['std_dev_time_between_clicks_ms'] = 0

    # --- 4. Device Info Features ---
    device_info = behavior_data.get('deviceInfo', {})
    
    if 'screenResolution' in device_info and device_info['screenResolution']:
        try:
            width, height = map(int, device_info['screenResolution'].split('x'))
            features['screen_width'] = width
            features['screen_height'] = height
            features['screen_aspect_ratio'] = width / height if height > 0 else 0
        except ValueError:
            features['screen_width'] = 0
            features['screen_height'] = 0
            features['screen_aspect_ratio'] = 0
    else:
        features['screen_width'] = 0
        features['screen_height'] = 0
        features['screen_aspect_ratio'] = 0
    
    if 'viewportSize' in device_info and device_info['viewportSize']:
        try:
            vw, vh = map(int, device_info['viewportSize'].split('x'))
            features['viewport_width'] = vw
            features['viewport_height'] = vh
            features['viewport_aspect_ratio'] = vw / vh if vh > 0 else 0
        except ValueError:
            features['viewport_width'] = 0
            features['viewport_height'] = 0
            features['viewport_aspect_ratio'] = 0
    else:
        features['viewport_width'] = 0
        features['viewport_height'] = 0
        features['viewport_aspect_ratio'] = 0

    features['browser'] = device_info.get('browser', 'unknown').lower()
    features['os'] = device_info.get('os', 'unknown').lower()
    # Use timezone from context if available, otherwise from deviceInfo, fallback to 'unknown'
    features['timezone'] = context_data.get('timezone', device_info.get('timezone', 'unknown'))

    # --- 5. Fingerprint Feature ---
    features['fingerprint_hash'] = hash(behavior_data.get('fingerprint', ''))

    # --- 6. Temporal features from timestamp ---
    # Use the top-level timestamp from log_entry_data
    timestamp_str = log_entry_data.get('timestamp')
    if timestamp_str:
        try:
            # Parse ISO format timestamp
            dt_object = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour = dt_object.hour
            day_of_week = dt_object.weekday()  # Monday is 0 and Sunday is 6
            features['time_of_day_sin'] = np.sin(2 * np.pi * hour / 24)
            features['time_of_day_cos'] = np.cos(2 * np.pi * hour / 24)
            features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        except ValueError:
            # Handle invalid timestamp format
            features['time_of_day_sin'] = 0
            features['time_of_day_cos'] = 0
            features['day_of_week_sin'] = 0
            features['day_of_week_cos'] = 0
    else:
        features['time_of_day_sin'] = 0
        features['time_of_day_cos'] = 0
        features['day_of_week_sin'] = 0
        features['day_of_week_cos'] = 0
        
    # --- 7. Geolocation Features ---
    if context_data and 'location' in context_data and context_data['location']:
        geolocation = context_data['location']
        if 'latitude' in geolocation and 'longitude' in geolocation:
            # Secunderabad, Telangana, India (as a fixed trusted location)
            trusted_latitude = 17.4375
            trusted_longitude = 78.4482
            
            # Haversine formula to calculate distance between two lat/lon points
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371  # Radius of Earth in kilometers
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                return R * c
                
            distance = haversine(trusted_latitude, trusted_longitude, geolocation['latitude'], geolocation['longitude'])
            
            features['distance_from_trusted_loc'] = distance
            # Flag if distance is greater than 100km (as a simple rule for "impossible travel")
            features['is_impossible_travel'] = 1 if distance > 100 else 0
        else:
            features['distance_from_trusted_loc'] = 0
            features['is_impossible_travel'] = 0
    else:
        # Default values if no geolocation is available
        features['distance_from_trusted_loc'] = 0
        features['is_impossible_travel'] = 0
        
    return features


# --- Main script execution for processing a JSONL file ---
if __name__ == "__main__":
    INPUT_JSONL_FILE = "behavior_combined_large.jsonl" # Or .json if you saved as single JSON array
    OUTPUT_CSV_FILE = "processed_behavior_features.csv"

    all_extracted_features = []
    labels = []

    try:
        with open(INPUT_JSONL_FILE, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    # Extract features using the updated function
                    extracted_features = extract_features(log_entry)
                    all_extracted_features.append(extracted_features)
                    labels.append(1 if log_entry.get('label') == 'anomalous' else 0)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                except Exception as e:
                    print(f"Error processing log entry: {log_entry} - Error: {e}")

    except FileNotFoundError:
        print(f"Error: '{INPUT_JSONL_FILE}' not found. Please ensure the file is in the same directory.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while reading {INPUT_JSONL_FILE}: {e}")
        exit()

    if not all_extracted_features:
        print("No features extracted. Exiting.")
        exit()

    # Create a DataFrame from the extracted features
    df = pd.DataFrame(all_extracted_features)

    # Add the 'label' column
    df['label'] = labels

    # --- Data Cleaning and Preprocessing (as in your original script) ---
    # Fill NaN values for numerical columns
    # A simple strategy is to fill with 0, but consider median or mean for numerical features.
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # One-hot encode categorical features (browser, os, timezone)
    # This creates new columns for each unique category
    # Handle potential missing categories by explicitly setting known categories if necessary
    # For now, rely on pandas to create columns as needed, but be aware for deployment
    df = pd.get_dummies(df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

    # Convert boolean columns created by get_dummies to int (0 or 1)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    print("\n--- Feature Engineering Complete ---")
    print("\nDataFrame Head:")
    print(df.head())

    print("\nDataFrame Info:")
    df.info()

    print("\nLabel Distribution:")
    print(df['label'].value_counts())

    # Save the processed DataFrame to a CSV file for later use
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nProcessed features saved to '{OUTPUT_CSV_FILE}'")
