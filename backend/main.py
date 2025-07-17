from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import datetime
import json
import os
import joblib # For loading/saving the model and scaler
import pandas as pd # For DataFrame operations during feature engineering
import numpy as np # For numerical operations
import math # For math.sqrt in feature engineering (Haversine)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest # For retraining
from fastapi.middleware.cors import CORSMiddleware
import hashlib # For SHA-256 hashing
import gzip # For compressing logs

# Configuration
BEHAVIOR_LOG_FILE = "behavior_logs.jsonl.gz" # Added .gz extension for compressed file
MODEL_PATH = "isolation_forest_model.joblib"
SCALER_PATH = "standard_scaler.joblib" # Path to load the StandardScaler
PROCESSED_DATA_PATH = "processed_behavior_features.csv" # Used to get column order for scaler

# Initialize FastAPI app
app = FastAPI(
        title="BEACON Behavioral Backend",
        description="API for receiving and processing behavioral data from the BEACON web app.",
        version="0.1.0"
)

# CORS Middleware
# This is crucial for allowing the frontend to communicate with the backend
origins = [
        "https://beacon-7vpe.onrender.com/",  # Your React dev server
        "http://127.0.0.1:5173"
]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],  # Allows all headers
)

# Pydantic model for the *inner* 'data' field's structure (behavioral events)
class BehaviorDataPayload(BaseModel):
        typingPatterns: Optional[List[Dict[str, Any]]] = None
        mouseMovements: Optional[List[Dict[str, Any]]] = None
        clickEvents: Optional[List[Dict[str, Any]]] = None
        deviceInfo: Dict[str, Any]
        fingerprint: Optional[str] = None

# Pydantic model for the *full incoming request body* from the frontend
class BehaviorLogEntry(BaseModel):
        timestamp: str
        user_email: Optional[str] = None
        context: Optional[Dict[str, Any]] = None # Frontend sends context here
        data: BehaviorDataPayload # Now uses the specific payload model


# Global Variables for ML Model and Scaler
# These will be loaded once when the application starts
ml_model = None
scaler = None
# This list will be populated during startup based on processed_behavior_features.csv
# to ensure column order and selection for scaling are consistent.
numerical_cols_to_scale: List[str] = []
all_feature_columns: List[str] = [] # To store the exact order of all features for prediction

# This list will hold our collected raw behavior data temporarily in memory
# Data will be lost when the server restarts unless loaded from file
temporary_behavior_storage: List[BehaviorLogEntry] = []

# Helper functions for file-based storage
def load_behavior_logs() -> List[BehaviorLogEntry]:
        """Loads behavior logs from the compressed JSONL file."""
        logs = []
        if os.path.exists(BEHAVIOR_LOG_FILE):
            with gzip.open(BEHAVIOR_LOG_FILE, 'rt', encoding='utf-8') as f: # CHANGED: Use gzip.open for reading
                for line in f:
                    try:
                        log_data = json.loads(line.strip())
                        # Validate and append using the Pydantic model
                        logs.append(BehaviorLogEntry(**log_data))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from compressed log file: {line.strip()} - {e}")
                    except Exception as e:
                        print(f"Error creating BehaviorLogEntry from compressed log file data: {log_data} - {e}")
        return logs

def save_behavior_log(log_entry: BehaviorLogEntry):
        """Appends a new behavior log entry to the compressed JSONL file."""
        with gzip.open(BEHAVIOR_LOG_FILE, 'at', encoding='utf-8') as f: # Use gzip.open for appending
            f.write(json.dumps(log_entry.model_dump(by_alias=True)) + '\n') # .model_dump() is for Pydantic v2

# Feature Extraction Function
# This function is designed to extract features from a single behavior data entry
# It must be identical to the one in feature_engineering.py that generated the processed_behavior_features.csv
def extract_features(behavior_data: Dict[str, Any], context_data: Optional[Dict[str, Any]] = None, timestamp_str: Optional[str] = None) -> Dict[str, Any]:
        features = {}

        # Access the 'data' and 'context' parts of the log_entry_data
        # Note: timestamp_str is passed directly from log_entry in receive_behavior_data
        # so we don't need to get it from behavior_data here.
        
        # Typing Patterns Features
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
                    features['avg_dwell_time_ms'] = np.mean(dwell_times)
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

        # Mouse movements
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

        # Click Events Features
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

        # Device Info Features
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

        # Fingerprint Feature
        # Use SHA-256 for fingerprint hashing
        fingerprint_raw = behavior_data.get('fingerprint', '')
        features['fingerprint_hash'] = int(hashlib.sha256(fingerprint_raw.encode('utf-8')).hexdigest(), 16) % (2**31 - 1) # Convert to int for model compatibility

        # Temporal features from timestamp
        # Use the top-level timestamp from log_entry_data
        if timestamp_str:
            try:
                # Parse ISO format timestamp
                dt_object = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
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
            
        # Geolocation Features
        # NEW: Check for batterySaverMode and prioritize it for location handling
        battery_saver_mode_active = context_data.get('batterySaverMode', False) if context_data else False
        
        if battery_saver_mode_active:
            features['distance_from_trusted_loc'] = 0
            features['is_impossible_travel'] = 0 # Location tracking is disabled, so no impossible travel
        elif context_data and 'location' in context_data:
            loc = context_data['location']
            if loc is not None and isinstance(loc, dict):
                if 'latitude' in loc and 'longitude' in loc:
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
                        
                    distance = haversine(trusted_latitude, trusted_longitude, loc['latitude'], loc['longitude'])
                    
                    features['distance_from_trusted_loc'] = distance
                    # Flag if distance is greater than 100km (as a simple rule for "impossible travel")
                    features['is_impossible_travel'] = 1 if distance > 100 else 0
                else:
                    features['distance_from_trusted_loc'] = 0
                    features['is_impossible_travel'] = 0
            else:
                features['distance_from_trusted_loc'] = 0
                features['is_impossible_travel'] = 0
        else:
            # Default values if no geolocation is available and not in battery saver mode
            features['distance_from_trusted_loc'] = 0
            features['is_impossible_travel'] = 0
            
        return features

# Load Model and Scaler on Startup
@app.on_event("startup")
async def load_ml_assets():
        global ml_model, scaler, numerical_cols_to_scale, all_feature_columns, temporary_behavior_storage

        # Load the trained Isolation Forest model
        try:
            ml_model = joblib.load(MODEL_PATH)
            print(f"ML Model '{MODEL_PATH}' loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError(f"ML Model file not found at {MODEL_PATH}. Ensure it's in the same directory.")
        except Exception as e:
            raise RuntimeError(f"Error loading ML model: {e}")

        # Load the processed data to fit the scaler and get the exact column order
        try:
            df_processed = pd.read_csv(PROCESSED_DATA_PATH)
            X_base = df_processed.drop('label', axis=1)
            
            # Store the exact order of all columns the model was trained on
            all_feature_columns = X_base.columns.tolist()

            # Identify numerical columns for scaling, including the new temporal and geolocation features
            numerical_cols_to_scale = [
                'typing_duration_sec', 'char_per_sec', 'avg_dwell_time_ms',
                'std_dev_dwell_time_ms', 'avg_flight_time_ms', 'std_dev_flight_time_ms',
                'backspace_ratio', 'total_mouse_distance_px', 'avg_mouse_speed_px_sec',
                'max_mouse_speed_px_sec', 'std_dev_mouse_speed_px_sec', 'mouse_movement_width',
                'mouse_movement_height', 'mouse_movement_area_px', 'num_clicks',
                'avg_hover_time_ms', 'std_dev_hover_time_ms', 'off_target_click_ratio',
                'clicks_per_sec', 'avg_time_between_clicks_ms', 'std_dev_time_between_clicks_ms',
                'screen_width', 'screen_height', 'screen_aspect_ratio', 'viewport_width',
                'viewport_height', 'viewport_aspect_ratio', 'fingerprint_hash',
                'time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin', 'day_of_week_cos',
                'distance_from_trusted_loc', 'is_impossible_travel'
            ]
            
            # Initialize and fit the scaler to the training data's numerical columns
            scaler = StandardScaler()
            scaler.fit(X_base[numerical_cols_to_scale])
            print("StandardScaler fitted successfully using processed training data.")

        except FileNotFoundError:
            raise RuntimeError(f"Processed data file not found at {PROCESSED_DATA_PATH}. Needed to fit StandardScaler.")
        except Exception as e:
            raise RuntimeError(f"Error initializing StandardScaler: {e}")

        # Initialize storage
        temporary_behavior_storage = load_behavior_logs()
        print(f"Loaded {len(temporary_behavior_storage)} existing behavior logs from {BEHAVIOR_LOG_FILE}")


@app.get("/")
async def root():
        return {"message": "BEACON Backend is running!"}

# NEW: Sophisticated Trust Logic Function
def calculate_trust_score(raw_if_score: float, is_impossible_travel: bool, battery_saver_mode_active: bool) -> float:
        """
        Calculates a sophisticated trust score based on raw Isolation Forest score
        and contextual factors like impossible travel and battery saver mode.
        """
        # Base sigmoid mapping for behavioral anomaly score
        # A score of 0.1 (raw IF score) maps to 0.5 trust. Lower raw scores give lower trust.
        # Adjusting parameters for a more intuitive mapping:
        # - Center the sigmoid around a slightly positive raw score (e.g., 0.1)
        #   so that scores below this are clearly "less trusted" and above are "more trusted".
        # - The '8' controls the steepness. Higher value means sharper transition.
        trust_from_behavior = 1 / (1 + np.exp(8 * (raw_if_score - 0.1)))
        
        # Apply contextual adjustment for impossible travel
        if is_impossible_travel:
            final_trust_score = min(trust_from_behavior, 0.2) # Cap at 20% if impossible travel
        else:
            final_trust_score = trust_from_behavior

        # NEW: Adjust for Battery Saver Mode
        if battery_saver_mode_active:
            # When in battery saver mode, data collection is reduced,
            # so the model's output might be less reliable or fluctuate more.
            # We can choose to:
            # 1. Be less punitive: Do not lower the score as aggressively for anomalies.
            # 2. Maintain a "stable" score: If the score is already high, keep it high.
            # 3. Apply a minimum score: Ensure it doesn't drop too low just because less data is available.
            # For this fix, let's make it less punitive and ensure a reasonable baseline.
            # If the base trust score is low, let's gently push it up,
            # but if it's already good, maintain it.
            if final_trust_score < 0.5: # If current trust is below 50%
                final_trust_score = (final_trust_score + 0.5) / 2 # Average with 0.5 to raise it slightly
            # Also, ensure it doesn't get capped by impossible travel if impossible travel is 0
            # because of battery saver (already handled in extract_features).
            # The intention here is to prevent score stagnation purely due to lack of data.
            pass # The existing logic for is_impossible_travel already takes precedence.
                 # No further direct trust score capping *solely* due to battery saver here.
            
        # Ensure the score is clamped between 0 and 1
        return max(0, min(1, final_trust_score))


@app.post("/api/behavior")
async def receive_behavior_data(log_entry: BehaviorLogEntry, request: Request):
        """
        Receives behavioral data from the frontend, processes it, and performs anomaly detection.
        Stores the raw data in a local file.
        """
        if ml_model is None or scaler is None or not all_feature_columns:
            raise HTTPException(status_code=503, detail="ML model, scaler, or feature columns not loaded. Server is not ready.")

        try:
            # Extract features from the incoming raw data and context
            features_dict = extract_features(
                log_entry.data.model_dump(by_alias=True), # Pass behavior data
                log_entry.context, # Pass context data
                log_entry.timestamp # Pass top-level timestamp for temporal features
            )

            # Convert features dictionary to a Pandas DataFrame row
            features_df = pd.DataFrame([features_dict])

            # Handle categorical features using one-hot encoding
            features_df_encoded = pd.get_dummies(features_df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

            # Align columns with the training data
            missing_cols = set(all_feature_columns) - set(features_df_encoded.columns)
            for c in missing_cols:
                features_df_encoded[c] = 0
            
            # Ensure the order of columns is the same as the training data
            features_final = features_df_encoded[all_feature_columns]

            # Apply scaling to the numerical columns
            features_final[numerical_cols_to_scale] = scaler.transform(features_final[numerical_cols_to_scale])

            # Make prediction and get anomaly score
            anomaly_prediction = ml_model.predict(features_final)[0]
            raw_anomaly_score = ml_model.decision_function(features_final)[0] # Renamed for clarity

            # Determine if it's anomalous from model prediction
            is_anomalous_from_model = 1 if anomaly_prediction == -1 else 0
            
            # Get is_impossible_travel flag from extracted features
            is_impossible_travel = bool(features_dict.get('is_impossible_travel', 0))
            
            # Get battery_saver_mode flag from context
            battery_saver_mode_active = log_entry.context.get('batterySaverMode', False) if log_entry.context else False

            # Apply Shadow Mode logic to the response
            is_safe_mode = log_entry.context.get('safeMode', False) if log_entry.context else False
            
            if is_safe_mode:
                print(f"Shadow Mode active. Prediction was {'Anomalous' if is_anomalous_from_model else 'Normal'} (Raw Score: {raw_anomaly_score:.4f}) but reporting 'Normal'.")
                is_anomalous_for_frontend = False
                anomaly_score_for_frontend = 0.5  # Neutral trust score for shadow mode
            else:
                is_anomalous_for_frontend = bool(is_anomalous_from_model)
                # Use the new sophisticated trust logic function, passing battery_saver_mode_active
                anomaly_score_for_frontend = calculate_trust_score(raw_anomaly_score, is_impossible_travel, battery_saver_mode_active)
                
            user_email = log_entry.user_email if log_entry.user_email else "anonymous@example.com"
            
            save_behavior_log(log_entry)
            temporary_behavior_storage.append(log_entry)

            print(f"Received behavior data for {user_email}. Model Prediction: {'Anomalous' if is_anomalous_from_model else 'Normal'} (Raw Score: {raw_anomaly_score:.4f}). Reported to frontend: {'Anomalous' if is_anomalous_for_frontend else 'Normal'} (Trust Score: {anomaly_score_for_frontend:.4f}). Battery Saver Active: {battery_saver_mode_active}")
            
            return JSONResponse(content={
                "message": "Behavior data processed successfully",
                "status": "success",
                "is_anomalous": is_anomalous_for_frontend,
                "anomaly_score": float(anomaly_score_for_frontend)
            }, status_code=200)

        except Exception as e:
            print(f"Error processing behavior data: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/api/logout_user")
async def logout_user(user_email: str):
        """
        Logs out a user by invalidating their session.
        Can be called when multiple anomalies are detected.
        """
        # In a real implementation, we would invalidate the user's session/token
        
        print(f"FORCED LOGOUT: User {user_email} was logged out due to security anomalies")
        
        return {
            "status": "success",
            "message": f"User {user_email} has been logged out",
            "logout_required": True
        }
# Behavioral Drift (Retraining) Logic
def retrain_model_on_logs(user_email: Optional[str] = None):
        """
        Function to retrain the Isolation Forest model using data from the logs.
        """
        print(f"Starting model retraining process for user: {user_email if user_email else 'All Users'}...")
        try:
            all_logs = load_behavior_logs()
            
            if user_email:
                logs_for_training = [log for log in all_logs if log.user_email == user_email]
                if not logs_for_training:
                    print(f"No logs found for user {user_email} to retrain the model.")
                    return
            else:
                logs_for_training = all_logs

            if not logs_for_training:
                print("No new logs to retrain the model with.")
                return

            features_list = []
            labels_list = []
            for log_entry in logs_for_training:
                # Pass the top-level timestamp to extract_features
                features = extract_features(log_entry.data.model_dump(by_alias=True), log_entry.context, log_entry.timestamp)
                features_list.append(features)
                labels_list.append(0)

            full_df = pd.DataFrame(features_list)
            full_df_encoded = pd.get_dummies(full_df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

            missing_cols_df = set(all_feature_columns) - set(full_df_encoded.columns)
            for c in missing_cols_df:
                full_df_encoded[c] = 0
            
            X_retrain = full_df_encoded[all_feature_columns]

            new_scaler = StandardScaler()
            X_retrain_scaled = X_retrain.copy()
            X_retrain_scaled[numerical_cols_to_scale] = new_scaler.fit_transform(X_retrain[numerical_cols_to_scale])
            
            contamination_rate_for_retrain = 0.01

            new_model = IsolationForest(n_estimators=100, contamination=contamination_rate_for_retrain, random_state=42)
            new_model.fit(X_retrain_scaled)
            
            joblib.dump(new_model, MODEL_PATH)
            joblib.dump(new_scaler, SCALER_PATH)
            
            global ml_model, scaler
            ml_model = new_model
            scaler = new_scaler

            print("Model retraining complete! New model and scaler saved and loaded into memory.")
            
        except Exception as e:
            print(f"Error during model retraining: {e}")
            import traceback
            traceback.print_exc()
        
@app.post("/api/retrain_model")
async def trigger_retrain(background_tasks: BackgroundTasks, user_email: Optional[str] = None):
        """
        Triggers a background task to retrain the model using logged data.
        """
        background_tasks.add_task(retrain_model_on_logs, user_email=user_email)
        return {"message": f"Model retraining has been triggered in the background for user {user_email if user_email else 'all users'}. Check server logs for status."}

@app.get("/api/behavior_logs")
async def get_behavior_logs():
        """
        Retrieves all temporarily stored raw behavior logs from memory.
        """
        return temporary_behavior_storage

@app.get("/api/user_anomalies/{user_email}")
async def get_user_anomalies(user_email: str):
        """
        Returns all logs and anomaly results for a specific user.
        """
        user_logs = [log for log in temporary_behavior_storage if log.user_email == user_email]
        response = []

        for log in user_logs:
            # Pass the top-level timestamp to extract_features
            features = extract_features(log.data.model_dump(by_alias=True), log.context, log.timestamp)
            df = pd.DataFrame([features])
            df = pd.get_dummies(df)
            for col in all_feature_columns:
                if col not in df:
                    df[col] = 0
            df = df[all_feature_columns]
            df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale])
            pred = ml_model.predict(df)[0]
            raw_anomaly_score = ml_model.decision_function(df)[0] # Get raw score
            
            # Get is_impossible_travel flag from extracted features for this log
            is_impossible_travel = bool(features.get('is_impossible_travel', 0))
            # Get battery_saver_mode flag from context for this log
            battery_saver_mode_active = log.context.get('batterySaverMode', False) if log.context else False


            response.append({
                "timestamp": log.timestamp,
                "anomaly": pred == -1,
                "anomaly_score": round(calculate_trust_score(raw_anomaly_score, is_impossible_travel, battery_saver_mode_active), 4), # Use new trust logic
                "context": log.context
            })

        return {"user": user_email, "logs": response}

@app.get("/api/dashboard_summary")
async def get_dashboard_summary():
        """
        Returns a summary of recent anomaly trends for visualization.
        """
        from collections import defaultdict
        count_by_user = defaultdict(lambda: {"total": 0, "anomalies": 0})

        for log in temporary_behavior_storage:
            email = log.user_email or "unknown"
            count_by_user[email]["total"] += 1
            # Pass the top-level timestamp to extract_features
            features = extract_features(log.data.model_dump(by_alias=True), log.context, log.timestamp)
            df = pd.DataFrame([features])
            df = pd.get_dummies(df)
            for col in all_feature_columns:
                if col not in df:
                    df[col] = 0
            df = df[all_feature_columns]
            df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale])
            pred = ml_model.predict(df)[0]
            if pred == -1:
                count_by_user[email]["anomalies"] += 1

        return count_by_user

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
