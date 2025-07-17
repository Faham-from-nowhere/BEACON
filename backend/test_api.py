import requests
import json
import time
import random
import uuid
import datetime # <--- ADDED THIS IMPORT

# Configuration for your FastAPI app
API_URL = "http://localhost:8000/api/behavior"
BEHAVIOR_LOG_FILE = "behavior_logs.jsonl" # <--- ADDED THIS CONSTANT

def generate_sample_behavior_data(user_email: str, is_anomalous_like: bool = False):
    """
    Generates a sample behavior data payload.
    Adjust parameters to simulate more 'normal' or 'anomalous' patterns.
    """
    timestamp_start = int(time.time() * 1000) # Current timestamp in ms

    # --- Typing Patterns ---
    typing_patterns = []
    if is_anomalous_like:
        # Faster, fewer dwell times, less natural
        keys = "asdlfkjweriouf9839485"
        for i, char in enumerate(keys):
            typing_patterns.append({
                "key": char,
                "code": f"Key{char.upper()}",
                "timestamp": timestamp_start + i * random.randint(30, 80), # Fast typing
                "dwellTime": random.randint(10, 40) # Short dwell times
            })
    else:
        # More natural, slightly slower, varied dwell times
        keys = "hello world this is a test message"
        for i, char in enumerate(keys):
            typing_patterns.append({
                "key": char,
                "code": f"Key{char.upper()}" if char != ' ' else 'Space',
                "timestamp": timestamp_start + i * random.randint(70, 150), # Normal typing speed
                "dwellTime": random.randint(50, 120) # Normal dwell times
            })

    # --- Mouse Movements ---
    mouse_movements = []
    current_x, current_y = 500, 300
    for i in range(50):
        if is_anomalous_like:
            # Erratic, large jumps
            current_x += random.randint(-200, 200)
            current_y += random.randint(-150, 150)
        else:
            # Smoother, smaller movements
            current_x += random.randint(-20, 20)
            current_y += random.randint(-15, 15)
        mouse_movements.append({
            "x": max(0, min(1920, current_x)), # Clamp to screen size
            "y": max(0, min(1080, current_y)),
            "timestamp": timestamp_start + i * random.randint(10, 50)
        })

    # --- Click Events ---
    click_events = []
    for i in range(random.randint(3, 8)):
        target_tag = random.choice(['BUTTON', 'INPUT', 'DIV'])
        if is_anomalous_like:
            # More off-target clicks
            target_tag = random.choice(['HTML', 'BODY', 'DIV'])
        click_events.append({
            "x": random.randint(100, 1000),
            "y": random.randint(100, 700),
            "target": target_tag,
            "timestamp": timestamp_start + i * 1000 + random.randint(100, 500),
            "hoverTime": random.randint(50, 500) if not is_anomalous_like else random.randint(10, 100) # Shorter hover for anomalous
        })

    # --- Device Info ---
    device_info = {
        "browser": random.choice(["Chrome", "Firefox", "Edge", "Safari"]),
        "screenResolution": f"{random.choice([1920, 1366])}x{random.choice([1080, 768])}",
        "viewportSize": f"{random.randint(800, 1200)}x{random.randint(600, 900)}",
        "os": random.choice(["Windows", "macOS", "Linux"]),
        "timezone": random.choice(["America/New_York", "Europe/London", "Asia/Kolkata"])
    }

    # --- Fingerprint ---
    fingerprint = str(uuid.uuid4()) # Unique ID for each session

    # This is the structure that main.py's BehaviorLogEntry expects
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_email": user_email,
        "data": { # This 'data' key holds the BehaviorDataPayload
            "typingPatterns": typing_patterns,
            "mouseMovements": mouse_movements,
            "clickEvents": click_events,
            "deviceInfo": device_info,
            "fingerprint": fingerprint
        }
    }

def send_test_data(data):
    """Sends the test data to the FastAPI endpoint."""
    print("\n--- Sending Data to API ---")
    print(f"Target URL: {API_URL}")
    
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        print(f"Response body: {errh.response.text}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}. Is your FastAPI server running at {API_URL}?")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")

if __name__ == "__main__":
    # Test with a "normal-like" behavior
    normal_data = generate_sample_behavior_data(user_email="test_user_normal@example.com", is_anomalous_like=False)
    print("Sending 'Normal' Behavior Data:")
    send_test_data(normal_data)

    time.sleep(1) # Small pause

    # Test with an "anomalous-like" behavior
    anomalous_data = generate_sample_behavior_data(user_email="test_user_anomaly@example.com", is_anomalous_like=True)
    print("\nSending 'Anomalous' Behavior Data:")
    send_test_data(anomalous_data)

    print("\n--- Test Complete ---")
    print(f"Check your '{BEHAVIOR_LOG_FILE}' file for logged raw data.")
    print("You can also visit http://localhost:8000/api/behavior_logs in your browser (if your server is running) to see the stored logs.")