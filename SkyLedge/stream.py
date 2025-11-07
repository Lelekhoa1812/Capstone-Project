import requests, random, datetime, time
from pynput import keyboard

# API Handler
API_URL = "https://binkhoale1812-obd-logger.hf.space/ingest"
DRIVING_STYLES = ["aggressive", "passive", "normal"]
SESSION_TS = datetime.datetime.now().isoformat()

running = True  # Global flag to stop the stream

# Random data generation
def generate_fake_obd_data():
    return {
        "RPM": random.randint(800, 4000),
        "THROTTLE_POS": round(random.uniform(5, 80), 2),
        "SPEED": random.randint(0, 120),
        "FUEL_PRESSURE": random.randint(30, 70),
        "ENGINE_LOAD": round(random.uniform(10, 80), 2),
        "COOLANT_TEMP": random.randint(70, 110),
        "INTAKE_TEMP": random.randint(20, 60),
        "MAF": round(random.uniform(0.5, 10.0), 2)
    }

# Handshake with server
def send_control_signal(stage):
    requests.post(API_URL, json={
        "timestamp": SESSION_TS,
        "driving_style": "none",
        "data": {},
        "status": stage  # 'start' or 'end'
    })

# Key press listener
def on_press(key):
    global running
    try:
        if key.char.lower() == 'q':
            print("\n[Q] pressed — stopping stream.")
            running = False
            return False  # Stop listener
    except AttributeError:
        pass

# Simulate driver sensor logging
def simulate_logging():
    global running
    print("Start Streaming Session. Press [Q] to stop logging...")
    send_control_signal("start")
    # Key press listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    # Replicate each 0.2s
    interval = 0.2
    i = 0
    while running:
        payload = {
            "timestamp": SESSION_TS,
            "driving_style": random.choice(DRIVING_STYLES),
            "data": generate_fake_obd_data()
        }
        try:
            res = requests.post(API_URL, json=payload)
            print(f"[✓] Entry {i+1} sent: {payload['timestamp']}")
        except Exception as e:
            print(f"[!] Error: {e}")
        time.sleep(interval)
        i += 1

    send_control_signal("end")
    print("✅ Logging session ended.")


if __name__ == "__main__":
    simulate_logging()
