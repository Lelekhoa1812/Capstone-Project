#!/bin/bash

HOME_SSID="[REDACTED]"
POST_ENDPOINT="https://binkhoale1812-obd-logger.hf.space/upload-csv"
POLL_INTERVAL=10 # seconds

# Poll for available SSIDs
scan_wifi() {
    nmcli -t -f SSID dev wifi | grep -Fx "$HOME_SSID"
}

# Connect to the home WiFi
connect_home_wifi() {
    echo "[INFO] Connecting to $HOME_SSID..."
    nmcli dev wifi connect "$HOME_SSID" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "[INFO] Connected to $HOME_SSID."
        return 0
    else
        echo "[ERROR] Failed to connect to $HOME_SSID."
        return 1
    fi
}

# Makes a POST request to $POST_ENDPOINT with the CSVs
upload_csv_files() {
    for file in ./*.csv; do
        [ -e "$file" ] || continue
        echo "[INFO] Uploading $file..."
        response=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
            -F "file=@${file}" \
            "$POST_ENDPOINT")
        
        if [ "$response" = "200" ]; then
            echo "[INFO] Successfully uploaded $file."
        else
            echo "[ERROR] Failed to upload $file. HTTP status: $response"
        fi
    done
}


echo "[INFO] Starting WiFi polling..."
while true; do
    if scan_wifi; then
        echo "[INFO] Home SSID detected."
        if connect_home_wifi; then
            upload_csv_files
            break  # Exit after successful upload
        fi
    else
        echo "[INFO] Home SSID not found. Retrying in $POLL_INTERVAL seconds..."
    fi
    sleep "$POLL_INTERVAL"
done
