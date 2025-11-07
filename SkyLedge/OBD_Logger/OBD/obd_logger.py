import obd
import time
import datetime
import csv
import os
import shutil
import subprocess
import sys
import select

try:
    from logging_wrapper import auto_score_on_completion
    SCORING_AVAILABLE = True
    print("Auto-scoring module loaded")
except ImportError:
    SCORING_AVAILABLE = False
    print("Auto-scoring module not found - scoring will be skipped")


CRITICAL_FUEL_PIDS = [
    obd.commands.RPM,           
    obd.commands.SPEED,           
    obd.commands.THROTTLE_POS,  
    obd.commands.MAF,    
]       

SECONDARY_FUEL_PIDS = [
    obd.commands.ENGINE_LOAD,      
    obd.commands.INTAKE_PRESSURE,  
]

TERTIARY_FUEL_PIDS = [
    obd.commands.SHORT_FUEL_TRIM_1,  
    obd.commands.SHORT_FUEL_TRIM_2,   
    obd.commands.LONG_FUEL_TRIM_1,   
    obd.commands.LONG_FUEL_TRIM_2,  
]

HIGH_FREQUENCY_PIDS = CRITICAL_FUEL_PIDS
LOW_FREQUENCY_PIDS_POOL = SECONDARY_FUEL_PIDS + TERTIARY_FUEL_PIDS

ALL_PIDS_TO_LOG = HIGH_FREQUENCY_PIDS + LOW_FREQUENCY_PIDS_POOL

CSV_FILENAME_BASE = "obd_data_log" 
LOGS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
FUEL_LOGS_DIR = os.path.join(LOGS_BASE_DIR, "FuelLogs")
ANALYSED_LOGS_DIR = os.path.join(LOGS_BASE_DIR, "analysedLogsAutomated")

SCORED_LOGS_DIR = os.path.join(LOGS_BASE_DIR, "ScoredLogs")
ORIGINAL_CSV_DIR = FUEL_LOGS_DIR

def get_pid_value(connection, pid_command):
    """Queries a PID and returns its value"""
    try:
        response = connection.query(pid_command, force=True)
        if response.is_null() or response.value is None:
            return None
        if hasattr(response.value, 'magnitude'):
            return response.value.magnitude
        return response.value
    except Exception as e:
        print(f"Error querying {pid_command.name}: {e}") 
        return None

ef calculate_fuel_metrics(csv_path):
    """Calculate fuel consumption and efficiency from MAF and SPEED data."""
    try:
        df = pd.read_csv(csv_path)
        
        # Constants
        AFR = 14.7  # Air-Fuel Ratio for petrol
        FUEL_DENSITY = 737  # gg/ for petrol
        
        # Calculate time delta between rows (in seconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds()
        df.loc[0, 'time_delta'] = 0  # First row has no previous row
        
        # Calculate instantaneous fuel rate (L/hr) from MAF
        df['fuel_rate_L_per_hr'] = (df['MAF'] * 3600) / (AFR * FUEL_DENSITY)
        
        # Calculate fuel used in this time interval (L)
        df['fuel_used_interval'] = (df['fuel_rate_L_per_hr'] / 3600) * df['time_delta']
        
        # Calculate distance traveled in this interval (km)
        df['distance_interval'] = (df['SPEED'] / 3600) * df['time_delta']
        
        # Calculate cumulative values
        df['Fuel_Used'] = df['fuel_used_interval'].cumsum()
        df['Distance'] = df['distance_interval'].cumsum()
        
        # Calculate fuel efficiency (L/100km)
        df['Fuel_efficiency (L/100km)'] = np.where(
            df['Distance'] > 0,
            (df['Fuel_Used'] / df['Distance']) * 100,
            0
        )
        
        df['Fuel_Used'] = df['Fuel_Used'].round(3)
        df['Distance'] = df['Distance'].round(2)
        df['Fuel_efficiency (L/100km)'] = df['Fuel_efficiency (L/100km)'].round(2)
        
        # Drop intermediate calculation columns
        df = df.drop(columns=['time_delta', 'fuel_rate_L_per_hr', 
                              'fuel_used_interval', 'distance_interval'])
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        
        # Print summary
        total_fuel = df['Fuel_Used'].iloc[-1]
        total_distance = df['Distance'].iloc[-1]
        avg_efficiency = df['Fuel_efficiency (L/100km)'].iloc[-1]
        
        print(f"Total Fuel Used: {total_fuel:.3f} L")
        print(f"Total Distance: {total_distance:.2f} km")
        print(f"Average Efficiency: {avg_efficiency:.2f} L/100km")
        
        return csv_path
        
    except Exception as e:
        print(f"Error calculating fuel metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

    
def perform_logging_session(connection):
    """Perform a single logging session with an existing OBD connection."""
    print(f"\nStarting new fuel efficiency logging session")
    print("Commands:")
    print("  - Type 'next' and press Enter to finish this drive and start a new one")
    print("  - Type 'quit' and press Enter to stop all logging")

    
    CRITICAL_PID_INTERVAL = 0.65
    SECONDARY_PID_INTERVAL = 2.0
    TERTIARY_PID_INTERVAL = 5.0
    
    last_critical_poll_time = time.monotonic() - CRITICAL_PID_INTERVAL
    last_secondary_poll_time = time.monotonic() - SECONDARY_PID_INTERVAL
    last_tertiary_poll_time = time.monotonic() - TERTIARY_PID_INTERVAL
    
    BASE_LOG_INTERVAL = CRITICAL_PID_INTERVAL
    
    current_pid_values = {pid.name: '' for pid in ALL_PIDS_TO_LOG} 

    for dir_path in [FUEL_LOGS_DIR, ANALYSED_LOGS_DIR, SCORED_LOGS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Ensured directory exists: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}. Attempting to use current directory.")
            if dir_path == FUEL_LOGS_DIR:
                 print("Cannot create fuel log directory. Exiting.")
                 return None 

    current_session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_name_only = f"{CSV_FILENAME_BASE}_{current_session_timestamp}.csv"
    original_csv_filepath = os.path.join(ORIGINAL_CSV_DIR, csv_file_name_only)

    try:
        if not connection or not connection.is_connected():
            print("OBD connection not available")
            return None, "quit"
            
        print(f"Using existing OBD connection: {connection.port_name()}")

        # Creating initial full PID sample to have fully populated rows from beginning 
        print("\nPerforming initial full PID sample...")
        initial_log_entry = {
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        print("Polling initial Critical Fuel PIDs...")
        for pid_command in CRITICAL_FUEL_PIDS:
            try:
                value = get_pid_value(connection, pid_command)
                current_pid_values[pid_command.name] = value if value is not None else ''
                initial_log_entry[pid_command.name] = current_pid_values[pid_command.name]
            except Exception as e:
                print(f"Warning: Failed to get {pid_command.name}: {e}")
                current_pid_values[pid_command.name] = ''
                initial_log_entry[pid_command.name] = ''

        print("Polling initial Secondary Fuel PIDs...")
        for pid_command in SECONDARY_FUEL_PIDS:
            try:
                value = get_pid_value(connection, pid_command)
                current_pid_values[pid_command.name] = value if value is not None else ''
                initial_log_entry[pid_command.name] = current_pid_values[pid_command.name]
            except Exception as e:
                print(f"Warning: Failed to get {pid_command.name}: {e}")
                current_pid_values[pid_command.name] = ''
                initial_log_entry[pid_command.name] = ''
            
        print("Polling initial Tertiary Fuel PIDs...")
        for pid_command in TERTIARY_FUEL_PIDS:
            try:
                value = get_pid_value(connection, pid_command)
                current_pid_values[pid_command.name] = value if value is not None else ''
                initial_log_entry[pid_command.name] = current_pid_values[pid_command.name]
            except Exception as e:
                print(f"Warning: Failed to get {pid_command.name}: {e}")
                current_pid_values[pid_command.name] = ''
                initial_log_entry[pid_command.name] = ''

        for pid_obj in ALL_PIDS_TO_LOG:
            if pid_obj.name not in initial_log_entry:
                initial_log_entry[pid_obj.name] = ''

        # Empty driving style and fuel columns
        initial_log_entry['Driving_style'] = ''
        initial_log_entry['Fuel_efficiency (L/100km)'] = ''
        initial_log_entry['Distance'] = ''
        initial_log_entry['Fuel_Used'] = ''
        initial_log_entry['Route'] = ''

    except Exception as e:
        print(f"An error occurred during connection or initial PID sample: {e}")
        if connection and connection.is_connected():
            connection.close()
        return None, "quit"

    file_exists = os.path.isfile(original_csv_filepath)
    try:
        with open(original_csv_filepath, 'a', newline='') as csvfile:
            header_names = ['timestamp'] + [pid.name for pid in ALL_PIDS_TO_LOG] + ['Driving_style', 'Fuel_efficiency (L/100km)', 'Distance', 'Fuel_Used', 'Route']

            writer = csv.DictWriter(csvfile, fieldnames=header_names)

            if not file_exists or os.path.getsize(original_csv_filepath) == 0:
                writer.writeheader()
                print(f"Created new CSV file: {original_csv_filepath} with headers: {header_names}")

            if initial_log_entry: 
                writer.writerow(initial_log_entry)
                csvfile.flush()
                print(f"Logged initial full sample with all fuel efficiency PIDs.")
            
            log_count = 0
            user_stop_requested = False

            print(f"Started logging")
            
            while not user_stop_requested:
                if log_count % 100 == 0 and log_count > 0:
                    print(f"Debug: Main loop running, iteration {log_count}")
                
                # Check for non-blocking input
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    user_command = sys.stdin.readline().strip().lower()
                    if user_command == "next":
                        print("\nUser typed 'next'. Finishing current drive...")
                        user_stop_requested = True
                        break
                    elif user_command == "quit":
                        print("\nUser typed 'quit'. Stopping all logging...")
                        user_stop_requested = True
                        return original_csv_filepath, "quit"
                    else:
                        print(f"Input detected: '{user_command}'. Type 'next' or 'quit'.", end='\r') 

                loop_start_time = time.monotonic()
                current_datetime = datetime.datetime.now()
                timestamp_iso = current_datetime.isoformat()
                
                critical_reads = 0
                secondary_reads = 0
                tertiary_reads = 0
                
                # Always poll critical PIDs (highest frequency)
                if (time.monotonic() - last_critical_poll_time) >= CRITICAL_PID_INTERVAL:
                    if not connection or not connection.is_connected():
                        print("\nOBD connection lost during logging. Ending session.")
                        user_stop_requested = True
                        break
                    
                    for pid_command in CRITICAL_FUEL_PIDS:
                        value = get_pid_value(connection, pid_command)
                        current_pid_values[pid_command.name] = value if value is not None else ''
                        if value is not None:
                            critical_reads += 1
                    last_critical_poll_time = time.monotonic()
                
                # Poll secondary PIDs at medium frequency
                if (time.monotonic() - last_secondary_poll_time) >= SECONDARY_PID_INTERVAL:
                    if not connection or not connection.is_connected():
                        print("\nOBD connection lost during logging. Ending session.")
                        user_stop_requested = True
                        break
                    
                    for pid_command in SECONDARY_FUEL_PIDS:
                        value = get_pid_value(connection, pid_command)
                        current_pid_values[pid_command.name] = value if value is not None else ''
                        if value is not None:
                            secondary_reads += 1
                    last_secondary_poll_time = time.monotonic()
                
                # Poll tertiary PIDs at low frequency
                if (time.monotonic() - last_tertiary_poll_time) >= TERTIARY_PID_INTERVAL:
                    if not connection or not connection.is_connected():
                        print("\nOBD connection lost during logging. Ending session.")
                        user_stop_requested = True
                        break
                    
                    for pid_command in TERTIARY_FUEL_PIDS:
                        value = get_pid_value(connection, pid_command)
                        current_pid_values[pid_command.name] = value if value is not None else ''
                        if value is not None:
                            tertiary_reads += 1
                    last_tertiary_poll_time = time.monotonic()

                final_log_entry = {
                    'timestamp': timestamp_iso
                }
                for pid_obj in ALL_PIDS_TO_LOG:
                     final_log_entry[pid_obj.name] = current_pid_values.get(pid_obj.name, '')

                final_log_entry['Driving_style'] = ''
                final_log_entry['Fuel_efficiency (L/100km)'] = ''
                final_log_entry['Distance'] = ''
                final_log_entry['Fuel_Used'] = ''
                final_log_entry['Route'] = ''

                writer.writerow(final_log_entry)
                csvfile.flush()  

                log_count += 1
                if log_count % 10 == 0: 
                    status_msg = f"Entry {log_count} - Critical: {critical_reads}/{len(CRITICAL_FUEL_PIDS)}"
                    if secondary_reads > 0:
                        status_msg += f" Secondary: {secondary_reads}/{len(SECONDARY_FUEL_PIDS)}"
                    if tertiary_reads > 0:
                        status_msg += f" Tertiary: {tertiary_reads}/{len(TERTIARY_FUEL_PIDS)}"
                    print(status_msg + " " * 20, end='\r')
                
                elapsed_time_in_loop = time.monotonic() - loop_start_time
                sleep_duration = max(0, BASE_LOG_INTERVAL - elapsed_time_in_loop)
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\nStopping data logging due to user interruption (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred during logging: {e}")
    finally:
        print(" " * 100, end='\r') 
        print(f"Drive completed - data saved to: {os.path.basename(original_csv_filepath)}")

    return original_csv_filepath, "next"

def run_scorer_on_csv(original_csv_path):
    if not SCORING_AVAILABLE:
        print("Scoring module not available, skipping aggressiveness scoring")
        return None
    
    if not original_csv_path or not os.path.exists(original_csv_path):
        print(f"Error: Original CSV not found for scoring: {original_csv_path}")
        return None
    
    print(f"\nRunning aggressiveness scorer...")
    
    original_filename = os.path.basename(original_csv_path)
    base, ext = os.path.splitext(original_filename)
    
    try:
        # Import and configure the scorer
        from driving_aggressiveness_scorer import DrivingAggressivenessScorer
        import json
        
        # Initialize scorer with bounds file in logs directory
        bounds_file = os.path.join(LOGS_BASE_DIR, 'obd_bounds.json')
        scorer = DrivingAggressivenessScorer(bounds_file=bounds_file)
        
        # Run analysis
        df_scored, results = scorer.analyze_drive(str(original_csv_path), update_bounds=True)
        
        df_scored['drive_score'] = results['final_score']
        
        # Save scored CSV to ScoredLogs directory
        scored_csv_path = os.path.join(SCORED_LOGS_DIR, f"{base}_scored{ext}")
        df_scored.to_csv(scored_csv_path, index=False)
        print(f"Scored CSV saved: {os.path.basename(scored_csv_path)}")
        
        # Save summary JSON to ScoredLogs directory
        summary_json_path = os.path.join(SCORED_LOGS_DIR, f"{base}_score_summary.json")
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'original_file': str(original_csv_path),
            'scored_file': str(scored_csv_path),
            'results': results
        }
        
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Score summary saved: {os.path.basename(summary_json_path)}")
        
        try:
            from visualiseScorer import visualize_drive
            visualization_path = os.path.join(SCORED_LOGS_DIR, f"{base}_visualization.png")
            visualize_drive(df_scored, results, save_path=visualization_path)
            print(f"Visualization saved: {os.path.basename(visualization_path)}")
        except Exception as viz_error:
            print(f"Warning: Could not generate visualization: {viz_error}")
        
        # Print quick summary
        print(f"Drive Score: {results['final_score']:.1f}/100")
        
        return scored_csv_path
        
    except Exception as e:
        print(f"Error running scorer: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_obd_connection():
    """Initialize OBD connection once for multiple sessions."""
    connection = None
    
    try:
        
        print("Attempting to connect via socat PTY /dev/ttys006...")
        connection = obd.OBD("/dev/ttys002", fast=True, timeout=30)
            
        if not connection.is_connected():
            print("Failed to connect to OBD-II adapter.")
            print(f"Connection status: {connection.status()}")
            return None
        
        print(f"Successfully connected to OBD-II adapter: {connection.port_name()}")
        print(f"Adapter status: {connection.status()}")
        return connection
        
    except Exception as e:
        print(f"An error occurred during OBD connection: {e}")
        return None


def main():
    print("Fuel Efficiency OBD Logger - Multi-Session Mode")
    if SCORING_AVAILABLE:
        print("Aggressiveness scoring enabled")
    print("=" * 50)
    
    # Initialize OBD connection once
    connection = initialize_obd_connection()
    if not connection:
        print("Could not establish OBD connection. Exiting.")
        return
    
    session_count = 0
    logged_files = []
    
    try:
        while True:
            session_count += 1
            print(f"\n📊 Session {session_count} ready to start")
            
            # Check if connection is still available before starting new session
            if not connection or not connection.is_connected():
                print("OBD connection not available. Attempting to reconnect...")
                connection = initialize_obd_connection()
                if not connection:
                    print("Could not re-establish OBD connection. Exiting.")
                    break
            
            result = perform_logging_session(connection)
            
            if isinstance(result, tuple):
                csv_file, command = result
            else:
                csv_file, command = result, "quit"
            
            # Handle the result
            if csv_file and os.path.exists(csv_file):
                try:
                    with open(csv_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # More than just the header
                            logged_files.append(csv_file)
                            print(f"Drive {session_count} saved: {os.path.basename(csv_file)}")

                            calculate_fuel_metrics(csv_file)

                            print(f"\nStarting aggressiveness scoring for drive {session_count}...")
                            scored_file = run_scorer_on_csv(csv_file)
                            if scored_file:
                                print(f"Aggressiveness scoring complete for drive {session_count}")
                            else:
                                print(f"Aggressiveness scoring failed for drive {session_count}, but drive data is still saved")
                            
                        else:
                            print(f"⚠️ Drive {session_count} had no data, skipping analysis")
                            os.remove(csv_file)
                except Exception as e:
                    print(f"Error checking file {csv_file}: {e}")
            
            # Check if user wants to quit
            if command == "quit":
                print("\nStopping all logging as requested")
                break
            
            # Otherwise continue to next session
            print(f"\n Ready for next drive (Session {session_count + 1})")
    
    except KeyboardInterrupt:
        print("\n Logging stopped by user (Ctrl+C)")
    
    finally:
        if connection and connection.is_connected():
            print("Closing OBD-II connection...")
            connection.close()
        
        print("\n" + "=" * 50)
        print(f"📈 LOGGING SUMMARY")
        print(f"Total drives logged: {len(logged_files)}")
        if logged_files:
            print("📁 Files saved to:")
            print("   - Raw logs:      logs/FuelLogs/")
            if SCORING_AVAILABLE:
                print("   - Scored logs:   logs/ScoredLogs/")
            print("\n📝 Files created:")
            for file in logged_files:
                print(f"  - {os.path.basename(file)}")
        print("=" * 50)


if __name__ == "__main__":
    main()