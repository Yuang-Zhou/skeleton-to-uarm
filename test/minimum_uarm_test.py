import os
import sys
import time

# --- Path Setup for uarm.wrapper ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_sdk_paths = [
        os.path.join(current_script_dir, "uarm-python-sdk", "uArm-Python-SDK-2.0"),
        os.path.join(os.path.dirname(current_script_dir), "uarm-python-sdk", "uArm-Python-SDK-2.0"),
    ]
    sdk_found_path = None
    for path_attempt in potential_sdk_paths:
        if os.path.isdir(os.path.join(path_attempt, "uarm")):
            sdk_found_path = path_attempt
            break
    if sdk_found_path and sdk_found_path not in sys.path:
        sys.path.insert(0, sdk_found_path)
        print(f"Info: Added '{sdk_found_path}' to sys.path for uarm.wrapper import.")
    from uarm.wrapper import SwiftAPI
    print("Successfully imported SwiftAPI from uarm.wrapper.")
except ImportError as e:
    print(f"Error: Failed to import SwiftAPI: {e}")
    sys.exit(1)
# --- End Path setup ---

def main():
    uarm_port = '/dev/cu.usbmodem144201'
    swift = None
    try:
        print(f"Attempting to connect to uArm on port: {uarm_port}...")

        swift = SwiftAPI(port=uarm_port,
                         enable_handle_thread=True,
                         enable_write_thread=False,
                         enable_handle_report_thread=False,
                         cmd_pend_size=5
                        )

        print("Attempting initial connection check with SwiftAPI constructor...")
        time.sleep(2) # Give some time for connection to establish

        if not swift.connected:
            print("Failed to connect to uArm (swift.connected is False).")
            return

        print("uArm connected. Getting device info...")
        device_info = swift.get_device_info(timeout=10)
        if device_info:
            print("Device Info:", device_info)
        else:
            print("Failed to get device info (returned None).")

        print(f"Current power status (property): {swift.power_status}")

        print("\nCalling waiting_ready()...")
        is_ready = swift.waiting_ready(timeout=20)
        print(f"waiting_ready() result: {is_ready}")
        print(f"Power status after waiting_ready: {swift.power_status}")

        # if not is_ready:
        # print("Warning: uArm did not report ready via waiting_ready(). Proceeding with caution.")

        print("\nGetting position BEFORE reset...")
        pos_before_reset = swift.get_position(wait=True, timeout=10)
        print(f"Position before reset: {pos_before_reset}")

        print("\nAttempting to set servo attach (part of reset logic)...")
        attach_result = swift.set_servo_attach(wait=True, timeout=10)
        print(f"set_servo_attach() result: {attach_result}")
        if attach_result != 'OK':
            print("Warning: set_servo_attach did not return 'OK'.")

        time.sleep(0.5) # Short pause

        print("\nAttempting to reset uArm (swift.reset)...")
        reset_result = swift.reset(wait=True, speed=5000, timeout=30)
        print(f"swift.reset() command result: {reset_result}")

        time.sleep(1) # Wait for physical movement to settle after reset attempt

        print("\nGetting position AFTER reset attempt...")
        pos_after_reset = swift.get_position(wait=True, timeout=10)
        print(f"Position after reset attempt: {pos_after_reset}")

        if reset_result == 'OK' and isinstance(pos_after_reset, list):
            print("Reset reported SUCCESSFUL by SDK!")
        elif isinstance(pos_after_reset, list) and \
             (pos_after_reset[0] > 190 and pos_after_reset[0] < 210 and \
              pos_after_reset[1] > -10 and pos_after_reset[1] < 10 and \
              pos_after_reset[2] > 140 and pos_after_reset[2] < 160):
            print("PHYSICAL reset seems to have occurred (position is near home), even if SDK reported: ", reset_result)
            print("This suggests a possible discrepancy in command acknowledgement for firmware 3.2.0.")
        else:
            print("Reset failed or position is not at home. Cannot proceed with further movements reliably.")


    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if swift and swift.connected:
            print("\nDisconnecting from uArm...")
            swift.disconnect()
            print("uArm disconnected.")
        print("Minimal test finished.")

if __name__ == '__main__':
    main()