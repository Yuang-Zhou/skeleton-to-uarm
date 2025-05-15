"""
uarm_control_tester.py

This script provides functionalities to connect to a uArm Swift Pro,
perform basic diagnostic checks, execute a series of predefined test movements,
and ensure a clean disconnection. It serves as a foundational test for
verifying hardware communication and control via the uarm-python-sdk.
"""
import os
import sys
import time
from typing import Optional

# --- Constants ---
# Define default speeds for different types of uArm movements.
# These can be adjusted as needed.
DEFAULT_CARTESIAN_SPEED_MMPM = 3000  # mm/min for set_position
DEFAULT_POLAR_SPEED_MMPS = 50  # mm/s for set_polar (Note: SDK might interpret this differently)
DEFAULT_WRIST_SPEED_DEGPM = 600  # degrees/min for set_wrist

# Predefined points for Cartesian movement tests (in mm)
CARTESIAN_TEST_POINTS = [
    {"x": 180, "y": 0, "z": 150, "label": "Center Front"},
    {"x": 150, "y": 70, "z": 100, "label": "Front Left"},
    {"x": 150, "y": -70, "z": 130, "label": "Front Right"},
    {"x": 220, "y": 0, "z": 70, "label": "Low Center Front"},
]

# Predefined angles for wrist rotation tests (in degrees)
WRIST_TEST_ANGLES = [30, 60, 90, 120, 150, 90]

# Timeout for uArm readiness and individual movements
UARM_READY_TIMEOUT_S = 15
MOVEMENT_TIMEOUT_S = 10
RESET_TIMEOUT_S = 15

# --- Path Setup for uarm.wrapper ---
# This section attempts to make the 'uarm' package from the SDK importable.
# It assumes a common project structure where 'uarm-python-sdk/uArm-Python-SDK-2.0/'
# might be a subdirectory relative to this script, or the package is installed.
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to find the SDK directory, assuming it might be a sibling or in a parent's subdirectory
    # This is a common structure if you clone the SDK into your project.
    # Adjust these paths if your SDK is located elsewhere or installed globally.
    potential_sdk_paths = [
        os.path.join(current_script_dir, "uarm-python-sdk", "uArm-Python-SDK-2.0"),  # If SDK is a subfolder
        os.path.join(os.path.dirname(current_script_dir), "uarm-python-sdk", "uArm-Python-SDK-2.0"),
        # If script is in 'scripts' and SDK is sibling
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
    print(
        "Please ensure the uarm-python-sdk is correctly placed or installed, and the path to the 'uarm' package is discoverable.")
    print("Consider installing the SDK using its setup.py or adjusting the path setup in this script.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during SDK path setup or import: {e}")
    sys.exit(1)


class UArmController:
    """
    A controller class for managing the connection and basic operations
    of a uArm Swift Pro.
    """

    def __init__(self, port: Optional[str] = None):
        """
        Initializes the UArmController.

        Args:
            port (Optional[str]): The serial port to connect to.
                                  If None, the SDK will attempt auto-detection.
        """
        self.swift: Optional[SwiftAPI] = None
        self.port: Optional[str] = port
        self.is_connected: bool = False

    def connect(self) -> bool:
        """
        Establishes a connection to the uArm.

        Returns:
            bool: True if connection was successful and arm is ready, False otherwise.
        """
        if self.is_connected:
            print("Info: uArm is already connected.")
            return True

        print(f"Attempting to connect to uArm on port: {self.port if self.port else 'Auto-detect'}...")
        try:
            self.swift = SwiftAPI(port=self.port)

            print("Waiting for uArm to become ready...")
            if self.swift.waiting_ready(timeout=UARM_READY_TIMEOUT_S):
                print("uArm is ready.")
                self.is_connected = True
            else:
                print(f"Warning: uArm did not report ready within {UARM_READY_TIMEOUT_S} seconds.")
                # It might still be partially usable, but proceed with caution.
                # self.swift.disconnect() # Optionally disconnect if not fully ready
                # self.swift = None
                # return False
                self.is_connected = True  # Let's assume it might still work for some commands

            self._log_device_info()
            self._set_initial_mode()
            return self.is_connected

        except Exception as e:
            print(f"Error: Failed to connect to uArm: {e}")
            print("Troubleshooting tips:")
            print("  1. Ensure uArm is powered on and connected via USB.")
            print("  2. If specifying a port, verify it's correct (e.g., '/dev/ttyUSB0', 'COM3').")
            print("  3. Ensure no other software (e.g., uArmStudio) is using the arm.")
            self.swift = None
            self.is_connected = False
            return False

    def _log_device_info(self):
        """Logs basic device information if connected."""
        if not self.swift:
            return
        device_info = self.swift.get_device_info()
        if device_info:
            print("\n--- uArm Device Information ---")
            for key, value in device_info.items():
                print(f"  {key}: {value}")
            print("-------------------------------")
        else:
            print("Warning: Could not retrieve uArm device info after connection.")

    def _set_initial_mode(self, target_mode: int = 0):
        """Sets the uArm to a specified initial operating mode."""
        if not self.swift:
            return
        try:
            current_mode = self.swift.get_mode(wait=True)
            print(f"Current uArm mode: {current_mode}")
            if current_mode != target_mode:
                print(f"Setting uArm to Mode {target_mode} (Normal Mode)...")
                set_mode_result = self.swift.set_mode(target_mode, wait=True)
                if set_mode_result == target_mode or self.swift.get_mode(
                        wait=True) == target_mode:  # Check if mode was set
                    print(f"uArm mode successfully set to: {target_mode}")
                else:
                    print(f"Warning: Failed to set uArm mode to {target_mode}. Result: {set_mode_result}")
        except Exception as e:
            print(f"Warning: Could not get or set uArm mode: {e}")

    def disconnect(self):
        """Disconnects from the uArm if a connection is active."""
        if self.swift and self.is_connected:
            print("\nDisconnecting from uArm...")
            try:
                self.swift.disconnect()
                print("uArm disconnected successfully.")
            except Exception as e:
                print(f"Error during uArm disconnection: {e}")
        elif self.swift and not self.is_connected:
            print("Info: uArm instance exists but was not marked as connected. Attempting cleanup.")
            try:
                self.swift.disconnect()  # Try to disconnect anyway
            except:
                pass  # Ignore errors if already disconnected or in a bad state
        else:
            print("Info: No active uArm connection to disconnect.")
        self.is_connected = False
        self.swift = None

    def reset_arm(self, speed: int = DEFAULT_CARTESIAN_SPEED_MMPM, wait: bool = True) -> bool:
        """
        Resets the uArm to its default home position.

        Args:
            speed (int): The speed for the reset movement (mm/min).
            wait (bool): Whether to wait for the movement to complete.

        Returns:
            bool: True if the reset command was acknowledged as 'OK', False otherwise.
        """
        if not self.is_connected or not self.swift:
            print("Error: uArm not connected. Cannot reset.")
            return False
        print(f"\nResetting uArm to home position (speed: {speed} mm/min)...")
        result = self.swift.reset(speed=speed, wait=wait, timeout=RESET_TIMEOUT_S)
        if result == 'OK':
            print(f"Reset successful. Current position: {self.swift.get_position(wait=True)}")
            return True
        else:
            print(f"Reset command failed or timed out. Result: {result}")
            return False

    def move_to_cartesian(self, x: float, y: float, z: float,
                          speed: int = DEFAULT_CARTESIAN_SPEED_MMPM,
                          label: str = "", wait: bool = True) -> bool:
        """
        Moves the uArm end-effector to a specified Cartesian coordinate.

        Args:
            x, y, z (float): Target coordinates in mm.
            speed (int): Movement speed in mm/min.
            label (str): An optional label for the movement for logging.
            wait (bool): Whether to wait for the movement to complete.

        Returns:
            bool: True if the command was acknowledged as 'OK', False otherwise.
        """
        if not self.is_connected or not self.swift:
            print("Error: uArm not connected. Cannot move.")
            return False

        log_label = f" {label}" if label else ""
        print(f"Moving to{log_label}: (X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm) at {speed} mm/min...")

        result = self.swift.set_position(x=x, y=y, z=z, speed=speed, wait=wait, timeout=MOVEMENT_TIMEOUT_S)

        if result == 'OK':
            if wait:  # Only query position if we waited for the move
                print(f"  Movement to{log_label} successful. Current position: {self.swift.get_position(wait=True)}")
            else:
                print(f"  Movement command to{log_label} sent.")
            return True
        else:
            print(f"  Movement to{log_label} failed or timed out. Result: {result}")
            return False

    def set_wrist_angle(self, angle: float, speed: int = DEFAULT_WRIST_SPEED_DEGPM, wait: bool = True) -> bool:
        """
        Sets the uArm wrist servo (Servo 3) to a specified angle.

        Args:
            angle (float): Target angle in degrees (typically 0-180).
            speed (int): Movement speed in degrees/min.
            wait (bool): Whether to wait for the movement to complete.

        Returns:
            bool: True if the command was acknowledged as 'OK', False otherwise.
        """
        if not self.is_connected or not self.swift:
            print("Error: uArm not connected. Cannot set wrist angle.")
            return False

        print(f"Setting wrist angle to: {angle:.1f} degrees at {speed} deg/min...")
        result = self.swift.set_wrist(angle=angle, speed=speed, wait=wait,
                                      timeout=MOVEMENT_TIMEOUT_S / 2)  # Wrist moves faster

        if result == 'OK':
            print(f"  Wrist angle successfully set to {angle:.1f} degrees.")
            # One could call self.swift.get_servo_angle(servo_id=3, wait=True) to verify if needed.
            return True
        else:
            print(f"  Setting wrist angle to {angle:.1f} failed or timed out. Result: {result}")
            return False

    def run_movement_test_sequence(self):
        """Executes a predefined sequence of test movements."""
        if not self.is_connected:
            print("Cannot run test sequence: uArm not connected.")
            return

        print("\n--- Starting uArm Movement Test Sequence ---")

        # 1. Initial Reset
        if not self.reset_arm(wait=True):
            print("Aborting test sequence due to reset failure.")
            return
        time.sleep(1)  # Pause after reset

        # 2. Cartesian Movements
        print("\nTesting Cartesian movements...")
        for i, point_info in enumerate(CARTESIAN_TEST_POINTS):
            success = self.move_to_cartesian(
                x=point_info['x'], y=point_info['y'], z=point_info['z'],
                label=point_info['label'], wait=True
            )
            if not success:
                print(f"  Skipping remaining Cartesian tests due to failure at {point_info['label']}.")
                break
            time.sleep(1.0)  # Pause between points

        # 3. Wrist Rotation Test
        print("\nTesting wrist rotations...")
        for angle in WRIST_TEST_ANGLES:
            success = self.set_wrist_angle(angle=angle, wait=True)
            if not success:
                print(f"  Skipping remaining wrist tests due to failure at angle {angle}.")
                break
            time.sleep(0.75)  # Pause between wrist movements

        # 4. Final Reset
        print("\nReturning to home position to conclude tests.")
        self.reset_arm(wait=True)

        print("\n--- uArm Movement Test Sequence Completed ---")


def main():
    """
    Main function to initialize, test, and shutdown the uArm.
    """
    print("--- uArm Swift Pro Control Test Script ---")

    # --- User Configuration ---
    # IMPORTANT: Specify the correct serial port for your uArm.
    # Set to None for auto-detection (less reliable if multiple serial devices are present).
    # Examples: '/dev/ttyUSB0' (Linux), '/dev/cu.usbmodemXXXX' (macOS), 'COM3' (Windows)
    uarm_port_setting: Optional[str] = '/dev/cu.usbmodem144201'  # MODIFY THIS if auto-detect fails
    # --- End User Configuration ---

    controller = UArmController(port=uarm_port_setting)

    try:
        if controller.connect():
            controller.run_movement_test_sequence()
        else:
            print("Failed to connect to uArm. Please check connections and settings.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()  # Print detailed traceback for debugging
    finally:
        # Ensure the arm is always disconnected properly
        print("Ensuring uArm is disconnected in finally block...")
        controller.disconnect()
        print("\n--- Program Finished ---")


if __name__ == '__main__':
    main()
