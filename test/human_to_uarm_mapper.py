import os
import sys
import time
import numpy as np
import math
from typing import Optional, Tuple, List  # Removed Dict, will use lowercase dict

# --- Constants and Configuration ---
# Configuration for human pose data
NPZ_FILE_RELATIVE_PATH = os.path.join("../data", "00", "00_01_poses.npz")  # MODIFY THIS to your NPZ file
SKELETON_TYPE = 'smpl_24'
TRACKED_ARM = 'right'  # 'right' or 'left'

# Configuration for uArm
# !!! CRITICAL: Set your uArm's serial port here !!!
UARM_SERIAL_PORT = '/dev/cu.usbmodem144201'  # MODIFY THIS - This was from your previous output
INITIAL_UARM_RESET_SPEED = 3000  # mm/min
MOVEMENT_SPEED_MMPM = 6000  # mm/min for set_position
WRIST_SPEED_DEGPM = 1000  # deg/min for set_wrist

# Mapping Parameters
HUMAN_TO_UARM_SCALE_FACTOR = 150.0
UARM_SHOULDER_ORIGIN_OFFSET = np.array([150.0, 0.0, 100.0])
TARGET_COMMAND_FPS = 20

# --- Helper: Path Setup ---
try:
    _current_script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = _current_script_dir

    _src_path_hpstm = os.path.join(_project_root, "src")
    if os.path.isdir(_src_path_hpstm) and _src_path_hpstm not in sys.path:
        sys.path.insert(0, _src_path_hpstm)
        print(f"Info: Added HPSTM src path '{_src_path_hpstm}' to sys.path.")

    _sdk_base_dir = os.path.join(_project_root, "uarm-python-sdk")
    _sdk_actual_path = os.path.join(_sdk_base_dir, "uArm-Python-SDK-2.0")
    if os.path.isdir(os.path.join(_sdk_actual_path, "uarm")) and _sdk_actual_path not in sys.path:
        sys.path.insert(0, _sdk_actual_path)
        print(f"Info: Added uArm SDK path '{_sdk_actual_path}' to sys.path.")

    from kinematics.skeleton_utils import get_num_joints as get_total_joints_for_skeleton
    from uarm.wrapper import SwiftAPI

    # --- Skeleton Definitions (Simplified inline for this script) ---
    SMPL24_JOINT_MAPPING = {
        'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3, 'L_Knee': 4, 'R_Knee': 5,
        'Spine2': 6, 'L_Ankle': 7, 'R_Ankle': 8, 'Spine3': 9, 'L_Foot': 10, 'R_Foot': 11,
        'Neck': 12, 'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
        'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19,
        'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23
    }
    SMPL24_ARM_KEY_JOINTS = {
        'right': {'shoulder': SMPL24_JOINT_MAPPING['R_Shoulder'], 'elbow': SMPL24_JOINT_MAPPING['R_Elbow'],
                  'wrist': SMPL24_JOINT_MAPPING['R_Wrist']},
        'left': {'shoulder': SMPL24_JOINT_MAPPING['L_Shoulder'], 'elbow': SMPL24_JOINT_MAPPING['L_Elbow'],
                 'wrist': SMPL24_JOINT_MAPPING['L_Wrist']}
    }


    def load_pose_sequence_from_npz(npz_file_path: str, expected_key: str = 'poses_r3j') -> Optional[np.ndarray]:
        # (Implementation from previous version)
        if not os.path.exists(npz_file_path):
            print(f"Error: File not found at '{npz_file_path}'.")
            return None
        try:
            data = np.load(npz_file_path)
            if expected_key not in data:
                print(
                    f"Error: Expected key '{expected_key}' not found in '{npz_file_path}'. Available: {list(data.keys())}")
                return None
            pose_sequence = data[expected_key]
            if pose_sequence.ndim != 3 or pose_sequence.shape[2] != 3:
                print(
                    f"Error: Data for key '{expected_key}' has unexpected shape: {pose_sequence.shape}. Expected (frames, joints, 3).")
                return None
            return pose_sequence.astype(np.float32)
        except Exception as e:
            print(f"Error loading .npz file '{npz_file_path}': {e}")
            return None


    # Use lowercase 'dict' for type hint, which is standard for Python 3.9+
    def get_arm_joint_indices(skeleton_type: str = 'smpl_24', arm_to_track: str = 'right') -> Optional[dict[str, int]]:
        # (Implementation from previous version)
        if skeleton_type.lower() == 'smpl_24':
            return SMPL24_ARM_KEY_JOINTS.get(arm_to_track.lower())
        print(f"Error: Key joint indices for skeleton type '{skeleton_type}' are not defined.")
        return None


    print("Successfully imported required modules.")
except ImportError as e:
    print(f"Critical Error: Failed to import a required module: {e}")
    sys.exit(1)


# --- End Path Setup & Helper Function Definitions ---


class UArmMimicController:
    """
    Controls the uArm to mimic human arm movements from pose data.
    """

    def __init__(self, port: Optional[str] = UARM_SERIAL_PORT):  # UARM_SERIAL_PORT should be globally defined
        self.swift: Optional[SwiftAPI] = None
        self.port: Optional[str] = port
        self.is_connected: bool = False
        self.key_joint_indices: Optional[dict[str, int]] = None
        self.human_pose_sequence: Optional[np.ndarray] = None
        self.initial_human_shoulder_pos: Optional[np.ndarray] = None

    def connect_uarm(self) -> bool:
        """
        Initializes and connects to the uArm.
        Checks power status after waiting_ready and confirms physical reset.
        """
        print(f"Attempting to connect to uArm on port: {self.port if self.port else 'Auto-detect'}...")
        try:
            self.swift = SwiftAPI(port=self.port)
            # Give a moment for the serial connection to establish before SDK commands
            time.sleep(1.5)  # Increased sleep slightly

            if not self.swift.connected:  # Check basic serial connection first
                print("Error: SwiftAPI failed to establish a serial connection.")
                self.swift = None  # Ensure swift is None if connection failed at constructor
                return False

            print("Serial connection established. Calling waiting_ready()...")
            # Call waiting_ready, but don't solely rely on its direct boolean return value
            # for firmware 3.2.0, as it might return None on success.
            _ = self.swift.waiting_ready(timeout=20)  # Increased timeout, store return if needed for debug

            # CRITICAL CHECK: Verify power status after waiting_ready
            if self.swift.power_status:
                print("uArm power status is ON after waiting_ready.")
                self.is_connected = True  # Tentatively mark as connected
            else:
                print(
                    "Warning: uArm power status is OFF after waiting_ready. Attempting to proceed with caution but this is unusual.")
                # Forcing is_connected to true here is risky if power_status is false.
                # However, if minimal_test showed it worked despite this, we can try.
                # A better approach might be to fail here if power_status isn't True.
                # For now, let's see if it can recover with subsequent commands.
                # If minimal_test showed power_status became True, this path shouldn't be hit often.
                self.is_connected = True  # Let's try to proceed based on minimal_test behavior

            if not self.is_connected:  # If still not marked connected (e.g. if we added stricter checks above)
                print("uArm connection failed or power status remained off. Disconnecting.")
                if self.swift: self.swift.disconnect()
                self.swift = None
                return False

            # Proceed with device info, mode setting, and reset
            device_info = self.swift.get_device_info(timeout=10)  # Increased timeout
            if device_info:
                print(f"Device Info: {device_info}")
                # Specifically check if device_type is now populated
                if not device_info.get('device_type') or not device_info.get('hardware_version'):
                    print(
                        "Warning: Device type or hardware version is still None in device_info. Communication might be unstable.")
            else:
                print("Warning: Failed to get complete device info (returned None or incomplete).")

            current_mode = self.swift.get_mode(wait=True, timeout=10)  # Increased timeout
            print(f"Current uArm mode: {current_mode}")
            if current_mode != 0 and current_mode is not None:  # Check for None if get_mode fails
                print(f"Setting uArm to Mode 0 (Normal Mode)...")
                set_mode_result = self.swift.set_mode(0, wait=True, timeout=10)  # Increased timeout
                # Check actual mode after setting
                final_mode = self.swift.get_mode(wait=True, timeout=5)
                print(f"set_mode(0) SDK reported: {set_mode_result}. Actual mode after set: {final_mode}")
                if final_mode != 0:
                    print("Warning: Failed to reliably set uArm mode to 0.")
            elif current_mode is None:
                print("Warning: Could not retrieve current uArm mode.")

            print("Resetting uArm on connect...")
            # The reset command itself in Swift class doesn't return a value.
            # The SwiftAPI wrapper for reset also just calls the Swift class's reset.
            # So, a 'None' return from swift.reset() is expected from the SDK's structure.
            # We rely on the physical action and subsequent position check.
            self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=25)  # Increased timeout
            print(f"swift.reset() command sent (expected SDK return: None or similar for this version).")

            time.sleep(1.0)  # Allow time for physical reset to complete

            pos_after_reset = self.swift.get_position(wait=True, timeout=10)  # Increased timeout
            print(f"uArm position after reset attempt: {pos_after_reset}")

            if isinstance(pos_after_reset, list) and \
                    (190 < pos_after_reset[0] < 210 and \
                     -10 < pos_after_reset[1] < 10 and \
                     140 < pos_after_reset[2] < 160):
                print("Physical reset to home position confirmed.")
                # self.is_connected remains True
                return True  # Connection and initialization successful
            else:
                print("Warning: Physical reset not confirmed at home position or get_position failed. Position:",
                      pos_after_reset)
                print("Connection considered unstable. Disconnecting.")
                self.is_connected = False
                if self.swift: self.swift.disconnect()
                self.swift = None
                return False

        except Exception as e:
            print(f"An unexpected error occurred during uArm connection/initialization: {e}")
            import traceback
            traceback.print_exc()
            if self.swift:
                try:
                    self.swift.disconnect()
                except:
                    pass
            self.swift = None
            self.is_connected = False
            return False

    # ... (load_human_data, map_and_execute_frame, run_mimicry_loop, cleanup methods remain the same) ...
    # Make sure to copy the rest of the UArmMimicController class and the main block
    # from the previous "Corrected: human_to_uarm_mapper.py (Type Hint Fix)" version.
    # The following are just placeholders to indicate they should be there.

    def load_human_data(self, npz_file_path: str, skeleton: str, arm: str) -> bool:
        # (Keep your existing implementation)
        self.human_pose_sequence = load_pose_sequence_from_npz(npz_file_path)
        if self.human_pose_sequence is None: return False
        self.key_joint_indices = get_arm_joint_indices(skeleton_type=skeleton, arm_to_track=arm)
        if not self.key_joint_indices: return False
        if self.human_pose_sequence.shape[0] > 0:
            self.initial_human_shoulder_pos = self.human_pose_sequence[0, self.key_joint_indices['shoulder']].copy()
        else:
            return False
        print(f"Human pose data loaded. {self.human_pose_sequence.shape[0]} frames.")
        return True

    def map_and_execute_frame(self, frame_idx: int):
        # (Keep your existing implementation, including safety clamp)
        if not self.is_connected or self.human_pose_sequence is None or \
                self.key_joint_indices is None or self.initial_human_shoulder_pos is None:
            return

        if frame_idx >= self.human_pose_sequence.shape[0]: return

        current_human_shoulder_pos = self.human_pose_sequence[frame_idx, self.key_joint_indices['shoulder']]
        current_human_wrist_pos = self.human_pose_sequence[frame_idx, self.key_joint_indices['wrist']]
        wrist_vec_in_human_shoulder_frame = current_human_wrist_pos - current_human_shoulder_pos
        scaled_wrist_vec = wrist_vec_in_human_shoulder_frame * HUMAN_TO_UARM_SCALE_FACTOR

        if TRACKED_ARM == 'right':  # TRACKED_ARM should be a global constant
            uarm_target_x_relative = scaled_wrist_vec[1]
            uarm_target_y_relative = -scaled_wrist_vec[0]
            uarm_target_z_relative = scaled_wrist_vec[2]
        elif TRACKED_ARM == 'left':
            uarm_target_x_relative = scaled_wrist_vec[1]
            uarm_target_y_relative = scaled_wrist_vec[0]
            uarm_target_z_relative = scaled_wrist_vec[2]
        else:
            return

        uarm_target_abs_x = uarm_target_x_relative + UARM_SHOULDER_ORIGIN_OFFSET[
            0]  # UARM_SHOULDER_ORIGIN_OFFSET global
        uarm_target_abs_y = uarm_target_y_relative + UARM_SHOULDER_ORIGIN_OFFSET[1]
        uarm_target_abs_z = uarm_target_z_relative + UARM_SHOULDER_ORIGIN_OFFSET[2]

        wrist_angle_uarm = 90.0

        if self.swift:
            uarm_target_abs_x = np.clip(uarm_target_abs_x, 50, 300)
            uarm_target_abs_y = np.clip(uarm_target_abs_y, -150, 150)
            uarm_target_abs_z = np.clip(uarm_target_abs_z, 0, 250)

            self.swift.set_position(x=uarm_target_abs_x, y=uarm_target_abs_y, z=uarm_target_abs_z,
                                    speed=MOVEMENT_SPEED_MMPM, wait=False)  # MOVEMENT_SPEED_MMPM global
            self.swift.set_wrist(angle=wrist_angle_uarm, speed=WRIST_SPEED_DEGPM,
                                 wait=False)  # WRIST_SPEED_DEGPM global

    def run_mimicry_loop(self):
        # (Keep your existing implementation)
        if not self.is_connected or self.human_pose_sequence is None: return
        num_frames = self.human_pose_sequence.shape[0]
        print(
            f"\nStarting uArm mimicry for {num_frames} frames at ~{TARGET_COMMAND_FPS} FPS...")  # TARGET_COMMAND_FPS global
        for frame_idx in range(num_frames):
            if not self.is_connected or not self.swift: break
            start_time = time.time()
            if frame_idx % (TARGET_COMMAND_FPS * 2) == 0:
                current_human_shoulder_pos = self.human_pose_sequence[frame_idx, self.key_joint_indices['shoulder']]
                current_human_wrist_pos = self.human_pose_sequence[frame_idx, self.key_joint_indices['wrist']]
                wrist_vec_in_human_shoulder_frame = current_human_wrist_pos - current_human_shoulder_pos
                print(
                    f"Processing Frame {frame_idx}/{num_frames}... Human wrist rel: ({wrist_vec_in_human_shoulder_frame[0]:.2f}, {wrist_vec_in_human_shoulder_frame[1]:.2f}, {wrist_vec_in_human_shoulder_frame[2]:.2f})")
            self.map_and_execute_frame(frame_idx)
            time_to_wait = (1.0 / TARGET_COMMAND_FPS) - (time.time() - start_time)
            if time_to_wait > 0: time.sleep(time_to_wait)
        print("Mimicry loop finished.")
        if self.swift:
            print("Waiting for final commands to process...")
            self.swift.flush_cmd(timeout=5, wait_stop=True)

    def cleanup(self):
        # (Keep your existing implementation)
        if self.swift and self.is_connected:
            print("\nResetting arm before disconnecting...")
            self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=15)  # INITIAL_UARM_RESET_SPEED global
            print(f"Final reset. Position: {self.swift.get_position(wait=True, timeout=5)}")
            print("Disconnecting from uArm...")
            self.swift.disconnect()
        elif self.swift:
            print("\nSwift object exists but not marked connected, attempting disconnect anyway...")
            try:
                self.swift.disconnect()
            except:
                pass
        self.is_connected = False
        self.swift = None
        print("Cleanup complete.")

def main():
    """Main execution function."""
    print("--- Human Arm to uArm Mimicry Script ---")

    try:
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root_dir = ".."
        print("Warning: Could not automatically determine project root. Assuming current directory.")

    npz_full_path = os.path.join(project_root_dir, NPZ_FILE_RELATIVE_PATH)

    mimic_controller = UArmMimicController(port=UARM_SERIAL_PORT)

    try:
        if not mimic_controller.load_human_data(npz_full_path, SKELETON_TYPE, TRACKED_ARM):
            print("Failed to load human data. Exiting.")
            return

        if not mimic_controller.connect_uarm():
            print("Failed to connect to uArm. Exiting.")
            return

        print("\nArm and data ready. Starting mimicry in 3 seconds...")
        time.sleep(3)

        mimic_controller.run_mimicry_loop()

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mimic_controller.cleanup()
        print("\n--- Program Finished ---")


if __name__ == '__main__':
    main()
