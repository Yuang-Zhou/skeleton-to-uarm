# arm_mapping_utils.py

import os
import sys
import numpy as np
from typing import Optional, Dict # Python 3.9+ can use dict

# Import constants (specifically for skeleton definitions if needed here, or pass them)
from config_arm_mapper import SMPL24_ARM_KEY_JOINTS, SKELETON_TYPE, UARM_PLACEMENT_MODE

# --- Path Setup Logic ---
# This setup assumes this utils file might be imported by a script in the project root,
# or that the project root is discoverable. Adjust if paths become problematic.
try:
    _current_script_dir_util = os.path.dirname(os.path.abspath(__file__))
    # Assuming this util file is in the same directory as main_mapper.py,
    # or main_mapper.py is in the project root.
    _project_root_util = _current_script_dir_util

    # Add local 'src' directory to sys.path
    _src_path_hpstm_util = os.path.join(_project_root_util, "src")
    if os.path.isdir(_src_path_hpstm_util) and _src_path_hpstm_util not in sys.path:
        sys.path.insert(0, _src_path_hpstm_util)
        # print(f"Info (from arm_mapping_utils.py): Added HPSTM src path '{_src_path_hpstm_util}' to sys.path.")

    # Add uArm SDK path to sys.path
    _sdk_base_dir_util = os.path.join(_project_root_util, "uarm-python-sdk")
    _sdk_actual_path_util = os.path.join(_sdk_base_dir_util, "uArm-Python-SDK-2.0")
    if os.path.isdir(os.path.join(_sdk_actual_path_util, "uarm")) and _sdk_actual_path_util not in sys.path:
        sys.path.insert(0, _sdk_actual_path_util)
        # print(f"Info (from arm_mapping_utils.py): Added uArm SDK path '{_sdk_actual_path_util}' to sys.path.")

    # These imports are here to confirm path setup works.
    # They might be used by functions within this file or be imported directly by other modules.
    # from kinematics.skeleton_utils import get_skeleton_parents # Example, if needed
    # from uarm.wrapper import SwiftAPI # Example, if needed

except ImportError as e:
    print(f"Critical Import Error in arm_mapping_utils.py during path setup: {e}")
    print("Ensure 'src' and 'uarm-python-sdk' are correctly placed relative to your project structure.")
    sys.exit(1)
except Exception as e_path:
    print(f"An error occurred during path setup in arm_mapping_utils.py: {e_path}")
    # Decide if this is critical enough to exit
    # sys.exit(1)


def load_pose_sequence_from_npz(npz_file_path: str, expected_key: str = 'poses_r3j') -> Optional[np.ndarray]:
    """
    Loads a pose sequence from an NPZ file.
    npz_file_path (str): Absolute or relative path to the .npz file.
    expected_key (str): The key in the .npz file containing the pose data.
    Returns:
        Optional[np.ndarray]: A NumPy array of shape (num_frames, num_joints, 3)
                              or None if loading fails.
    """
    if not os.path.exists(npz_file_path):
        print(f"Error (utils): File not found: {npz_file_path}")
        return None
    try:
        data = np.load(npz_file_path)
        if expected_key not in data:
            print(f"Error (utils): Expected key '{expected_key}' not found in NPZ file {npz_file_path}.")
            available_keys = list(data.keys())
            print(f"Available keys: {available_keys}")
            return None
        pose_sequence = data[expected_key]
        if pose_sequence.ndim != 3 or pose_sequence.shape[2] != 3:
            print(f"Error (utils): Pose sequence has incorrect shape {pose_sequence.shape}. Expected (frames, joints, 3).")
            return None
        # print(f"Successfully loaded '{expected_key}' from '{npz_file_path}'. Shape: {pose_sequence.shape}")
        return pose_sequence.astype(np.float32)
    except Exception as e:
        print(f"Error (utils): Loading pose sequence from {npz_file_path}: {e}")
        return None


def get_arm_joint_indices(skeleton_type: str = SKELETON_TYPE, arm_to_track: str = 'right') -> Optional[Dict[str, int]]:
    """
    Gets key joint indices (shoulder, elbow, wrist) for the specified arm and skeleton type.
    skeleton_type (str): The type of skeleton (e.g., 'smpl_24').
    arm_to_track (str): Specifies which arm ('right' or 'left').
    Returns:
        Optional[Dict[str, int]]: A dictionary mapping 'shoulder', 'elbow', 'wrist'
                                  to their indices, or None if not found.
    """
    # Uses SMPL24_ARM_KEY_JOINTS imported from config_arm_mapper.py
    if skeleton_type.lower() == 'smpl_24':
        selected_arm_joints = SMPL24_ARM_KEY_JOINTS.get(arm_to_track.lower())
        if selected_arm_joints:
            return selected_arm_joints
        else:
            print(f"Error (utils): Invalid 'arm_to_track' value: '{arm_to_track}' for skeleton '{skeleton_type}'. Must be 'right' or 'left'.")
            return None
    else:
        print(f"Error (utils): Key joint indices for skeleton type '{skeleton_type}' are not defined in SMPL24_ARM_KEY_JOINTS.")
        return None

def get_rotation_matrix_native_to_viz() -> np.ndarray:
    """
    Calculates the rotation matrix to transform coordinates from uArm NATIVE SDK space
    to the common VISUALIZATION space (X-right, Y-forward, Z-up, typically aligned with pinned human model).
    This depends on UARM_PLACEMENT_MODE.
    Returns:
        np.ndarray: A 3x3 rotation matrix. P_viz = R @ P_native_sdk
    """
    # UARM_PLACEMENT_MODE is imported from config_arm_mapper
    if UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
        # uArm Native SDK CS for this mount (as interpreted):
        #   SDK_X: Points UPWARDS along the robot's physical 'height' when mounted. (World Z if robot stands on its mounting plate)
        #   SDK_Y: Points to the ROBOT'S LEFT along its physical 'width'. (World -X if robot faces world Y)
        #   SDK_Z: Points FORWARD along the robot's physical 'reach'. (World Y if robot faces world Y)
        # Visualization CS (Human-centric):
        #   Viz_X: To the Right
        #   Viz_Y: Forward
        #   Viz_Z: Upwards
        # Mapping from Native SDK to Visualization:
        #   SDK_X (Robot Up)    -> Viz_Z (Up)
        #   SDK_Y (Robot Left)  -> Viz_-X (Left -> -Right)  => Viz_X = -SDK_Y
        #   SDK_Z (Robot Reach) -> Viz_Y (Forward)
        # So, P_viz = R @ P_sdk_native
        # Viz_X_col = [ 0, -1,  0]
        # Viz_Y_col = [ 0,  0,  1]
        # Viz_Z_col = [ 1,  0,  0]
        R_native_to_viz = np.array([
            [0, -1, 0],  # Viz_X = -SDK_Y
            [0,  0, 1],  # Viz_Y =  SDK_Z
            [1,  0, 0]   # Viz_Z =  SDK_X
        ])
    elif UARM_PLACEMENT_MODE == 'upright':
        # uArm Native SDK CS (standard upright):
        #   SDK_X: Robot Front (Reach)
        #   SDK_Y: Robot Up
        #   SDK_Z: Robot Right
        # Visualization CS (Human-centric):
        #   Viz_X: To the Right
        #   Viz_Y: Forward
        #   Viz_Z: Upwards
        # Mapping:
        #   SDK_X (Robot Front)  -> Viz_Y (Forward)
        #   SDK_Y (Robot Up)     -> Viz_Z (Up)
        #   SDK_Z (Robot Right)  -> Viz_X (Right)
        # Viz_X_col = [0, 0, 1]
        # Viz_Y_col = [1, 0, 0]
        # Viz_Z_col = [0, 1, 0]
        R_native_to_viz = np.array([
            [0, 0, 1],  # Viz_X = SDK_Z
            [1, 0, 0],  # Viz_Y = SDK_X
            [0, 1, 0]   # Viz_Z = SDK_Y
        ])
    else:
        print(f"Warning (utils): Unknown UARM_PLACEMENT_MODE '{UARM_PLACEMENT_MODE}'. Defaulting to identity matrix for viz transform.")
        R_native_to_viz = np.eye(3)
    return R_native_to_viz