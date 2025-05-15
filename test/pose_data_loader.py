"""
pose_data_loader.py

This script is responsible for loading 3D human pose sequences from .npz files,
identifying key anatomical joints (shoulder, elbow, wrist) based on a specified
skeleton definition, and providing utility functions to access this data.

It expects .npz files to contain a 'poses_r3j' key, which maps to a NumPy array
of shape (num_frames, num_joints, 3).
"""
import os
import sys
import numpy as np
from typing import Optional, Dict, Tuple, List

# --- Constants for Skeleton Definitions ---
# These could be moved to a dedicated configuration file or a more comprehensive skeleton utility module.

# SMPL 24-joint definition (common in AMASS)
# Indices are based on a typical SMPL joint ordering.
# Users should verify these against their specific dataset's joint definition.
SMPL24_JOINT_MAPPING = {
    'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3, 'L_Knee': 4, 'R_Knee': 5,
    'Spine2': 6, 'L_Ankle': 7, 'R_Ankle': 8, 'Spine3': 9, 'L_Foot': 10, 'R_Foot': 11,
    'Neck': 12, 'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
    'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19,
    'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23
}
# Note: For some tasks, L_Hand might be considered the same as L_Wrist, and R_Hand as R_Wrist.
# The original HPSTM code used a specific mapping for its 24 joints from SMPL+H:
# smplh_to_smpl24_body_indices = list(range(22)) + [20, 21]
# This implies that for the final 24 joints:
#   - Joints 0-21 are taken directly from a 22-joint (or more) definition.
#   - Joint 22 (L_Hand in the 24-joint set) is mapped from joint 20 (L_Wrist in the source).
#   - Joint 23 (R_Hand in the 24-joint set) is mapped from joint 21 (R_Wrist in the source).
# For simplicity here, we'll use the direct R_Shoulder, R_Elbow, R_Wrist indices (17, 19, 21).
# If your 'poses_r3j' already reflects the final 24-joint selection where R_Hand is the primary hand marker,
# you might use SMPL24_JOINT_MAPPING['R_Hand'] (index 23).

SMPL24_ARM_KEY_JOINTS = {
    'right': {
        'shoulder': SMPL24_JOINT_MAPPING['R_Shoulder'],  # 17
        'elbow': SMPL24_JOINT_MAPPING['R_Elbow'],  # 19
        'wrist': SMPL24_JOINT_MAPPING['R_Wrist']  # 21 (or R_Hand: 23, depending on convention)
    },
    'left': {
        'shoulder': SMPL24_JOINT_MAPPING['L_Shoulder'],  # 16
        'elbow': SMPL24_JOINT_MAPPING['L_Elbow'],  # 18
        'wrist': SMPL24_JOINT_MAPPING['L_Wrist']  # 20 (or L_Hand: 22)
    }
}

# --- Path Setup ---
# This section helps Python find modules in the 'src' directory.
# It assumes this script is located in the project root, and 'src' is a subdirectory.
# Adjust if your project structure is different.
try:
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assume the project root is one level up if this script is in a 'scripts' subdir,
    # or the same directory if it's in the root.
    # For this example, let's assume the script is in the project root.
    project_root = current_script_dir
    src_path_attempt = os.path.join(project_root, "src")

    if not os.path.isdir(src_path_attempt):
        # If 'src' is not in project_root, maybe project_root is 'src's parent
        project_root_parent = os.path.dirname(project_root)
        src_path_attempt = os.path.join(project_root_parent, "src")

    if os.path.isdir(src_path_attempt) and src_path_attempt not in sys.path:
        sys.path.insert(0, src_path_attempt)
        print(f"Info: Added '{src_path_attempt}' to sys.path for module imports.")

    from kinematics.skeleton_utils import get_num_joints as get_total_joints_for_skeleton

    print("Successfully imported 'get_total_joints_for_skeleton' from 'src.kinematics.skeleton_utils'.")

except ImportError as e:
    print(f"Error: Failed to import modules from 'src.kinematics.skeleton_utils': {e}")
    print("Ensure that the 'src' directory is in your Python path or structured correctly relative to this script.")
    print(f"Attempted src path: {src_path_attempt if 'src_path_attempt' in locals() else 'Not determined'}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)  # Exit if essential modules cannot be imported
except Exception as e:
    print(f"An unexpected error occurred during path setup or initial imports: {e}")
    sys.exit(1)


def load_pose_sequence_from_npz(npz_file_path: str, expected_key: str = 'poses_r3j') -> Optional[np.ndarray]:
    """
    Loads a 3D pose sequence from a specified .npz file.

    Args:
        npz_file_path (str): The absolute or relative path to the .npz file.
        expected_key (str): The key in the .npz file that contains the pose data.
                            Defaults to 'poses_r3j'.

    Returns:
        Optional[np.ndarray]: A NumPy array of shape (num_frames, num_joints, 3)
                              containing the 3D joint positions for each frame,
                              or None if loading fails or data is invalid.
    """
    if not os.path.exists(npz_file_path):
        print(f"Error: File not found at '{npz_file_path}'.")
        return None
    if not os.path.isfile(npz_file_path):
        print(f"Error: Path '{npz_file_path}' is not a file.")
        return None

    try:
        data = np.load(npz_file_path)
        if expected_key not in data:
            print(f"Error: Expected key '{expected_key}' not found in '{npz_file_path}'.")
            print(f"Available keys: {list(data.keys())}")
            return None

        pose_sequence = data[expected_key]

        # Validate shape: (num_frames, num_joints, 3)
        if pose_sequence.ndim != 3 or pose_sequence.shape[2] != 3:
            print(
                f"Error: Data for key '{expected_key}' in '{npz_file_path}' has an unexpected shape: {pose_sequence.shape}. "
                "Expected (num_frames, num_joints, 3).")
            return None

        if pose_sequence.shape[0] == 0:  # No frames
            print(f"Warning: Data for key '{expected_key}' in '{npz_file_path}' contains zero frames.")
            # Return empty array with correct dimensions for consistency, or None
            return np.empty((0, pose_sequence.shape[1], 3), dtype=pose_sequence.dtype)

        print(f"Successfully loaded '{expected_key}' from '{npz_file_path}'. Shape: {pose_sequence.shape}")
        return pose_sequence.astype(np.float32)  # Ensure float32 for consistency

    except Exception as e:
        print(f"Error loading or processing .npz file '{npz_file_path}': {e}")
        return None


def get_arm_joint_indices(skeleton_type: str = 'smpl_24', arm_to_track: str = 'right') -> Optional[Dict[str, int]]:
    """
    Retrieves the indices for shoulder, elbow, and wrist joints for a specified arm
    based on the skeleton type.

    Args:
        skeleton_type (str): The type of skeleton definition (e.g., 'smpl_24').
        arm_to_track (str): Specifies which arm to track ('right' or 'left').

    Returns:
        Optional[Dict[str, int]]: A dictionary mapping 'shoulder', 'elbow', 'wrist'
                                  to their respective joint indices. Returns None if
                                  the skeleton type or arm is unsupported.
    """
    if skeleton_type.lower() == 'smpl_24':
        if arm_to_track.lower() in SMPL24_ARM_KEY_JOINTS:
            return SMPL24_ARM_KEY_JOINTS[arm_to_track.lower()]
        else:
            print(f"Error: Invalid 'arm_to_track' value: '{arm_to_track}'. Must be 'right' or 'left'.")
            return None
    else:
        print(f"Error: Key joint indices for skeleton type '{skeleton_type}' are not defined in this utility.")
        print("Please add or verify the definition in SMPL24_ARM_KEY_JOINTS or a similar structure.")
        return None


def main():
    """
    Main function to demonstrate loading pose data and identifying key joints.
    """
    print("--- Pose Data Loader Demonstration ---")

    # --- Configuration ---
    # Determine the project root directory. This assumes 'data' and 'src' are subdirectories
    # of the project root, and this script might be in the project root or a 'scripts' subdirectory.
    try:
        # Assumes this script (pose_data_loader.py) is in the project root.
        # If it's in a 'scripts' subdirectory, then project_root should be os.path.dirname(current_script_dir)
        _current_script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = _current_script_dir  # Modify if script is in a subfolder like 'scripts'

        # Construct the path to the example .npz file
        # Replace with the actual path to your .npz file relative to the project root.
        relative_npz_path = os.path.join("../data", "00", "00_01_poses.npz")  # Example path
        npz_file_to_load = os.path.join(_project_root, relative_npz_path)
    except NameError:  # __file__ is not defined (e.g. in an interactive session not run as script)
        print("Warning: Could not automatically determine project root. Please set 'npz_file_to_load' manually.")
        npz_file_to_load = "path/to/your/data/your_sequence.npz"  # Fallback: User must edit this

    target_skeleton_type = 'smpl_24'
    arm_choice = 'right'  # Options: 'right' or 'left'
    frames_to_display_info_for = 5
    # --- End Configuration ---

    print(f"\nAttempting to load data for skeleton '{target_skeleton_type}', tracking '{arm_choice}' arm.")
    print(f"Target .npz file: '{npz_file_to_load}'")

    pose_data = load_pose_sequence_from_npz(npz_file_to_load)

    if pose_data is None:
        print("\nExiting due to data loading failure.")
        return

    num_frames, num_joints_data, num_coords = pose_data.shape
    print(
        f"\nLoaded sequence details: {num_frames} frames, {num_joints_data} joints, {num_coords} coordinates per joint.")

    # Validate against expected number of joints for the skeleton type
    expected_joints = get_total_joints_for_skeleton(target_skeleton_type)
    if num_joints_data != expected_joints:
        print(f"Warning: Loaded data has {num_joints_data} joints, but skeleton type '{target_skeleton_type}' "
              f"is defined with {expected_joints} joints in 'skeleton_utils.py'. "
              "Ensure consistency for correct joint indexing.")
        # Decide if to proceed or exit. For now, we'll proceed but key_joint_indices might be incorrect.

    key_joint_indices = get_arm_joint_indices(skeleton_type=target_skeleton_type, arm_to_track=arm_choice)

    if not key_joint_indices:
        print("\nCould not retrieve key joint indices. Exiting.")
        return

    print(f"\nKey joint indices for '{arm_choice}' arm (skeleton: '{target_skeleton_type}'):")
    for joint_name, joint_idx in key_joint_indices.items():
        print(f"  {joint_name.capitalize()}: Index {joint_idx}")

    # Display coordinates for the first few frames
    if num_frames > 0:
        print(
            f"\nDisplaying coordinates for key joints in the first {min(frames_to_display_info_for, num_frames)} frames:")
        for i in range(min(frames_to_display_info_for, num_frames)):
            print(f"  --- Frame {i} ---")
            for joint_name, joint_idx in key_joint_indices.items():
                if joint_idx < num_joints_data:  # Check if index is valid for the loaded data
                    joint_coords = pose_data[i, joint_idx, :]
                    print(
                        f"    {joint_name.capitalize()} (Idx {joint_idx}): X={joint_coords[0]:.3f}, Y={joint_coords[1]:.3f}, Z={joint_coords[2]:.3f}")
                else:
                    print(
                        f"    {joint_name.capitalize()} (Idx {joint_idx}): Index out of bounds for loaded data ({num_joints_data} joints).")

        # Calculate and print the vector from shoulder to wrist for the first frame
        shoulder_idx = key_joint_indices['shoulder']
        wrist_idx = key_joint_indices['wrist']
        if shoulder_idx < num_joints_data and wrist_idx < num_joints_data:
            shoulder_pos_f0 = pose_data[0, shoulder_idx, :]
            wrist_pos_f0 = pose_data[0, wrist_idx, :]
            vector_shoulder_to_wrist_f0 = wrist_pos_f0 - shoulder_pos_f0
            print(f"\nVector from Shoulder to Wrist (Frame 0): {vector_shoulder_to_wrist_f0}")
            print(f"  Magnitude (approx. arm length segment): {np.linalg.norm(vector_shoulder_to_wrist_f0):.3f}")
    else:
        print("\nNo frames in the loaded sequence to display.")

    print("\n--- End of Pose Data Loader Demonstration ---")


if __name__ == '__main__':
    main()
