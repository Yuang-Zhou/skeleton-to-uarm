import os
import sys
import time
import numpy as np
import math  # math is imported but not explicitly used, np functions are generally preferred for arrays.
import threading
import collections
from typing import Optional, Tuple, List

import matplotlib

matplotlib.use('TkAgg')  # Using TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Constants and Configuration ---
NPZ_FILE_RELATIVE_PATH = os.path.join("../data", "00", "joints_drc_smooth.npz")
SKELETON_TYPE = 'smpl_24'
TRACKED_ARM = 'right'  # Options: 'right' or 'left'

# !!! CRITICAL: Replace with your actual uArm serial port or set to None for auto-detect !!!
UARM_SERIAL_PORT = '/dev/cu.usbmodem144201'  # Example for macOS, use 'COMx' for Windows or None
INITIAL_UARM_RESET_SPEED = 3000  # mm/min
MOVEMENT_SPEED_MMPM = 7000  # Speed for set_position in mm/min
WRIST_SPEED_DEGPM = 1200  # Speed for set_wrist in deg/min

# Parameters for mapping human motion to uArm space
UARM_TARGET_MAPPED_ARM_LENGTH_MM = 400.0  # Target uArm reach when human arm is conceptually extended (mm)
MIN_HUMAN_ARM_LENGTH_FOR_SCALING_M = 0.1  # Minimum human arm length (meters) for stable dynamic scaling
FALLBACK_SCALE_FACTOR_M_TO_MM = 180.0  # Fallback scale if human arm length is too small or zero

# UARM_SHOULDER_ORIGIN_OFFSET defines the position of the human's shoulder joint
# mapped into the uArm's NATIVE coordinate system (in mm).
# This point acts as the conceptual "shoulder" or "base of operation" for the uArm.
# For side_mounted_native_x_cw90: Native X (robot up), Y (robot left/right), Z (robot reach)
UARM_SHOULDER_ORIGIN_OFFSET = np.array([100.0, 0.0, 150.0])

TARGET_FPS = 25  # Target FPS for uArm command sending and animation updates

# Trail lengths for visualization
UARM_TARGET_TRAIL_LENGTH = 60
HUMAN_WRIST_TRAIL_LENGTH = 60

HUMAN_ROOT_JOINT_IDX = 0  # Typically Pelvis (index 0) for SMPL-like skeletons
UARM_PLACEMENT_MODE = 'side_mounted_native_x_cw90'  # Options: 'upright', 'side_mounted_native_x_cw90'

# --- Helper: Path Setup for custom modules and SDK ---
try:
    _current_script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = _current_script_dir  # Assuming script is at project root

    # Add local 'src' directory to sys.path if it exists (for kinematics.skeleton_utils)
    _src_path_hpstm = os.path.join(_project_root, "src")
    if os.path.isdir(_src_path_hpstm) and _src_path_hpstm not in sys.path:
        sys.path.insert(0, _src_path_hpstm)
        print(f"Info: Added HPSTM src path '{_src_path_hpstm}' to sys.path.")

    # Add uArm SDK path to sys.path if it exists
    _sdk_base_dir = os.path.join(_project_root, "uarm-python-sdk")  # Expected SDK location
    _sdk_actual_path = os.path.join(_sdk_base_dir, "uArm-Python-SDK-2.0")  # Specific SDK version folder
    if os.path.isdir(os.path.join(_sdk_actual_path, "uarm")) and _sdk_actual_path not in sys.path:
        sys.path.insert(0, _sdk_actual_path)
        print(f"Info: Added uArm SDK path '{_sdk_actual_path}' to sys.path.")

    # Import necessary modules after path setup
    from kinematics.skeleton_utils import get_skeleton_parents
    from uarm.wrapper import SwiftAPI

    # SMPL24 joint mapping and key arm joints
    SMPL24_JOINT_MAPPING = {
        'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3, 'L_Knee': 4, 'R_Knee': 5,
        'Spine2': 6, 'L_Ankle': 7, 'R_Ankle': 8, 'Spine3': 9, 'L_Foot': 10, 'R_Foot': 11,
        'Neck': 12, 'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
        'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19,
        'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23
    }
    SMPL24_ARM_KEY_JOINTS = {
        'right': {'shoulder': SMPL24_JOINT_MAPPING['R_Shoulder'],
                  'elbow': SMPL24_JOINT_MAPPING['R_Elbow'],
                  'wrist': SMPL24_JOINT_MAPPING['R_Wrist']},
        'left': {'shoulder': SMPL24_JOINT_MAPPING['L_Shoulder'],
                 'elbow': SMPL24_JOINT_MAPPING['L_Elbow'],
                 'wrist': SMPL24_JOINT_MAPPING['L_Wrist']}
    }
    print("Successfully imported required modules and SDK.")
except ImportError as e:
    print(
        f"Critical Import Error: {e}. Please ensure 'kinematics' module and uArm SDK are correctly placed and accessible.")
    sys.exit(1)


# --- End Path Setup ---

def load_pose_sequence_from_npz(npz_file_path: str, expected_key: str = 'poses_r3j') -> Optional[np.ndarray]:
    """Loads a pose sequence from an NPZ file."""
    if not os.path.exists(npz_file_path):
        print(f"Error: File not found: {npz_file_path}")
        return None
    try:
        data = np.load(npz_file_path)
        if expected_key not in data:
            print(f"Error: Expected key '{expected_key}' not found in NPZ file {npz_file_path}.")
            return None
        pose_sequence = data[expected_key]
        if pose_sequence.ndim != 3 or pose_sequence.shape[2] != 3:
            print(f"Error: Pose sequence has incorrect shape {pose_sequence.shape}. Expected (frames, joints, 3).")
            return None
        return pose_sequence.astype(np.float32)
    except Exception as e:
        print(f"Error loading pose sequence from {npz_file_path}: {e}")
        return None


def get_arm_joint_indices(skeleton_type: str = 'smpl_24', arm_to_track: str = 'right') -> Optional[dict[str, int]]:
    """Gets key joint indices for the specified arm and skeleton type."""
    if skeleton_type.lower() == 'smpl_24':
        return SMPL24_ARM_KEY_JOINTS.get(arm_to_track.lower())
    print(f"Error: Skeleton type '{skeleton_type}' not supported or arm '{arm_to_track}' is invalid.")
    return None


class UArmMimicControllerWithViz:
    """
    Controls the uArm to mimic human arm movements loaded from pose data,
    and provides data for 3D visualization.
    """

    def __init__(self, port: Optional[str] = UARM_SERIAL_PORT):
        self.swift: Optional[SwiftAPI] = None
        self.port: Optional[str] = port
        self.is_connected_and_ready: bool = False
        self.key_joint_indices: Optional[dict[str, int]] = None

        self.raw_human_pose_sequence: Optional[np.ndarray] = None
        self.human_pose_sequence_for_viz: Optional[np.ndarray] = None  # Root-pinned version for visualization
        self.initial_human_root_pos_raw: Optional[np.ndarray] = None
        self.initial_human_shoulder_pos_raw: Optional[np.ndarray] = None  # Shoulder of the tracked arm

        self.uarm_control_thread: Optional[threading.Thread] = None
        self.stop_thread_flag = threading.Event()
        self.current_human_frame_idx: int = 0
        self.latest_uarm_target_abs_mm = collections.deque(maxlen=1)  # Stores the latest uArm target in native CS
        self.uarm_target_trail_mm = collections.deque(maxlen=UARM_TARGET_TRAIL_LENGTH)  # For viz
        self.human_wrist_trail_for_viz = collections.deque(maxlen=HUMAN_WRIST_TRAIL_LENGTH)  # For viz
        self.human_skeleton_parents: Optional[np.ndarray] = None  # Will be loaded based on SKELETON_TYPE
        self.last_calculated_dynamic_scale_factor: float = FALLBACK_SCALE_FACTOR_M_TO_MM

    def connect_uarm(self) -> bool:
        """Attempts to connect to the uArm and initialize it."""
        print(f"Attempting to connect to uArm on port: {self.port if self.port else 'Auto-detect'}...")
        try:
            self.swift = SwiftAPI(port=self.port,
                                  臂吸펌웨어_업그레이드=False)  # Disable firmware upgrade prompt for faster connection
            time.sleep(2.0)  # Allow time for connection to establish

            if not self.swift.connected:
                print("Error: SwiftAPI serial connection failed.")
                self.swift = None
                return False

            _ = self.swift.waiting_ready(timeout=20)
            power_status = self.swift.get_power_status(wait=True, timeout=5)  # Updated call
            if power_status:  # True if powered, False if not, None on error
                print("uArm power status is ON.")
            else:
                print("Warning: uArm power status is OFF or could not be determined. Please ensure it's powered on.")

            device_info = self.swift.get_device_info(timeout=10)
            if device_info:
                print(f"Device Info: {device_info}")
            else:
                print("Warning: Failed to get device info.")

            current_mode = self.swift.get_mode(wait=True, timeout=10)
            if current_mode != 0 and current_mode is not None:  # Mode 0 is Normal Mode
                print(f"uArm is in mode {current_mode}, setting to mode 0 (Normal).")
                self.swift.set_mode(0, wait=True, timeout=10)

            print("Resetting uArm on connect (standard SDK reset)...")
            self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=25)
            pos_after_reset = self.swift.get_position(wait=True, timeout=10)
            print(f"uArm position after SDK reset: {pos_after_reset}")

            if isinstance(pos_after_reset, list) and \
                    (190 < pos_after_reset[0] < 210 and \
                     -10 < pos_after_reset[1] < 10 and \
                     140 < pos_after_reset[2] < 160):  # Check if near typical home position
                print("Physical reset to home position confirmed by SDK reset.")
                self.is_connected_and_ready = True
                return True
            else:
                print("Warning: Physical reset not confirmed by SDK reset. Position is unexpected.")
                if self.swift: self.swift.disconnect()
                self.swift = None
                return False
        except Exception as e:
            print(f"Error connecting to uArm: {e}")
            if self.swift: self.swift.disconnect()  # Ensure disconnection on error
            self.swift = None
            return False

    def load_human_data(self, npz_file_path: str, skeleton_def: str, arm_choice: str) -> bool:
        """Loads human pose data and prepares it for mimicry and visualization."""
        self.raw_human_pose_sequence = load_pose_sequence_from_npz(npz_file_path)
        if self.raw_human_pose_sequence is None or self.raw_human_pose_sequence.shape[0] == 0:
            print(f"Error: Failed to load human pose data from '{npz_file_path}' or sequence is empty.")
            return False

        self.key_joint_indices = get_arm_joint_indices(skeleton_type=skeleton_def, arm_to_track=arm_choice)
        if not self.key_joint_indices:
            print(f"Error: Could not get key joint indices for skeleton '{skeleton_def}' and arm '{arm_choice}'.")
            return False

        self.human_skeleton_parents = get_skeleton_parents(skeleton_def)
        if self.human_skeleton_parents is None:
            print(f"Error: Could not get skeleton parents for '{skeleton_def}'.")
            return False

        num_joints_in_data = self.raw_human_pose_sequence.shape[1]
        max_expected_joint_idx = max(self.key_joint_indices.values())
        if not (0 <= HUMAN_ROOT_JOINT_IDX < num_joints_in_data and max_expected_joint_idx < num_joints_in_data):
            print(f"Error: Joint indices (Root: {HUMAN_ROOT_JOINT_IDX}, MaxArm: {max_expected_joint_idx}) "
                  f"are out of bounds for the loaded skeleton with {num_joints_in_data} joints.")
            return False

        self.initial_human_root_pos_raw = self.raw_human_pose_sequence[0, HUMAN_ROOT_JOINT_IDX, :].copy()
        self.initial_human_shoulder_pos_raw = self.raw_human_pose_sequence[0, self.key_joint_indices['shoulder'],
                                              :].copy()

        print(
            f"Raw initial human shoulder (frame 0, Index {self.key_joint_indices['shoulder']}): {self.initial_human_shoulder_pos_raw}")
        print(f"Raw initial human root (frame 0, Index {HUMAN_ROOT_JOINT_IDX}): {self.initial_human_root_pos_raw}")

        # Create a root-relative (pinned at origin) version for visualization
        self.human_pose_sequence_for_viz = self.raw_human_pose_sequence - self.initial_human_root_pos_raw[np.newaxis,
                                                                          np.newaxis, :]

        print(f"Human pose data loaded: {self.raw_human_pose_sequence.shape[0]} frames.")
        print(f"Visualization data (human_pose_sequence_for_viz) created and is root-relative.")
        return True

    def move_uarm_to_initial_human_pose(self, initial_frame_idx: int = 0,
                                        speed_mm_per_min: Optional[int] = None) -> bool:
        """Calculates and moves uArm to correspond with the initial human pose frame."""
        if not self.is_connected_and_ready or not self.swift:
            print("Error: uArm not connected/ready for initial pose movement.")
            return False

        print(f"\nMoving uArm to correspond with human pose at frame {initial_frame_idx}...")
        targets = self.calculate_uarm_target_for_frame(initial_frame_idx,
                                                       is_initial_move=True)  # Pass flag for initial move debug
        if targets:
            uarm_target_abs_mm, wrist_angle_uarm_deg = targets
            positioning_speed = speed_mm_per_min if speed_mm_per_min is not None else MOVEMENT_SPEED_MMPM

            print(f"  Targeting initial uArm pos (Native mm): {uarm_target_abs_mm}, Wrist: {wrist_angle_uarm_deg} deg")
            print(f"  Movement speed: {positioning_speed} mm/min")

            pos_result = self.swift.set_position(
                x=uarm_target_abs_mm[0], y=uarm_target_abs_mm[1], z=uarm_target_abs_mm[2],
                speed=positioning_speed, wait=True, timeout=20
            )
            wrist_result = self.swift.set_wrist(
                angle=wrist_angle_uarm_deg, speed=WRIST_SPEED_DEGPM, wait=True, timeout=10
            )
            final_pos = self.swift.get_position(wait=True, timeout=5)
            print(
                f"  uArm pos after initial move attempt: {final_pos}. Pos cmd success: {pos_result}, Wrist cmd success: {wrist_result}")

            if isinstance(final_pos, list) and np.allclose(final_pos, uarm_target_abs_mm, atol=5.0):
                print("  uArm successfully moved to initial human pose.")
                if len(self.latest_uarm_target_abs_mm) > 0: self.latest_uarm_target_abs_mm.popleft()
                self.latest_uarm_target_abs_mm.append(uarm_target_abs_mm.copy())
                self.uarm_target_trail_mm.append(uarm_target_abs_mm.copy())  # Start trail
                return True
            else:
                print("  Warning: uArm final position doesn't match target for initial pose.");
                return False
        else:
            print("  Error: Could not calculate uArm target for initial human pose.");
            return False

    '''
    def calculate_uarm_target_for_frame(self, frame_idx: int, is_initial_move: bool = False) -> Optional[
        Tuple[np.ndarray, float]]:
        """
        Calculates the uArm target position and wrist angle for a given human pose frame.
        Includes debug prints for vector lengths.
        """
        if self.raw_human_pose_sequence is None or self.key_joint_indices is None: return None
        if not (0 <= frame_idx < self.raw_human_pose_sequence.shape[0]): return None

        shoulder_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['shoulder']]
        elbow_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['elbow']]
        wrist_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['wrist']]

        # --- Human arm vectors calculation (in meters) ---
        upper_arm_vec_raw = elbow_pos_raw - shoulder_pos_raw
        forearm_vec_raw = wrist_pos_raw - elbow_pos_raw
        human_upper_arm_length_m = np.linalg.norm(upper_arm_vec_raw)
        human_forearm_length_m = np.linalg.norm(forearm_vec_raw)
        # total_human_arm_length_m is based on sum of upper and forearm, used for dynamic scaling factor
        total_human_arm_length_m = human_upper_arm_length_m + human_forearm_length_m

        # wrist_vec_in_raw_human_shoulder_frame is the direct vector from shoulder to wrist
        wrist_vec_in_raw_human_shoulder_frame = wrist_pos_raw - shoulder_pos_raw
        human_shoulder_to_wrist_vec_length_m = np.linalg.norm(wrist_vec_in_raw_human_shoulder_frame)

        current_dynamic_scale_factor: float
        if total_human_arm_length_m < MIN_HUMAN_ARM_LENGTH_FOR_SCALING_M or abs(total_human_arm_length_m) < 1e-6:
            current_dynamic_scale_factor = FALLBACK_SCALE_FACTOR_M_TO_MM
        else:
            current_dynamic_scale_factor = UARM_TARGET_MAPPED_ARM_LENGTH_MM / total_human_arm_length_m
        self.last_calculated_dynamic_scale_factor = current_dynamic_scale_factor

        # Scale the shoulder-to-wrist vector to millimeters for uArm mapping
        # This scaled vector represents the desired position of the uArm effector
        # relative to its conceptual shoulder, before coordinate system transformation.
        scaled_human_shoulder_to_wrist_vec_mm = wrist_vec_in_raw_human_shoulder_frame * current_dynamic_scale_factor

        # Components of the scaled human shoulder-to-wrist vector
        h_sx, h_sy, h_sz = scaled_human_shoulder_to_wrist_vec_mm[0], \
            scaled_human_shoulder_to_wrist_vec_mm[1], \
            scaled_human_shoulder_to_wrist_vec_mm[2]

        # Transform scaled human vector to uArm's native coordinate system (relative to uArm's conceptual shoulder)
        # u_x_rel_n, u_y_rel_n, u_z_rel_n is the vector from uArm's conceptual shoulder to its effector in uArm Native CS
        u_x_rel_n, u_y_rel_n, u_z_rel_n = 0.0, 0.0, 0.0
        if UARM_PLACEMENT_MODE == 'upright':
            # Human X (e.g. body right/left) -> uArm +/-Y
            # Human Y (e.g. body forward/back) -> uArm +/-X
            # Human Z (e.g. body up/down) -> uArm +/-Z
            if TRACKED_ARM == 'right':  # Assuming human X is to their right, Y is forward, Z is up
                u_x_rel_n, u_y_rel_n, u_z_rel_n = h_sy, -h_sx, h_sz  # Map Human Y->uArm X, Human X->uArm -Y
            elif TRACKED_ARM == 'left':
                u_x_rel_n, u_y_rel_n, u_z_rel_n = h_sy, h_sx, h_sz  # Untested, adjust if necessary
            else:
                return None
        elif UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
            # Human X (e.g. body right/left) -> uArm +/-Y (robot's side-to-side)
            # Human Y (e.g. body forward/back) -> uArm +/-Z (robot's reach)
            # Human Z (e.g. body up/down) -> uArm +/-X (robot's vertical)
            if TRACKED_ARM == 'right':
                u_x_rel_n, u_y_rel_n, u_z_rel_n = h_sz, -h_sx, h_sy  # Map Human Z->uArm X, Human X->uArm -Y, Human Y->uArm Z
            elif TRACKED_ARM == 'left':
                u_x_rel_n, u_y_rel_n, u_z_rel_n = h_sz, h_sx, h_sy  # Untested, adjust if necessary
            else:
                return None
        else:
            print(f"Error: Unknown UARM_PLACEMENT_MODE '{UARM_PLACEMENT_MODE}'.");
            return None

        # This is the uArm effector's desired position relative to its conceptual shoulder, in uArm Native CS.
        uarm_effector_relative_to_conceptual_shoulder_native_mm_vec = np.array([u_x_rel_n, u_y_rel_n, u_z_rel_n])
        uarm_effector_relative_to_conceptual_shoulder_native_mm_length = np.linalg.norm(
            uarm_effector_relative_to_conceptual_shoulder_native_mm_vec)

        # --- DEBUG PRINTING ---
        # Controlled printing frequency (e.g., once per second or on initial move)
        print_debug_this_frame = is_initial_move or (frame_idx % TARGET_FPS == 0)

        if print_debug_this_frame:
            print(f"\n--- Frame {frame_idx} Debug Info {'(Initial Move)' if is_initial_move else ''} ---")
            print(f"  [Human Side - Raw Coords, Meters]")
            # print(f"    Shoulder Pos: ({shoulder_pos_raw[0]:.3f}, {shoulder_pos_raw[1]:.3f}, {shoulder_pos_raw[2]:.3f})")
            # print(f"    Wrist Pos: ({wrist_pos_raw[0]:.3f}, {wrist_pos_raw[1]:.3f}, {wrist_pos_raw[2]:.3f})")
            print(
                f"    Shoulder-to-Wrist Vector (m): ({wrist_vec_in_raw_human_shoulder_frame[0]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[1]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[2]:.3f})")
            print(f"    Shoulder-to-Wrist Vector Length (m): {human_shoulder_to_wrist_vec_length_m:.4f}")
            print(
                f"    Upper Arm Length (m): {human_upper_arm_length_m:.4f}, Forearm Length (m): {human_forearm_length_m:.4f}")
            print(f"    Total Human Arm Length (sum of segments, for scaling) (m): {total_human_arm_length_m:.4f}")
            print(f"  [Scaling & Mapping]")
            print(f"    Dynamic Scale Factor (mm/m): {current_dynamic_scale_factor:.2f}")
            print(f"    Target Mapped Arm Length (mm): {UARM_TARGET_MAPPED_ARM_LENGTH_MM:.2f}")
            # This is the length we expect the uArm's shoulder-to-effector vector to be if the human arm is fully extended
            # and scale factor is based on total_human_arm_length_m.
            # If human arm is bent, human_shoulder_to_wrist_vec_length_m will be shorter.
            print(
                f"    Scaled Human Shoulder-to-Wrist Length (mm): {human_shoulder_to_wrist_vec_length_m * current_dynamic_scale_factor:.2f}")
            print(f"  [uArm Side - Native CS, Relative to Conceptual Shoulder, Before Clipping]")
            print(f"    Effector Relative Vector (mm): ({u_x_rel_n:.2f}, {u_y_rel_n:.2f}, {u_z_rel_n:.2f})")
            print(
                f"    Effector Relative Vector Length (mm): {uarm_effector_relative_to_conceptual_shoulder_native_mm_length:.2f}")

        # Final uArm target in its absolute native coordinate system (effector position)
        # This is: uArm_Conceptual_Shoulder_Position + Effector_Relative_Vector
        uarm_target_abs_mm_before_clip = UARM_SHOULDER_ORIGIN_OFFSET + uarm_effector_relative_to_conceptual_shoulder_native_mm_vec
        uarm_target_abs_mm = uarm_target_abs_mm_before_clip.copy()  # Work with a copy for clipping

        # Clipping to uArm's reachable workspace (adjust these based on actual uArm limits and placement)
        if UARM_PLACEMENT_MODE == 'upright':
            uarm_target_abs_mm[0] = np.clip(uarm_target_abs_mm[0], 50, 300)
            uarm_target_abs_mm[1] = np.clip(uarm_target_abs_mm[1], -150, 150)
            uarm_target_abs_mm[2] = np.clip(uarm_target_abs_mm[2], 0, 250)
        elif UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
            uarm_target_abs_mm[0] = np.clip(uarm_target_abs_mm[0], 50, 200)  # Native X (robot up/down)
            uarm_target_abs_mm[1] = np.clip(uarm_target_abs_mm[1], -100, 100)  # Native Y (robot left/right)
            uarm_target_abs_mm[2] = np.clip(uarm_target_abs_mm[2], 50, 280)  # Native Z (robot reach)

        if print_debug_this_frame and not np.array_equal(uarm_target_abs_mm, uarm_target_abs_mm_before_clip):
            # Recalculate relative vector length if clipping occurred
            uarm_effector_relative_after_clip = uarm_target_abs_mm - UARM_SHOULDER_ORIGIN_OFFSET
            uarm_effector_relative_after_clip_length = np.linalg.norm(uarm_effector_relative_after_clip)
            print(f"  [uArm Side - Clipping Info]")
            print(
                f"    Target before clip: ({uarm_target_abs_mm_before_clip[0]:.2f}, {uarm_target_abs_mm_before_clip[1]:.2f}, {uarm_target_abs_mm_before_clip[2]:.2f})")
            print(
                f"    Target after clip:  ({uarm_target_abs_mm[0]:.2f}, {uarm_target_abs_mm[1]:.2f}, {uarm_target_abs_mm[2]:.2f})")
            print(
                f"    Effector Relative Vector Length (mm) AFTER Clipping: {uarm_effector_relative_after_clip_length:.2f}")
            if not np.isclose(uarm_effector_relative_to_conceptual_shoulder_native_mm_length,
                              uarm_effector_relative_after_clip_length):
                print(f"    NOTE: Clipping changed the uArm's relative arm length.")

        # Update human wrist trail for visualization (using pinned/visualization coordinates)
        if self.human_pose_sequence_for_viz is not None and 0 <= frame_idx < self.human_pose_sequence_for_viz.shape[0]:
            current_human_wrist_pos_viz = self.human_pose_sequence_for_viz[frame_idx, self.key_joint_indices['wrist']]
            self.human_wrist_trail_for_viz.append(current_human_wrist_pos_viz.copy())

        wrist_angle_uarm_deg = 90.0  # Fixed wrist angle for now
        return uarm_target_abs_mm, wrist_angle_uarm_deg

    '''

    # In mapper_rotate.py, inside UArmMimicControllerWithViz class:

    def calculate_uarm_target_for_frame(self, frame_idx: int, is_initial_move: bool = False) -> Optional[
        Tuple[np.ndarray, float]]:
        """
        Calculates the uArm target position and wrist angle for a given human pose frame.
        Includes debug prints for vector lengths.
        Human CS (Left-Handed): H_x: Front(+)/Back(-), H_y: Left(+)/Right(-), H_z: Up(+)/Down(-)
        uArm SDK CS (Right-Handed): SDK_X: Front(+), SDK_Y: Up(+), SDK_Z: Right(+)
        """
        if self.raw_human_pose_sequence is None or self.key_joint_indices is None: return None
        if not (0 <= frame_idx < self.raw_human_pose_sequence.shape[0]): return None

        shoulder_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['shoulder']]
        elbow_pos_raw = self.raw_human_pose_sequence[
            frame_idx, self.key_joint_indices['elbow']]  # Needed for arm length
        wrist_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['wrist']]

        # --- Human arm vectors calculation (in input units, e.g., meters) ---
        upper_arm_vec_raw = elbow_pos_raw - shoulder_pos_raw
        forearm_vec_raw = wrist_pos_raw - elbow_pos_raw
        human_upper_arm_length_m = np.linalg.norm(upper_arm_vec_raw)
        human_forearm_length_m = np.linalg.norm(forearm_vec_raw)
        total_human_arm_length_m = human_upper_arm_length_m + human_forearm_length_m

        wrist_vec_in_raw_human_shoulder_frame = wrist_pos_raw - shoulder_pos_raw  # Vector from shoulder to wrist in Human CS
        human_shoulder_to_wrist_vec_length_m = np.linalg.norm(wrist_vec_in_raw_human_shoulder_frame)

        current_dynamic_scale_factor: float
        if total_human_arm_length_m < MIN_HUMAN_ARM_LENGTH_FOR_SCALING_M or abs(total_human_arm_length_m) < 1e-6:
            current_dynamic_scale_factor = FALLBACK_SCALE_FACTOR_M_TO_MM
            # print(f"Frame {frame_idx}: Using fallback scale factor: {current_dynamic_scale_factor}") # Optional debug
        else:
            current_dynamic_scale_factor = UARM_TARGET_MAPPED_ARM_LENGTH_MM / total_human_arm_length_m
        self.last_calculated_dynamic_scale_factor = current_dynamic_scale_factor

        # Scale the human shoulder-to-wrist vector to millimeters for uArm mapping.
        # This vector is still in the Human Coordinate System's orientation.
        scaled_human_s_to_w_vec_mm = wrist_vec_in_raw_human_shoulder_frame * current_dynamic_scale_factor

        # Components of the scaled human shoulder-to-wrist vector in Human CS
        # H_x_sw: human_X component (front/back relative to human shoulder)
        # H_y_sw: human_Y component (left/right relative to human shoulder)
        # H_z_sw: human_Z component (up/down relative to human shoulder)
        H_x_sw = scaled_human_s_to_w_vec_mm[0]
        H_y_sw = scaled_human_s_to_w_vec_mm[1]
        H_z_sw = scaled_human_s_to_w_vec_mm[2]

        # Initialize relative SDK coordinates (these are the components of the vector from
        # the uArm's conceptual shoulder to its effector, in the uArm SDK CS)
        sdk_x_rel, sdk_y_rel, sdk_z_rel = 0.0, 0.0, 0.0

        if UARM_PLACEMENT_MODE == 'upright':
            # uArm SDK: X=Front, Y=Up, Z=Right
            # Human CS: H_x=Front, H_y=Left, H_z=Up
            # TRACKED_ARM is assumed to be 'right' for this example, adjust if needed for 'left'
            if TRACKED_ARM == 'right':
                sdk_x_rel = H_x_sw  # Human Front (+H_x) -> uArm Front (+SDK_X)
                sdk_y_rel = H_z_sw  # Human Up    (+H_z) -> uArm Up   (+SDK_Y)
                sdk_z_rel = -H_y_sw  # Human Left  (+H_y) -> uArm Left (-SDK_Z), since SDK_Z+ is Right
            elif TRACKED_ARM == 'left':
                # For left arm, human's left (+H_y) might still map to uArm's left (-SDK_Z)
                # Human's forward (+H_x) still maps to uArm's forward (+SDK_X)
                # Human's up (+H_z) still maps to uArm's up (+SDK_Y)
                # This mapping might feel more natural if the robot is mirroring.
                # If the robot is an extension of the same side, then H_y might map to +SDK_Z.
                # Let's assume mirroring for now.
                sdk_x_rel = H_x_sw
                sdk_y_rel = H_z_sw
                sdk_z_rel = -H_y_sw  # Human Left (+H_y) means wrist is to the left of shoulder.
                # If uArm mirrors, its effector should be to its left (-SDK_Z).
            else:
                print(f"Error: Invalid TRACKED_ARM '{TRACKED_ARM}'.")
                return None

        elif UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
            # In this mode, the uArm is on its side.
            # uArm SDK (still X=Front, Y=Up, Z=Right relative to its own base)
            # But now the physical orientation in the world is different.
            # Your previous comments implied:
            #   Robot's "Vertical" (world up/down) is controlled by Human Z (up/down)
            #   Robot's "Reach" (world front/back) is controlled by Human X (front/back)
            #   Robot's "Side-to-Side" (world left/right) is controlled by Human Y (left/right)

            # To achieve this with SDK (X_Front, Y_Up, Z_Right):
            # If robot is on its 'back' panel, its SDK Y-axis (Up) points horizontally to one side.
            # If robot is on its 'side' panel (e.g., the one opposite to the motor ports),
            # its SDK Y-axis (Up) might point vertically upwards in the world.
            # Let's assume for 'side_mounted_native_x_cw90' the robot's SDK Y-axis becomes the world's vertical axis.
            # And the robot's SDK X-axis (Front) becomes the world's reach axis.
            # And the robot's SDK Z-axis (Right) becomes the world's side-to-side axis.

            if TRACKED_ARM == 'right':
                sdk_x_rel = H_x_sw  # Human Front (+H_x) -> uArm SDK X (Reach)
                sdk_y_rel = H_z_sw  # Human Up    (+H_z) -> uArm SDK Y (Vertical movement of the arm)
                sdk_z_rel = -H_y_sw  # Human Left  (+H_y) -> uArm SDK Z (Side-to-side, to uArm's left)
            elif TRACKED_ARM == 'left':
                sdk_x_rel = H_x_sw
                sdk_y_rel = H_z_sw
                sdk_z_rel = -H_y_sw  # Assuming mirroring
            else:
                print(f"Error: Invalid TRACKED_ARM '{TRACKED_ARM}'.")
                return None
        else:
            print(f"Error: Unknown UARM_PLACEMENT_MODE '{UARM_PLACEMENT_MODE}'.")
            return None

        # uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec is the vector from uArm's conceptual shoulder
        # to its effector, expressed in the uArm SDK's coordinate system.
        uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec = np.array([sdk_x_rel, sdk_y_rel, sdk_z_rel])

        # --- DEBUG PRINTING ---
        print_debug_this_frame = is_initial_move or (frame_idx % TARGET_FPS == 0)
        if print_debug_this_frame:
            print(f"\n--- Frame {frame_idx} Debug Info {'(Initial Move)' if is_initial_move else ''} ---")
            print(f"  Human CS (Input - Left-Handed: X-Front, Y-Left, Z-Up)")
            print(
                f"    Raw Shoulder Pos (m): ({shoulder_pos_raw[0]:.3f}, {shoulder_pos_raw[1]:.3f}, {shoulder_pos_raw[2]:.3f})")
            print(f"    Raw Wrist Pos (m): ({wrist_pos_raw[0]:.3f}, {wrist_pos_raw[1]:.3f}, {wrist_pos_raw[2]:.3f})")
            print(
                f"    Shoulder-to-Wrist Vec (Human CS, m): ({wrist_vec_in_raw_human_shoulder_frame[0]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[1]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[2]:.3f})")
            print(f"    Shoulder-to-Wrist Length (m): {human_shoulder_to_wrist_vec_length_m:.4f}")
            print(f"  Scaling & Mapping")
            print(f"    Dynamic Scale Factor (mm/m): {current_dynamic_scale_factor:.2f}")
            print(
                f"    Scaled Human S-to-W Vec (Human CS, mm): (H_x_sw:{H_x_sw:.1f}, H_y_sw:{H_y_sw:.1f}, H_z_sw:{H_z_sw:.1f})")
            print(f"  uArm SDK CS (Output - Right-Handed: X-Front, Y-Up, Z-Right)")
            print(f"    Conceptual Shoulder Offset (SDK CS, mm): {UARM_SHOULDER_ORIGIN_OFFSET}")
            print(
                f"    Effector Relative to Conceptual Shoulder (SDK CS, mm): (sdk_x_rel:{sdk_x_rel:.1f}, sdk_y_rel:{sdk_y_rel:.1f}, sdk_z_rel:{sdk_z_rel:.1f})")

        # Final uArm target in its absolute SDK coordinate system (effector position)
        # This is: uArm_Conceptual_Shoulder_Position_In_SDK_CS + Effector_Relative_Vector_In_SDK_CS
        uarm_target_abs_mm_before_clip = UARM_SHOULDER_ORIGIN_OFFSET + uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec
        uarm_target_abs_mm = uarm_target_abs_mm_before_clip.copy()

        # Clipping to uArm's reachable workspace (in SDK coordinates)
        # !!! THESE LIMITS ARE CRUCIAL AND DEPEND ON YOUR UARM MODEL AND MOUNTING !!!
        # You MUST verify these against your uArm's actual workspace.
        # For uArm Swift Pro (example limits, X:front, Y:up, Z:right):
        min_x, max_x = 50, 320  # Reach forward/backward
        min_y, max_y = -180, 220  # Height (can go below base a bit)
        min_z, max_z = -180, 180  # Side to side (relative to center line)

        # Adjust clipping based on UARM_PLACEMENT_MODE if the conceptual meaning of limits changes
        # However, the clipping here should ALWAYS be in the uArm SDK's native X, Y, Z.
        uarm_target_abs_mm[0] = np.clip(uarm_target_abs_mm[0], min_x, max_x)  # SDK X
        uarm_target_abs_mm[1] = np.clip(uarm_target_abs_mm[1], min_y, max_y)  # SDK Y
        uarm_target_abs_mm[2] = np.clip(uarm_target_abs_mm[2], min_z, max_z)  # SDK Z

        if print_debug_this_frame:
            if not np.array_equal(uarm_target_abs_mm, uarm_target_abs_mm_before_clip):
                print(f"  Clipping Applied:")
                print(
                    f"    Target BEFORE clip (SDK CS, mm): ({uarm_target_abs_mm_before_clip[0]:.1f}, {uarm_target_abs_mm_before_clip[1]:.1f}, {uarm_target_abs_mm_before_clip[2]:.1f})")
                print(
                    f"    Target AFTER clip  (SDK CS, mm): ({uarm_target_abs_mm[0]:.1f}, {uarm_target_abs_mm[1]:.1f}, {uarm_target_abs_mm[2]:.1f})")
            else:
                print(
                    f"    Target (SDK CS, mm): ({uarm_target_abs_mm[0]:.1f}, {uarm_target_abs_mm[1]:.1f}, {uarm_target_abs_mm[2]:.1f}) (No clipping needed)")

        # Update human wrist trail for visualization (using pinned/visualization coordinates)
        if self.human_pose_sequence_for_viz is not None and 0 <= frame_idx < self.human_pose_sequence_for_viz.shape[0]:
            current_human_wrist_pos_viz = self.human_pose_sequence_for_viz[frame_idx, self.key_joint_indices['wrist']]
            self.human_wrist_trail_for_viz.append(current_human_wrist_pos_viz.copy())

        wrist_angle_uarm_deg = 90.0  # Fixed wrist angle for now
        return uarm_target_abs_mm, wrist_angle_uarm_deg

    # ... (rest of the UArmMimicControllerWithViz class: _uarm_control_loop, start_mimicry, stop_mimicry, cleanup)
    # You should keep those methods as they were, as this change only affects the coordinate transformation.

    def _uarm_control_loop(self, start_frame_offset: int = 0):
        """Internal method for the uArm control thread."""
        if self.raw_human_pose_sequence is None:
            print("[uArm Thread] Error: Raw human pose data not loaded.")
            return

        num_frames = self.raw_human_pose_sequence.shape[0]
        print(f"[uArm Thread] Starting uArm control loop from frame {start_frame_offset} for "
              f"{num_frames - start_frame_offset} frames at ~{TARGET_FPS} FPS...")

        for frame_idx in range(start_frame_offset, num_frames):
            if self.stop_thread_flag.is_set() or not self.swift or not self.swift.connected:
                print("[uArm Thread] Stopping uArm control loop.")
                break

            loop_start_time = time.perf_counter()
            self.current_human_frame_idx = frame_idx  # Update for visualization thread

            targets = self.calculate_uarm_target_for_frame(frame_idx)

            if targets:
                uarm_target_abs_mm, wrist_angle_uarm_deg = targets

                if len(self.latest_uarm_target_abs_mm) > 0: self.latest_uarm_target_abs_mm.popleft()
                self.latest_uarm_target_abs_mm.append(uarm_target_abs_mm.copy())
                self.uarm_target_trail_mm.append(uarm_target_abs_mm.copy())

                self.swift.set_position(x=uarm_target_abs_mm[0], y=uarm_target_abs_mm[1], z=uarm_target_abs_mm[2],
                                        speed=MOVEMENT_SPEED_MMPM, wait=False)
                self.swift.set_wrist(angle=wrist_angle_uarm_deg, speed=WRIST_SPEED_DEGPM, wait=False)

            # Log uArm commands less frequently than debug prints in calculate_uarm_target_for_frame
            if frame_idx % (TARGET_FPS * 2) == 0:  # Log approx every 2 seconds
                log_msg_ctrl = f"[uArm Thread] Sent cmd for Frame {frame_idx}/{num_frames}."
                if targets: log_msg_ctrl += f" Target uArm (Native mm): ({targets[0][0]:.0f}, {targets[0][1]:.0f}, {targets[0][2]:.0f})"
                # log_msg_ctrl += f" DynScale: {self.last_calculated_dynamic_scale_factor:.2f}" # Already printed in calculate
                print(log_msg_ctrl)

            elapsed_time = time.perf_counter() - loop_start_time
            time_to_wait = (1.0 / TARGET_FPS) - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        print("[uArm Thread] uArm control loop finished.")
        if self.swift and self.swift.connected:
            self.swift.flush_cmd(timeout=5, wait_stop=True)

    def start_mimicry(self, start_from_frame_one: bool = False):
        """Starts the uArm control thread."""
        if not self.is_connected_and_ready or self.raw_human_pose_sequence is None:
            print("Error: Cannot start mimicry. uArm not ready or human data not loaded.")
            return False
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Info: Mimicry thread already running.");
            return True

        self.stop_thread_flag.clear()
        # If frame 0 was used for initial pose, start control loop from frame 1
        loop_start_offset = 1 if start_from_frame_one else 0

        self.uarm_control_thread = threading.Thread(target=self._uarm_control_loop, args=(loop_start_offset,),
                                                    daemon=True)
        self.uarm_control_thread.start()
        print(f"uArm control thread started (loop will begin at frame index {loop_start_offset}).")
        return True

    def stop_mimicry(self):
        """Stops the uArm control thread."""
        print("Attempting to stop uArm mimicry...")
        self.stop_thread_flag.set()
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            self.uarm_control_thread.join(timeout=7)  # Wait for thread to finish
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():  # Check again if join timed out
            print("Warning: uArm control thread did not join in time.")
        self.uarm_control_thread = None
        print("Mimicry stop sequence initiated.")

    def cleanup(self):
        """Stops mimicry, resets uArm, and disconnects."""
        self.stop_mimicry()  # Ensure thread is stopped first
        if self.swift and self.swift.connected:
            print("\nResetting arm before disconnecting...")
            try:
                # Consider moving to a known safe position before reset if necessary
                # self.swift.set_position(x=200, y=0, z=150, speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=10)
                self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=15)
                self.swift.disconnect()
            except Exception as e:
                print(f"Error during uArm cleanup (reset/disconnect): {e}")
        self.is_connected_and_ready = False  # Update status
        self.swift = None
        print("Cleanup complete.")


# --- Matplotlib Animation Global Variables & Functions ---
fig_anim: Optional[plt.Figure] = None
ax_anim: Optional[Axes3D] = None  # Matplotlib 3D axis
# Human skeleton artists
human_full_scatter_viz: Optional[plt.Artist] = None
human_full_lines_viz: List[plt.Line2D] = []
# Tracked arm artists (human)
tracked_arm_scatter_viz: Optional[plt.Artist] = None
tracked_arm_lines_viz: List[plt.Line2D] = []
# uArm visualization artists
uarm_base_viz: Optional[plt.Artist] = None  # Conceptual uArm base (physical 0,0,0 of uArm)
uarm_target_point_viz: Optional[plt.Line2D] = None  # uArm effector target
uarm_trail_line_viz: Optional[plt.Line2D] = None  # uArm effector trail
# Human wrist trail artist
human_wrist_trail_line_viz: Optional[plt.Line2D] = None
# Controller instance for animation access
animation_controller: Optional[UArmMimicControllerWithViz] = None


def get_rotation_matrix_native_to_viz() -> np.ndarray:
    """
    Calculates the rotation matrix to transform coordinates from uArm NATIVE space
    to the common VISUALIZATION space (aligned with pinned human model: X-right, Y-forward, Z-up).
    This depends on UARM_PLACEMENT_MODE.
    """
    if UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
        # uArm Native CS (X_robot_up, Y_robot_left, Z_robot_reach) for this mount.
        # Viz CS (X_viz_right, Y_viz_fwd, Z_viz_up).
        # Mapping:
        #   Native X_up    -> Viz Z_up
        #   Native Y_left  -> Viz -X_right (so Native -Y is Viz X)
        #   Native Z_reach -> Viz Y_fwd
        # P_viz = R @ P_native
        # Viz_X = -Native_Y
        # Viz_Y =  Native_Z
        # Viz_Z =  Native_X
        R_native_to_viz = np.array([
            [0, -1, 0],  # Viz X = -Native Y
            [0, 0, 1],  # Viz Y =  Native Z
            [1, 0, 0]  # Viz Z =  Native X
        ])
    elif UARM_PLACEMENT_MODE == 'upright':
        # uArm Native CS (X_robot_fwd, Y_robot_left, Z_robot_up) for standard upright.
        # Viz CS (X_viz_right, Y_viz_fwd, Z_viz_up).
        # Mapping:
        #   Native X_fwd   -> Viz Y_fwd
        #   Native Y_left  -> Viz -X_right
        #   Native Z_up    -> Viz Z_up
        R_native_to_viz = np.array([
            [0, -1, 0],  # Viz X = -Native Y
            [1, 0, 0],  # Viz Y =  Native X
            [0, 0, 1]  # Viz Z =  Native Z
        ])
    else:  # Default to identity if mode is unknown, though this shouldn't happen
        R_native_to_viz = np.eye(3)
    return R_native_to_viz


def init_animation_artists():
    """Initializes all artists for the Matplotlib animation."""
    global human_full_scatter_viz, human_full_lines_viz, tracked_arm_scatter_viz, tracked_arm_lines_viz
    global uarm_base_viz, uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz, ax_anim

    if not all([animation_controller,
                animation_controller.human_pose_sequence_for_viz is not None,
                animation_controller.key_joint_indices,
                animation_controller.initial_human_root_pos_raw is not None,
                animation_controller.initial_human_shoulder_pos_raw is not None,
                animation_controller.human_skeleton_parents is not None]):
        print("Warning: Animation controller or necessary data not fully ready for init_animation_artists.")
        return []

    ax_anim.clear()  # Clear previous artists if any
    viz_data = animation_controller.human_pose_sequence_for_viz

    # Auto-scaling plot limits based on human pose data
    margin = 0.3
    x_min, x_max = viz_data[..., 0].min() - margin, viz_data[..., 0].max() + margin
    y_min, y_max = viz_data[..., 1].min() - margin, viz_data[..., 1].max() + margin
    z_min, z_max = viz_data[..., 2].min() - margin, viz_data[..., 2].max() + margin
    ax_anim.set_xlim(x_min, x_max);
    ax_anim.set_ylim(y_min, y_max);
    ax_anim.set_zlim(z_min, z_max)

    # Make axes equal for better 3D perception
    axis_ranges = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_r = axis_ranges[axis_ranges > 1e-6].max() if (axis_ranges > 1e-6).any() else 1.0
    mid_x, mid_y, mid_z = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5, (z_max + z_min) * 0.5
    ax_anim.set_xlim(mid_x - max_r / 2, mid_x + max_r / 2)
    ax_anim.set_ylim(mid_y - max_r / 2, mid_y + max_r / 2)
    ax_anim.set_zlim(mid_z - max_r / 2, mid_z + max_r / 2)

    ax_anim.set_xlabel("X (Pinned Human / Viz)");
    ax_anim.set_ylabel("Y (Pinned Human / Viz)");
    ax_anim.set_zlabel("Z (Pinned Human / Viz)")
    ax_anim.set_title("Human (Pinned) & uArm Conceptualization");
    ax_anim.view_init(elev=20., azim=-75)

    # --- Initialize Human Skeleton Artists ---
    initial_pose_viz = animation_controller.human_pose_sequence_for_viz[0]
    human_full_scatter_viz = ax_anim.scatter(
        initial_pose_viz[:, 0], initial_pose_viz[:, 1], initial_pose_viz[:, 2],
        s=15, c='gray', alpha=0.6, label='Full Human (Pinned)'
    )
    human_full_lines_viz.clear()
    for i in range(initial_pose_viz.shape[0]):
        if animation_controller.human_skeleton_parents[i] != -1:  # Not a root joint
            p_idx = animation_controller.human_skeleton_parents[i]
            line, = ax_anim.plot(initial_pose_viz[[i, p_idx], 0], initial_pose_viz[[i, p_idx], 1],
                                 initial_pose_viz[[i, p_idx], 2], 'k-', alpha=0.2, lw=1.5)
            human_full_lines_viz.append(line)

    # --- Initialize Tracked Arm Artists (Human) ---
    s_idx = animation_controller.key_joint_indices['shoulder']
    e_idx = animation_controller.key_joint_indices['elbow']
    w_idx = animation_controller.key_joint_indices['wrist']
    tracked_arm_joint_indices = [s_idx, e_idx, w_idx]
    tracked_arm_viz_data = initial_pose_viz[tracked_arm_joint_indices]
    tracked_arm_scatter_viz = ax_anim.scatter(
        tracked_arm_viz_data[:, 0], tracked_arm_viz_data[:, 1], tracked_arm_viz_data[:, 2],
        s=40, c='red', label=f'Tracked {TRACKED_ARM} Arm (Pinned)', zorder=5
    )
    tracked_arm_lines_viz.clear()
    line_shoulder_elbow, = ax_anim.plot(initial_pose_viz[[s_idx, e_idx], 0], initial_pose_viz[[s_idx, e_idx], 1],
                                        initial_pose_viz[[s_idx, e_idx], 2], 'r-', lw=4, zorder=4)
    line_elbow_wrist, = ax_anim.plot(initial_pose_viz[[e_idx, w_idx], 0], initial_pose_viz[[e_idx, w_idx], 1],
                                     initial_pose_viz[[e_idx, w_idx], 2], 'r-', lw=4, zorder=4)
    tracked_arm_lines_viz.extend([line_shoulder_elbow, line_elbow_wrist])

    # --- Initialize uArm Related Visualization Artists ---
    current_viz_scale_factor_init = animation_controller.last_calculated_dynamic_scale_factor
    if abs(current_viz_scale_factor_init) < 1e-9: current_viz_scale_factor_init = FALLBACK_SCALE_FACTOR_M_TO_MM

    # Human shoulder in the pinned visualization CS (this is the reference for uArm placement)
    human_shoulder_in_pinned_viz = animation_controller.initial_human_shoulder_pos_raw - \
                                   animation_controller.initial_human_root_pos_raw

    R_native_to_viz = get_rotation_matrix_native_to_viz()  # Get rotation matrix based on placement mode

    # UARM_SHOULDER_ORIGIN_OFFSET is from uArm physical base (0,0,0) to its conceptual shoulder, in uArm Native CS.
    # Transform this offset vector into the Visualization CS.
    uarm_conceptual_shoulder_offset_in_viz_cs_mm = R_native_to_viz @ UARM_SHOULDER_ORIGIN_OFFSET
    uarm_conceptual_shoulder_offset_in_viz_cs_human_units = \
        uarm_conceptual_shoulder_offset_in_viz_cs_mm / current_viz_scale_factor_init

    # The uArm's physical base (0,0,0 in its own CS) is located such that:
    # human_shoulder_in_pinned_viz = conceptual_uarm_physical_base_in_viz + uarm_conceptual_shoulder_offset_in_viz_cs_human_units
    # So, conceptual_uarm_physical_base_in_viz = human_shoulder_in_pinned_viz - uarm_conceptual_shoulder_offset_in_viz_cs_human_units
    conceptual_uarm_physical_base_in_viz_init = \
        human_shoulder_in_pinned_viz - uarm_conceptual_shoulder_offset_in_viz_cs_human_units

    uarm_base_viz = ax_anim.scatter(
        [conceptual_uarm_physical_base_in_viz_init[0]], [conceptual_uarm_physical_base_in_viz_init[1]],
        [conceptual_uarm_physical_base_in_viz_init[2]],
        s=150, c='purple', marker='s', label='uArm Base (Conceptual)', zorder=9, edgecolors='k'
    )

    # Initial uArm effector target point
    initial_uarm_effector_plot_point_viz = conceptual_uarm_physical_base_in_viz_init  # Default if no target
    if animation_controller.latest_uarm_target_abs_mm:  # Populated by calculate_uarm_target_for_frame(0)
        uarm_target_effector_native_mm = animation_controller.latest_uarm_target_abs_mm[0]

        # Transform uArm effector's absolute native position to Viz CS relative to uArm's physical base.
        # The target_effector_native_mm is an ABSOLUTE position in uArm Native CS.
        # To plot it relative to conceptual_uarm_physical_base_in_viz, we transform it directly.
        uarm_target_effector_in_viz_cs_mm = R_native_to_viz @ uarm_target_effector_native_mm
        uarm_target_effector_in_viz_cs_human_units = \
            uarm_target_effector_in_viz_cs_mm / current_viz_scale_factor_init

        initial_uarm_effector_plot_point_viz = conceptual_uarm_physical_base_in_viz_init + \
                                               uarm_target_effector_in_viz_cs_human_units

    uarm_target_point_viz, = ax_anim.plot(
        [initial_uarm_effector_plot_point_viz[0]], [initial_uarm_effector_plot_point_viz[1]],
        [initial_uarm_effector_plot_point_viz[2]],
        'go', ms=12, mec='k', label='uArm Effector (Conceptual)', zorder=10
    )

    uarm_trail_line_viz, = ax_anim.plot([], [], [], 'g--', alpha=0.7, lw=1.5, label='uArm Effector Trail', zorder=3)
    human_wrist_trail_line_viz, = ax_anim.plot([], [], [], 'm:', alpha=0.7, lw=1.5,
                                               label=f'Human {TRACKED_ARM} Wrist Trail (Pinned)', zorder=3)

    ax_anim.legend(loc='upper left', fontsize='small');
    plt.tight_layout()

    # Consolidate all artists to be returned
    artists = ([human_full_scatter_viz] + human_full_lines_viz +
               [tracked_arm_scatter_viz] + tracked_arm_lines_viz +
               [human_wrist_trail_line_viz])  # Start with human artists
    if uarm_base_viz: artists.append(uarm_base_viz)
    if uarm_target_point_viz: artists.append(uarm_target_point_viz)
    if uarm_trail_line_viz: artists.append(uarm_trail_line_viz)
    return artists


def update_animation_frame(frame_num_anim: int):  # frame_num_anim is provided by FuncAnimation
    """Updates all artists for the current animation frame."""
    global human_full_scatter_viz, human_full_lines_viz, tracked_arm_scatter_viz, tracked_arm_lines_viz
    global uarm_base_viz, uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz

    # Initial checks for readiness
    if not all([animation_controller,
                animation_controller.human_pose_sequence_for_viz is not None,
                animation_controller.key_joint_indices,
                animation_controller.initial_human_root_pos_raw is not None,
                animation_controller.initial_human_shoulder_pos_raw is not None,
                animation_controller.human_skeleton_parents is not None]):
        # Return existing artists if some are initialized, otherwise empty
        current_artists = [human_full_scatter_viz] + human_full_lines_viz if human_full_scatter_viz else []
        if tracked_arm_scatter_viz: current_artists.extend([tracked_arm_scatter_viz] + tracked_arm_lines_viz)
        if uarm_base_viz: current_artists.append(uarm_base_viz)
        if uarm_target_point_viz: current_artists.append(uarm_target_point_viz)
        if uarm_trail_line_viz: current_artists.append(uarm_trail_line_viz)
        if human_wrist_trail_line_viz: current_artists.append(human_wrist_trail_line_viz)
        return filter(None, current_artists)  # Filter out None items

    current_data_frame_idx = animation_controller.current_human_frame_idx  # Use index from control loop
    max_viz_frames = animation_controller.human_pose_sequence_for_viz.shape[0]

    # Ensure current_data_frame_idx is valid
    if not (0 <= current_data_frame_idx < max_viz_frames):
        if animation_controller.stop_thread_flag.is_set() and current_data_frame_idx >= max_viz_frames - 1:
            current_data_frame_idx = max_viz_frames - 1  # Clamp to last frame if stopping at end
        else:  # Frame out of bounds and not at the end of a stopped sequence
            # (Same return logic as above for consistency)
            current_artists = [human_full_scatter_viz] + human_full_lines_viz if human_full_scatter_viz else []
            if tracked_arm_scatter_viz: current_artists.extend([tracked_arm_scatter_viz] + tracked_arm_lines_viz)
            if uarm_base_viz: current_artists.append(uarm_base_viz)
            if uarm_target_point_viz: current_artists.append(uarm_target_point_viz)
            if uarm_trail_line_viz: current_artists.append(uarm_trail_line_viz)
            if human_wrist_trail_line_viz: current_artists.append(human_wrist_trail_line_viz)
            return filter(None, current_artists)

    # --- Update Human Skeleton Visualization ---
    current_pose_human_viz = animation_controller.human_pose_sequence_for_viz[current_data_frame_idx]
    human_full_scatter_viz._offsets3d = (current_pose_human_viz[:, 0],
                                         current_pose_human_viz[:, 1],
                                         current_pose_human_viz[:, 2])
    for line_idx, line_artist in enumerate(human_full_lines_viz):  # Assuming order matches parent connections
        # This part needs to map line_idx to the correct joint and its parent
        # A more robust way is to iterate through joints and update their corresponding line
        joint_idx_for_line = -1
        parent_idx_for_line = -1
        current_line = 0
        for i in range(current_pose_human_viz.shape[0]):
            if animation_controller.human_skeleton_parents[i] != -1:
                if current_line == line_idx:
                    joint_idx_for_line = i
                    parent_idx_for_line = animation_controller.human_skeleton_parents[i]
                    break
                current_line += 1
        if joint_idx_for_line != -1:
            line_artist.set_data_3d(
                current_pose_human_viz[[joint_idx_for_line, parent_idx_for_line], 0],
                current_pose_human_viz[[joint_idx_for_line, parent_idx_for_line], 1],
                current_pose_human_viz[[joint_idx_for_line, parent_idx_for_line], 2]
            )

    s_idx = animation_controller.key_joint_indices['shoulder']
    e_idx = animation_controller.key_joint_indices['elbow']
    w_idx = animation_controller.key_joint_indices['wrist']
    tracked_arm_data_viz = current_pose_human_viz[[s_idx, e_idx, w_idx]]
    tracked_arm_scatter_viz._offsets3d = (tracked_arm_data_viz[:, 0],
                                          tracked_arm_data_viz[:, 1],
                                          tracked_arm_data_viz[:, 2])
    tracked_arm_lines_viz[0].set_data_3d(current_pose_human_viz[[s_idx, e_idx], 0],
                                         current_pose_human_viz[[s_idx, e_idx], 1],
                                         current_pose_human_viz[[s_idx, e_idx], 2])
    tracked_arm_lines_viz[1].set_data_3d(current_pose_human_viz[[e_idx, w_idx], 0],
                                         current_pose_human_viz[[e_idx, w_idx], 1],
                                         current_pose_human_viz[[e_idx, w_idx], 2])

    # --- Update uArm Related Visualization ---
    current_viz_scale_factor_update = animation_controller.last_calculated_dynamic_scale_factor
    if abs(current_viz_scale_factor_update) < 1e-9: current_viz_scale_factor_update = FALLBACK_SCALE_FACTOR_M_TO_MM

    human_shoulder_in_pinned_viz = animation_controller.initial_human_shoulder_pos_raw - \
                                   animation_controller.initial_human_root_pos_raw
    R_native_to_viz_update = get_rotation_matrix_native_to_viz()

    uarm_conceptual_shoulder_offset_in_viz_cs_mm = R_native_to_viz_update @ UARM_SHOULDER_ORIGIN_OFFSET
    uarm_conceptual_shoulder_offset_in_viz_cs_human_units = \
        uarm_conceptual_shoulder_offset_in_viz_cs_mm / current_viz_scale_factor_update

    conceptual_uarm_physical_base_in_viz_update = \
        human_shoulder_in_pinned_viz - uarm_conceptual_shoulder_offset_in_viz_cs_human_units

    if uarm_base_viz:
        uarm_base_viz._offsets3d = ([conceptual_uarm_physical_base_in_viz_update[0]],
                                    [conceptual_uarm_physical_base_in_viz_update[1]],
                                    [conceptual_uarm_physical_base_in_viz_update[2]])

    if uarm_target_point_viz and animation_controller.latest_uarm_target_abs_mm:
        uarm_target_effector_native_mm = animation_controller.latest_uarm_target_abs_mm[0]
        uarm_target_effector_in_viz_cs_mm = R_native_to_viz_update @ uarm_target_effector_native_mm
        uarm_target_effector_in_viz_cs_human_units = \
            uarm_target_effector_in_viz_cs_mm / current_viz_scale_factor_update

        plot_point_uarm_effector_viz = conceptual_uarm_physical_base_in_viz_update + \
                                       uarm_target_effector_in_viz_cs_human_units
        uarm_target_point_viz.set_data_3d([plot_point_uarm_effector_viz[0]],
                                          [plot_point_uarm_effector_viz[1]],
                                          [plot_point_uarm_effector_viz[2]])

    if uarm_trail_line_viz and animation_controller.uarm_target_trail_mm:
        trail_data_native_mm = np.array(list(animation_controller.uarm_target_trail_mm))
        if trail_data_native_mm.size > 0:
            trail_plot_points_viz = []
            for point_native_mm in trail_data_native_mm:  # point_native_mm is an ABSOLUTE uArm effector position
                point_effector_in_viz_cs_mm = R_native_to_viz_update @ point_native_mm
                point_effector_in_viz_cs_human_units = \
                    point_effector_in_viz_cs_mm / current_viz_scale_factor_update
                plot_point = conceptual_uarm_physical_base_in_viz_update + \
                             point_effector_in_viz_cs_human_units
                trail_plot_points_viz.append(plot_point)
            if trail_plot_points_viz:
                trail_plot_points_viz_np = np.array(trail_plot_points_viz)
                uarm_trail_line_viz.set_data_3d(trail_plot_points_viz_np[:, 0],
                                                trail_plot_points_viz_np[:, 1],
                                                trail_plot_points_viz_np[:, 2])

    # Update Human Wrist Trail (already in viz coordinates)
    if human_wrist_trail_line_viz and animation_controller.human_wrist_trail_for_viz:
        wrist_trail_data_viz = np.array(list(animation_controller.human_wrist_trail_for_viz))
        if wrist_trail_data_viz.size > 0:
            human_wrist_trail_line_viz.set_data_3d(wrist_trail_data_viz[:, 0],
                                                   wrist_trail_data_viz[:, 1],
                                                   wrist_trail_data_viz[:, 2])

    ax_anim.set_title(f"Frame: {current_data_frame_idx} / {max_viz_frames - 1}")

    artists_to_return = ([human_full_scatter_viz] + human_full_lines_viz +
                         [tracked_arm_scatter_viz] + tracked_arm_lines_viz)
    if uarm_base_viz: artists_to_return.append(uarm_base_viz)
    if uarm_target_point_viz: artists_to_return.append(uarm_target_point_viz)
    if uarm_trail_line_viz: artists_to_return.append(uarm_trail_line_viz)
    if human_wrist_trail_line_viz: artists_to_return.append(human_wrist_trail_line_viz)
    return filter(None, artists_to_return)  # Filter out any None artists before returning


def main_with_viz():
    """Main function to run the uArm mimicry with visualization."""
    global fig_anim, ax_anim, animation_controller  # Allow modification of global figure/axis objects

    print(f"--- Human Arm to uArm Mimicry (Placement: {UARM_PLACEMENT_MODE}, Scaling: Dynamic) ---")
    try:
        # Determine project root directory to reliably find data files
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:  # Fallback if __file__ is not defined (e.g., in some interactive environments)
        project_root_dir = os.getcwd()
    npz_full_path = os.path.join(project_root_dir, NPZ_FILE_RELATIVE_PATH)

    if not os.path.exists(npz_full_path):
        print(f"Error: NPZ data file not found: '{npz_full_path}'")
        print(
            f"Please ensure '{NPZ_FILE_RELATIVE_PATH}' exists relative to the script location or current working directory.")
        return

    animation_controller = UArmMimicControllerWithViz(port=UARM_SERIAL_PORT)
    try:
        if not animation_controller.load_human_data(npz_full_path, SKELETON_TYPE, TRACKED_ARM):
            print("Failed to load human data. Exiting.")
            return

        uarm_successfully_connected = animation_controller.connect_uarm()
        if not uarm_successfully_connected:
            print("Warning: uArm connection failed. Proceeding with visualization only (if possible).")
            animation_controller.is_connected_and_ready = False  # Explicitly ensure this is false

        # Pre-calculate target for frame 0 to populate last_calculated_dynamic_scale_factor
        # and initial trail data for visualization, regardless of uArm connection.
        if animation_controller.raw_human_pose_sequence is not None and \
                animation_controller.raw_human_pose_sequence.shape[0] > 0:
            animation_controller.calculate_uarm_target_for_frame(0, is_initial_move=True)  # For debug prints
        else:
            print("Error: No human data loaded; cannot run animation or uArm control.")
            if animation_controller: animation_controller.cleanup(); return

        if animation_controller.is_connected_and_ready:  # Only if uArm is connected
            if not animation_controller.move_uarm_to_initial_human_pose(initial_frame_idx=0):
                print("Warning: Failed to accurately move uArm to initial human pose.")
            else:
                print("uArm successfully positioned to initial human pose (frame 0).")
            animation_controller.current_human_frame_idx = 0  # Visuals start at frame 0
            time.sleep(1)  # Pause after initial uArm positioning

        print("\nPreparing visualization...")
        fig_anim = plt.figure(figsize=(16, 12))  # Adjusted figure size
        ax_anim = fig_anim.add_subplot(111, projection='3d')

        mimicry_control_started = False
        if animation_controller.is_connected_and_ready:
            print("Starting uArm control thread...")
            if animation_controller.start_mimicry(start_from_frame_one=True):  # Start uArm control from frame 1
                mimicry_control_started = True
            else:
                print("Failed to start uArm control thread. Visualization will run without live uArm movement.")
        else:
            print("uArm not connected. Visualization will run; uArm control thread will not start.")

        num_anim_frames = animation_controller.raw_human_pose_sequence.shape[0]
        anim_interval = int(1000 / TARGET_FPS)  # Interval in milliseconds

        print("Starting animation...")
        ani = FuncAnimation(fig_anim, update_animation_frame, frames=num_anim_frames,
                            init_func=init_animation_artists, blit=False,  # blit=False often more robust for 3D
                            interval=anim_interval, repeat=False)
        plt.tight_layout()
        plt.show()  # This call blocks until the animation window is closed

        print("Plot window closed or animation finished.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred in main_with_viz: {e}")
        import traceback;
        traceback.print_exc()
    finally:
        if animation_controller:
            print("Initiating cleanup sequence...")
            animation_controller.cleanup()  # Stops thread, resets uArm, disconnects
        print("\n--- Program Finished ---")


if __name__ == '__main__':
    main_with_viz()