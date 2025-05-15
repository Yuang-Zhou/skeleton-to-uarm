import os
import sys
import time
import numpy as np
import math
import threading  # For running uArm control in a separate thread
import collections  # For deque, a thread-safe way to store recent points
from typing import Optional, Tuple, List, Dict  # Using Dict from typing

import matplotlib

matplotlib.use('TkAgg')  # Explicitly set backend, 'Qt5Agg' is also good
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Constants and Configuration ---
NPZ_FILE_RELATIVE_PATH = os.path.join("../data", "00", "joints_drc_smooth.npz")
SKELETON_TYPE = 'smpl_24'
TRACKED_ARM = 'right'  # Options: 'right' or 'left'

UARM_SERIAL_PORT = '/dev/cu.usbmodem144201'  # !!! MODIFY THIS TO YOUR ACTUAL PORT !!!
INITIAL_UARM_RESET_SPEED = 3000  # mm/min
MOVEMENT_SPEED_MMPM = 6000  # Speed for set_position in mm/min
WRIST_SPEED_DEGPM = 1000  # Speed for set_wrist in deg/min

# Critical parameters for mapping human motion to uArm space
HUMAN_TO_UARM_SCALE_FACTOR = 150.0  # Adjust based on human data units (e.g., meters) vs uArm (mm)
UARM_SHOULDER_ORIGIN_OFFSET = np.array(
    [150.0, 0.0, 100.0])  # Conceptual human shoulder origin in uArm's mm coordinate system (X, Y, Z)

TARGET_FPS = 20  # Target FPS for both uArm command sending and animation updates

# Visualization settings
UARM_TARGET_TRAIL_LENGTH = 50
HUMAN_WRIST_TRAIL_LENGTH = 50

# Root joint index for pinning the human pose visualization (e.g., Pelvis for SMPL)
HUMAN_ROOT_JOINT_IDX = 0

# --- Helper: Path Setup ---
try:
    _current_script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = _current_script_dir  # Assuming this script is in the project root

    # Path to your HPSTM 'src' directory (if kinematics_utils is still there)
    _src_path_hpstm = os.path.join(_project_root, "src")
    if os.path.isdir(_src_path_hpstm) and _src_path_hpstm not in sys.path:
        sys.path.insert(0, _src_path_hpstm)
        print(f"Info: Added HPSTM src path '{_src_path_hpstm}' to sys.path.")

    # Path to uArm SDK
    _sdk_base_dir = os.path.join(_project_root, "uarm-python-sdk")
    _sdk_actual_path = os.path.join(_sdk_base_dir, "uArm-Python-SDK-2.0")
    if os.path.isdir(os.path.join(_sdk_actual_path, "uarm")) and _sdk_actual_path not in sys.path:
        sys.path.insert(0, _sdk_actual_path)
        print(f"Info: Added uArm SDK path '{_sdk_actual_path}' to sys.path.")

    from kinematics.skeleton_utils import get_num_joints as get_total_joints_for_skeleton, get_skeleton_parents
    from uarm.wrapper import SwiftAPI

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
        """Loads a 3D pose sequence from a specified .npz file."""
        if not os.path.exists(npz_file_path):
            print(f"Error: File not found at '{npz_file_path}'.")
            return None
        try:
            data = np.load(npz_file_path)
            if expected_key not in data:
                print(
                    f"Error: Expected key '{expected_key}' not found in '{npz_file_path}'. Available keys: {list(data.keys())}")
                return None
            pose_sequence = data[expected_key]
            if pose_sequence.ndim != 3 or pose_sequence.shape[2] != 3:
                print(
                    f"Error: Data for key '{expected_key}' has unexpected shape: {pose_sequence.shape}. Expected (frames, joints, 3).")
                return None
            return pose_sequence.astype(np.float32)  # Ensure float32
        except Exception as e:
            print(f"Error loading .npz file '{npz_file_path}': {e}")
            return None


    def get_arm_joint_indices(skeleton_type: str = 'smpl_24', arm_to_track: str = 'right') -> Optional[Dict[str, int]]:
        """Retrieves joint indices for shoulder, elbow, and wrist for the specified arm."""
        if skeleton_type.lower() == 'smpl_24':
            return SMPL24_ARM_KEY_JOINTS.get(arm_to_track.lower())
        print(f"Error: Key joint indices for skeleton type '{skeleton_type}' are not defined.")
        return None


    print("Successfully imported required modules.")
except ImportError as e:
    print(f"Critical Error: Failed to import a required module: {e}")
    sys.exit(1)


# --- End Path Setup ---

class UArmMimicControllerWithViz:
    def __init__(self, port: Optional[str] = UARM_SERIAL_PORT):
        self.swift: Optional[SwiftAPI] = None
        self.port: Optional[str] = port
        self.is_connected_and_ready: bool = False
        self.key_joint_indices: Optional[Dict[str, int]] = None

        self.raw_human_pose_sequence: Optional[np.ndarray] = None  # Original data with global translation
        self.human_pose_sequence_for_viz: Optional[np.ndarray] = None  # Pinned to origin for visualization

        self.initial_human_root_pos_raw: Optional[np.ndarray] = None  # For pinning calculation
        self.initial_human_shoulder_pos_raw: Optional[np.ndarray] = None  # For conceptual uArm origin in viz

        self.uarm_control_thread: Optional[threading.Thread] = None
        self.stop_thread_flag = threading.Event()

        self.current_human_frame_idx = 0  # Updated by control thread, read by animation
        self.latest_uarm_target_abs = collections.deque(maxlen=1)  # Stores uArm target in mm
        self.uarm_target_trail = collections.deque(maxlen=UARM_TARGET_TRAIL_LENGTH)  # Stores uArm targets in mm
        self.human_wrist_trail_for_viz = collections.deque(
            maxlen=HUMAN_WRIST_TRAIL_LENGTH)  # Stores pinned human wrist pos

        self.human_skeleton_parents = get_skeleton_parents(SKELETON_TYPE)

    def connect_uarm(self) -> bool:
        """Initializes and connects to the uArm, confirming physical reset."""
        print(f"Attempting to connect to uArm on port: {self.port if self.port else 'Auto-detect'}...")
        try:
            self.swift = SwiftAPI(port=self.port)
            time.sleep(1.5)
            if not self.swift.connected:
                print("Error: SwiftAPI failed to establish a serial connection.");
                self.swift = None;
                return False

            _ = self.swift.waiting_ready(timeout=20)
            if self.swift.power_status:
                print("uArm power status is ON after waiting_ready.")
            else:
                print(
                    "Warning: uArm power status is OFF after waiting_ready. This might cause issues.");  # Proceed with caution

            device_info = self.swift.get_device_info(timeout=10)
            if device_info:
                print(f"Device Info: {device_info}")
            else:
                print("Warning: Failed to get complete device info.");

            current_mode = self.swift.get_mode(wait=True, timeout=10)
            print(f"Current uArm mode: {current_mode}")
            if current_mode != 0 and current_mode is not None:
                print(f"Setting uArm to Mode 0 (Normal Mode)...")
                self.swift.set_mode(0, wait=True, timeout=10)
                print(f"New mode: {self.swift.get_mode(wait=True, timeout=5)}")
            elif current_mode is None:
                print("Warning: Could not retrieve current uArm mode.")

            print("Resetting uArm on connect...")
            self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=25)
            pos_after_reset = self.swift.get_position(wait=True, timeout=10)
            print(f"uArm position after reset attempt: {pos_after_reset}")

            if isinstance(pos_after_reset, list) and \
                    (190 < pos_after_reset[0] < 210 and \
                     -10 < pos_after_reset[1] < 10 and \
                     140 < pos_after_reset[2] < 160):
                print("Physical reset to home position confirmed.")
                self.is_connected_and_ready = True
                return True
            else:
                print("Warning: Physical reset not confirmed at home position or get_position failed. Position:",
                      pos_after_reset)
                print("Connection considered unstable. Disconnecting.")
                self.is_connected_and_ready = False
                if self.swift: self.swift.disconnect()
                self.swift = None
                return False
        except Exception as e:
            print(f"An unexpected error occurred during uArm connection/initialization: {e}")
            if self.swift:
                try:
                    self.swift.disconnect()
                except:
                    pass
            self.swift = None;
            self.is_connected_and_ready = False;
            return False

    def load_human_data(self, npz_file_path: str, skeleton: str, arm: str) -> bool:
        """Loads human pose data and prepares raw and visualization-specific (pinned) versions."""
        self.raw_human_pose_sequence = load_pose_sequence_from_npz(npz_file_path)  # Using the inline loader

        if self.raw_human_pose_sequence is None or self.raw_human_pose_sequence.shape[0] == 0:
            print("Failed to load human data or sequence is empty.")
            return False

        self.key_joint_indices = get_arm_joint_indices(skeleton_type=skeleton, arm_to_track=arm)
        if not self.key_joint_indices:
            print(f"Error: Could not get key joint indices for skeleton '{skeleton}' and arm '{arm}'.")
            return False

        # Ensure HUMAN_ROOT_JOINT_IDX is valid for the loaded skeleton
        if not (0 <= HUMAN_ROOT_JOINT_IDX < self.raw_human_pose_sequence.shape[1]):
            print(f"Error: HUMAN_ROOT_JOINT_IDX ({HUMAN_ROOT_JOINT_IDX}) is out of bounds for "
                  f"the loaded skeleton with {self.raw_human_pose_sequence.shape[1]} joints.")
            return False

        # Store raw initial shoulder and root positions
        self.initial_human_root_pos_raw = self.raw_human_pose_sequence[0, HUMAN_ROOT_JOINT_IDX, :].copy()
        self.initial_human_shoulder_pos_raw = self.raw_human_pose_sequence[0, self.key_joint_indices['shoulder']].copy()
        print(f"Raw initial human shoulder (frame 0): {self.initial_human_shoulder_pos_raw}")
        print(f"Raw initial human root (frame 0, Idx {HUMAN_ROOT_JOINT_IDX}): {self.initial_human_root_pos_raw}")

        # Prepare data for VIZ (pinned to initial root by subtracting initial root's position from all frames)
        self.human_pose_sequence_for_viz = self.raw_human_pose_sequence - self.initial_human_root_pos_raw[np.newaxis,
                                                                          np.newaxis, :]

        print(f"Human pose data loaded. {self.raw_human_pose_sequence.shape[0]} frames.")
        print(f"Visualization data (human_pose_sequence_for_viz) created and is root-relative (pinned).")
        return True

    def calculate_uarm_target_for_frame(self, frame_idx: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Calculates uArm target (XYZ abs in mm, wrist angle in degrees) for a given human pose frame.
        Uses self.raw_human_pose_sequence for mapping calculations.
        Updates self.human_wrist_trail_for_viz with pinned wrist data.
        """
        if self.raw_human_pose_sequence is None or self.key_joint_indices is None:
            return None
        if not (0 <= frame_idx < self.raw_human_pose_sequence.shape[0]):
            return None

        # Use RAW human pose data for calculating relative vectors for uArm mapping
        current_human_shoulder_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['shoulder']]
        current_human_wrist_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['wrist']]

        wrist_vec_in_human_shoulder_frame = current_human_wrist_pos_raw - current_human_shoulder_pos_raw
        scaled_wrist_vec = wrist_vec_in_human_shoulder_frame * HUMAN_TO_UARM_SCALE_FACTOR

        # Coordinate transformation
        if TRACKED_ARM == 'right':
            uarm_target_x_relative = scaled_wrist_vec[1]  # Human Y (forward/depth) -> uArm X
            uarm_target_y_relative = -scaled_wrist_vec[0]  # Human X (rightward) -> uArm -Y (rightward)
            uarm_target_z_relative = scaled_wrist_vec[2]  # Human Z (upward) -> uArm Z
        elif TRACKED_ARM == 'left':
            uarm_target_x_relative = scaled_wrist_vec[1]
            uarm_target_y_relative = scaled_wrist_vec[0]  # Human X (leftward) -> uArm +Y (leftward)
            uarm_target_z_relative = scaled_wrist_vec[2]
        else:
            print(f"Error: Unknown TRACKED_ARM '{TRACKED_ARM}' in mapping.")
            return None

        uarm_target_abs_mm = np.array([
            uarm_target_x_relative + UARM_SHOULDER_ORIGIN_OFFSET[0],
            uarm_target_y_relative + UARM_SHOULDER_ORIGIN_OFFSET[1],
            uarm_target_z_relative + UARM_SHOULDER_ORIGIN_OFFSET[2]
        ])

        # Safety clamp for uArm target coordinates (in mm)
        uarm_target_abs_mm[0] = np.clip(uarm_target_abs_mm[0], 50, 300)
        uarm_target_abs_mm[1] = np.clip(uarm_target_abs_mm[1], -150, 150)
        uarm_target_abs_mm[2] = np.clip(uarm_target_abs_mm[2], 0,
                                        250)  # Min Z might need to be higher depending on setup

        wrist_angle_uarm_deg = 90.0  # Fixed wrist angle for now

        # Update human wrist trail for VISUALIZATION using the PINNED data
        if self.human_pose_sequence_for_viz is not None and \
                0 <= frame_idx < self.human_pose_sequence_for_viz.shape[0]:
            current_human_wrist_pos_viz = self.human_pose_sequence_for_viz[frame_idx, self.key_joint_indices['wrist']]
            self.human_wrist_trail_for_viz.append(current_human_wrist_pos_viz.copy())

        return uarm_target_abs_mm, wrist_angle_uarm_deg

    def _uarm_control_loop(self):
        """Internal method, runs in a separate thread, for sending commands to uArm."""
        if self.raw_human_pose_sequence is None:
            print("[uArm Thread] Error: Raw human pose sequence not loaded.")
            return

        num_frames = self.raw_human_pose_sequence.shape[0]
        print(f"[uArm Thread] Starting uArm control loop for {num_frames} frames at ~{TARGET_FPS} FPS...")

        for frame_idx in range(num_frames):
            if self.stop_thread_flag.is_set() or not self.swift or not self.swift.connected:
                print("[uArm Thread] Stopping uArm control loop (flag set or disconnected).")
                break

            loop_start_time = time.perf_counter()  # More precise timer
            self.current_human_frame_idx = frame_idx  # Share current frame index for animation

            targets = self.calculate_uarm_target_for_frame(frame_idx)
            if targets:
                uarm_target_abs_mm, wrist_angle_uarm_deg = targets

                # Store for visualization (uArm target is in mm)
                if len(self.latest_uarm_target_abs) > 0: self.latest_uarm_target_abs.popleft()
                self.latest_uarm_target_abs.append(uarm_target_abs_mm.copy())
                self.uarm_target_trail.append(uarm_target_abs_mm.copy())

                # Send commands to uArm
                self.swift.set_position(x=uarm_target_abs_mm[0], y=uarm_target_abs_mm[1], z=uarm_target_abs_mm[2],
                                        speed=MOVEMENT_SPEED_MMPM, wait=False)
                self.swift.set_wrist(angle=wrist_angle_uarm_deg, speed=WRIST_SPEED_DEGPM, wait=False)

            # Log periodically
            if frame_idx % (TARGET_FPS * 2) == 0:  # Log approx every 2 seconds
                log_msg = f"[uArm Thread] Frame {frame_idx}/{num_frames}."
                if targets:
                    log_msg += f" Target uArm (mm): ({targets[0][0]:.0f}, {targets[0][1]:.0f}, {targets[0][2]:.0f})"
                print(log_msg)

            # Maintain target FPS for command sending
            elapsed_time = time.perf_counter() - loop_start_time
            time_to_wait = (1.0 / TARGET_FPS) - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        print("[uArm Thread] uArm control loop finished.")
        if self.swift and self.swift.connected:
            print("[uArm Thread] Waiting for final uArm commands to process...")
            self.swift.flush_cmd(timeout=5, wait_stop=True)  # Wait for motion to stop

    def start_mimicry(self):
        """Starts the uArm control thread if not already running."""
        if not self.is_connected_and_ready or self.raw_human_pose_sequence is None:
            print("Error: Cannot start mimicry. Connect uArm and load data first.")
            return False
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Info: Mimicry thread already running.")
            return True  # Or False if you want to prevent multiple starts

        self.stop_thread_flag.clear()
        self.uarm_control_thread = threading.Thread(target=self._uarm_control_loop, daemon=True)
        self.uarm_control_thread.start()
        print("uArm control thread started.")
        return True

    def stop_mimicry(self):
        """Signals the uArm control thread to stop and waits for it to join."""
        print("Attempting to stop uArm mimicry...")
        self.stop_thread_flag.set()
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Waiting for uArm control thread to join...")
            self.uarm_control_thread.join(timeout=7)  # Increased timeout for join
            if self.uarm_control_thread.is_alive():
                print("Warning: uArm control thread did not join in time.")
        self.uarm_control_thread = None  # Clear the thread object
        print("Mimicry stop sequence initiated.")

    def cleanup(self):
        """Stops mimicry and disconnects from the uArm."""
        self.stop_mimicry()  # Ensure thread is stopped first
        if self.swift and self.swift.connected:
            print("\nResetting arm before disconnecting...")
            try:
                self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=15)
                pos = self.swift.get_position(wait=True, timeout=5)
                print(f"Final reset. Position: {pos}")
                print("Disconnecting from uArm...")
                self.swift.disconnect()
            except Exception as e:
                print(f"Error during cleanup reset/disconnect: {e}")
        elif self.swift:  # If swift object exists but not marked connected
            print("\nSwift object exists but not marked connected, attempting disconnect anyway...")
            try:
                self.swift.disconnect()
            except:
                pass  # Ignore errors if already disconnected or in a bad state
        self.is_connected_and_ready = False
        self.swift = None
        print("Cleanup complete.")


# --- Matplotlib Animation Setup ---
fig_anim: Optional[plt.Figure] = None  # Use a more specific name for the figure object
ax_anim: Optional[Axes3D] = None  # Use a more specific name for the axes object
human_full_scatter_viz = None;
human_full_lines_viz = []
tracked_arm_scatter_viz = None;
tracked_arm_lines_viz = []
uarm_target_point_viz = None;
uarm_trail_line_viz = None;
human_wrist_trail_line_viz = None
animation_controller: Optional[UArmMimicControllerWithViz] = None  # Global controller instance for animation callbacks


def init_animation_artists():
    """Initializes and returns all animation artists."""
    global human_full_scatter_viz, human_full_lines_viz, tracked_arm_scatter_viz, tracked_arm_lines_viz, \
        uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz, ax_anim

    if animation_controller is None or animation_controller.human_pose_sequence_for_viz is None or \
            animation_controller.key_joint_indices is None or animation_controller.initial_human_root_pos_raw is None:
        print("Anim Init: Controller or necessary data not ready for artist creation.")
        return []

    ax_anim.clear()  # Clear previous frame contents

    # Determine plot limits based on the root-relative human data for visualization
    viz_data = animation_controller.human_pose_sequence_for_viz
    margin = 0.3  # Margin for plot limits
    x_min, x_max = viz_data[..., 0].min() - margin, viz_data[..., 0].max() + margin
    y_min, y_max = viz_data[..., 1].min() - margin, viz_data[..., 1].max() + margin
    z_min, z_max = viz_data[..., 2].min() - margin, viz_data[..., 2].max() + margin

    ax_anim.set_xlim(x_min, x_max);
    ax_anim.set_ylim(y_min, y_max);
    ax_anim.set_zlim(z_min, z_max)
    # Attempt to make axes scales equal for a better 3D representation
    axis_ranges = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_axis_range = axis_ranges[axis_ranges > 0].max() if (axis_ranges > 0).any() else 1.0
    mid_x, mid_y, mid_z = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5, (z_max + z_min) * 0.5
    ax_anim.set_xlim(mid_x - max_axis_range / 2, mid_x + max_axis_range / 2)
    ax_anim.set_ylim(mid_y - max_axis_range / 2, mid_y + max_axis_range / 2)
    ax_anim.set_zlim(mid_z - max_axis_range / 2, mid_z + max_axis_range / 2)

    ax_anim.set_xlabel("X (Pinned Human Frame)");
    ax_anim.set_ylabel("Y (Pinned Human Frame)");
    ax_anim.set_zlabel("Z (Pinned Human Frame)")
    ax_anim.set_title("Human (Pinned) & uArm Target Mimicry");
    ax_anim.view_init(elev=20., azim=-75)

    initial_pose_human_viz = animation_controller.human_pose_sequence_for_viz[0]

    # Full human skeleton (pinned)
    human_full_scatter_viz = ax_anim.scatter(initial_pose_human_viz[:, 0], initial_pose_human_viz[:, 1],
                                             initial_pose_human_viz[:, 2], s=15, c='gray', alpha=0.6,
                                             label='Full Human (Pinned)')
    human_full_lines_viz = []
    for i in range(initial_pose_human_viz.shape[0]):
        if animation_controller.human_skeleton_parents[i] != -1:
            p_idx = animation_controller.human_skeleton_parents[i]
            line, = ax_anim.plot(initial_pose_human_viz[[i, p_idx], 0], initial_pose_human_viz[[i, p_idx], 1],
                                 initial_pose_human_viz[[i, p_idx], 2], 'k-', alpha=0.2, linewidth=1.5)
            human_full_lines_viz.append(line)

    # Tracked arm (pinned)
    s_idx, e_idx, w_idx = animation_controller.key_joint_indices['shoulder'], animation_controller.key_joint_indices[
        'elbow'], animation_controller.key_joint_indices['wrist']
    tracked_arm_joints_data_viz = initial_pose_human_viz[[s_idx, e_idx, w_idx]]
    tracked_arm_scatter_viz = ax_anim.scatter(tracked_arm_joints_data_viz[:, 0], tracked_arm_joints_data_viz[:, 1],
                                              tracked_arm_joints_data_viz[:, 2], s=40, c='red',
                                              label=f'Tracked Human {TRACKED_ARM} Arm (Pinned)', depthshade=False,
                                              zorder=5)
    line_s_e, = ax_anim.plot(initial_pose_human_viz[[s_idx, e_idx], 0], initial_pose_human_viz[[s_idx, e_idx], 1],
                             initial_pose_human_viz[[s_idx, e_idx], 2], 'r-', linewidth=4, zorder=4)
    line_e_w, = ax_anim.plot(initial_pose_human_viz[[e_idx, w_idx], 0], initial_pose_human_viz[[e_idx, w_idx], 1],
                             initial_pose_human_viz[[e_idx, w_idx], 2], 'r-', linewidth=4, zorder=4)
    tracked_arm_lines_viz = [line_s_e, line_e_w]

    # uArm target point and trail visualization (conceptual)
    uarm_target_point_viz, = ax_anim.plot([], [], [], 'go', markersize=12, markeredgecolor='k',
                                          label='uArm Target (Conceptual)', zorder=10)
    uarm_trail_line_viz, = ax_anim.plot([], [], [], 'g--', alpha=0.7, linewidth=1.5, label='uArm Target Trail',
                                        zorder=3)
    human_wrist_trail_line_viz, = ax_anim.plot([], [], [], 'm:', alpha=0.7, linewidth=1.5,
                                               label=f'Human {TRACKED_ARM} Wrist Trail (Pinned)', zorder=3)

    ax_anim.legend(loc='upper left', fontsize='small');
    plt.tight_layout()

    return ([human_full_scatter_viz] + human_full_lines_viz +
            [tracked_arm_scatter_viz] + tracked_arm_lines_viz +
            [uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz])


def update_animation_frame(frame_num_anim):  # frame_num_anim is from FuncAnimation
    """Updates artists for each animation frame."""
    # Use global artists defined in init_animation_artists
    global human_full_scatter_viz, human_full_lines_viz, tracked_arm_scatter_viz, tracked_arm_lines_viz, \
        uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz

    if animation_controller is None or animation_controller.human_pose_sequence_for_viz is None or \
            animation_controller.key_joint_indices is None:
        return []  # Should not happen if init was successful

    # Use frame index from the uArm control thread for data consistency
    current_data_frame_idx = animation_controller.current_human_frame_idx

    # Ensure current_data_frame_idx is valid
    max_viz_frames = animation_controller.human_pose_sequence_for_viz.shape[0]
    if not (0 <= current_data_frame_idx < max_viz_frames):
        if animation_controller.stop_thread_flag.is_set() and current_data_frame_idx >= max_viz_frames - 1:
            current_data_frame_idx = max_viz_frames - 1  # Hold last valid frame if thread stopped at end
        else:
            # print(f"Anim Update: Data frame index {current_data_frame_idx} out of bounds for viz data ({max_viz_frames} frames). Anim frame: {frame_num_anim}")
            return []  # Return empty if data not ready or out of sync

    current_pose_human_viz = animation_controller.human_pose_sequence_for_viz[current_data_frame_idx]

    # Update full human skeleton (pinned)
    human_full_scatter_viz._offsets3d = (
    current_pose_human_viz[:, 0], current_pose_human_viz[:, 1], current_pose_human_viz[:, 2])
    line_idx = 0
    for i in range(current_pose_human_viz.shape[0]):
        if animation_controller.human_skeleton_parents[i] != -1:
            p_idx = animation_controller.human_skeleton_parents[i]
            human_full_lines_viz[line_idx].set_data_3d(current_pose_human_viz[[i, p_idx], 0],
                                                       current_pose_human_viz[[i, p_idx], 1],
                                                       current_pose_human_viz[[i, p_idx], 2])
            line_idx += 1

    # Update tracked arm (pinned)
    s_idx, e_idx, w_idx = animation_controller.key_joint_indices['shoulder'], animation_controller.key_joint_indices[
        'elbow'], animation_controller.key_joint_indices['wrist']
    tracked_arm_data_viz = current_pose_human_viz[[s_idx, e_idx, w_idx]]
    tracked_arm_scatter_viz._offsets3d = (
    tracked_arm_data_viz[:, 0], tracked_arm_data_viz[:, 1], tracked_arm_data_viz[:, 2])
    tracked_arm_lines_viz[0].set_data_3d(current_pose_human_viz[[s_idx, e_idx], 0],
                                         current_pose_human_viz[[s_idx, e_idx], 1],
                                         current_pose_human_viz[[s_idx, e_idx], 2])
    tracked_arm_lines_viz[1].set_data_3d(current_pose_human_viz[[e_idx, w_idx], 0],
                                         current_pose_human_viz[[e_idx, w_idx], 1],
                                         current_pose_human_viz[[e_idx, w_idx], 2])

    # Update uArm target point and trail (visualized in pinned human frame)
    # This visualization logic for uArm target needs careful calibration of origins and scales
    if animation_controller.latest_uarm_target_abs and \
            animation_controller.initial_human_shoulder_pos_raw is not None and \
            animation_controller.initial_human_root_pos_raw is not None:
        # This is where the uArm's (0,0,0) + UARM_SHOULDER_ORIGIN_OFFSET would be if plotted in the pinned human viz space
        conceptual_mapping_origin_in_pinned_viz = \
            (animation_controller.initial_human_shoulder_pos_raw - animation_controller.initial_human_root_pos_raw) + \
            (UARM_SHOULDER_ORIGIN_OFFSET / HUMAN_TO_UARM_SCALE_FACTOR)

        uarm_target_mm = animation_controller.latest_uarm_target_abs[0]  # Current target in uArm's mm space

        # Vector from uArm's defined mapping origin (UARM_SHOULDER_ORIGIN_OFFSET in mm) to current target (in mm)
        vec_uarm_origin_to_target_mm = uarm_target_mm - UARM_SHOULDER_ORIGIN_OFFSET

        # Scale this vector to "human units" (e.g., meters if human data was meters)
        vec_uarm_origin_to_target_human_units = vec_uarm_origin_to_target_mm / HUMAN_TO_UARM_SCALE_FACTOR

        # The final plot point for uArm target in the pinned visualization
        plot_point_uarm_viz = conceptual_mapping_origin_in_pinned_viz + vec_uarm_origin_to_target_human_units
        uarm_target_point_viz.set_data_3d([plot_point_uarm_viz[0]], [plot_point_uarm_viz[1]], [plot_point_uarm_viz[2]])

    if animation_controller.uarm_target_trail and \
            animation_controller.initial_human_shoulder_pos_raw is not None and \
            animation_controller.initial_human_root_pos_raw is not None:

        conceptual_mapping_origin_in_pinned_viz = \
            (animation_controller.initial_human_shoulder_pos_raw - animation_controller.initial_human_root_pos_raw) + \
            (UARM_SHOULDER_ORIGIN_OFFSET / HUMAN_TO_UARM_SCALE_FACTOR)

        trail_data_mm = np.array(list(animation_controller.uarm_target_trail))  # (N, 3) in uArm's mm space
        if trail_data_mm.size > 0:
            # For each point in trail_data_mm, calculate its position in pinned human viz space
            trail_plot_points_viz = []
            for point_mm in trail_data_mm:
                vec_mm = point_mm - UARM_SHOULDER_ORIGIN_OFFSET
                vec_human_units = vec_mm / HUMAN_TO_UARM_SCALE_FACTOR
                plot_point = conceptual_mapping_origin_in_pinned_viz + vec_human_units
                trail_plot_points_viz.append(plot_point)

            if trail_plot_points_viz:
                trail_plot_points_viz_np = np.array(trail_plot_points_viz)
                uarm_trail_line_viz.set_data_3d(trail_plot_points_viz_np[:, 0], trail_plot_points_viz_np[:, 1],
                                                trail_plot_points_viz_np[:, 2])

    # Update human wrist trail (already in pinned human viz coordinates)
    if animation_controller.human_wrist_trail_for_viz:
        wrist_trail_data_viz = np.array(list(animation_controller.human_wrist_trail_for_viz))
        if wrist_trail_data_viz.size > 0:
            human_wrist_trail_line_viz.set_data_3d(wrist_trail_data_viz[:, 0], wrist_trail_data_viz[:, 1],
                                                   wrist_trail_data_viz[:, 2])

    ax_anim.set_title(
        f"Frame: {current_data_frame_idx} / {animation_controller.human_pose_sequence_for_viz.shape[0] - 1}")

    return ([human_full_scatter_viz] + human_full_lines_viz +
            [tracked_arm_scatter_viz] + tracked_arm_lines_viz +
            [uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz])


def main_with_viz():
    global fig_anim, ax_anim, animation_controller  # Use renamed global vars
    print("--- Human Arm to uArm Mimicry with Pinned Visualization ---")

    try:
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root_dir = os.getcwd()  # Fallback if __file__ is not defined
        print(f"Warning: __file__ not defined. Assuming project root is current working directory: {project_root_dir}")

    npz_full_path = os.path.join(project_root_dir, NPZ_FILE_RELATIVE_PATH)
    if not os.path.exists(npz_full_path):
        print(f"Error: NPZ data file not found at '{npz_full_path}'. Please check NPZ_FILE_RELATIVE_PATH.")
        return

    animation_controller = UArmMimicControllerWithViz(port=UARM_SERIAL_PORT)

    try:
        if not animation_controller.load_human_data(npz_full_path, SKELETON_TYPE, TRACKED_ARM):
            print("Failed to load human data. Exiting.")
            return
        if not animation_controller.connect_uarm():
            print("Failed to connect to uArm. Exiting.")
            return

        print("\nArm and data ready. Preparing visualization and starting mimicry...")

        fig_anim = plt.figure(figsize=(15, 10))  # Adjusted figure size
        ax_anim = fig_anim.add_subplot(111, projection='3d')

        if not animation_controller.start_mimicry():
            print("Failed to start uArm control thread. Exiting.")
            return

        # Number of frames for animation should be based on the loaded sequence length
        if animation_controller.raw_human_pose_sequence is None:  # Should be loaded by now
            print("Error: Raw human pose sequence is None before starting animation.")
            return
        num_anim_frames = animation_controller.raw_human_pose_sequence.shape[0]
        anim_interval = int(1000 / TARGET_FPS)

        # Create animation
        # blit=False is generally more robust for complex 3D plots, though potentially slower.
        ani = FuncAnimation(fig_anim, update_animation_frame, frames=num_anim_frames,
                            init_func=init_animation_artists, blit=False,
                            interval=anim_interval, repeat=False)

        plt.tight_layout()
        plt.show()  # This call blocks until the Matplotlib window is closed.

        print("Plot window closed by user or animation finished.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C in console).")
    except Exception as e:
        print(f"An unexpected error occurred in main_with_viz: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if animation_controller:
            print("Initiating cleanup...")
            animation_controller.cleanup()  # This will also attempt to stop the thread
        print("\n--- Program Finished ---")


if __name__ == '__main__':
    main_with_viz()
