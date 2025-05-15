import os
import sys
import time
import numpy as np
# import math # math was imported but not explicitly used
import threading
import collections # Kept for collections.deque for latest_uarm_target_abs_mm
from typing import Optional, Tuple, List

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

UARM_SHOULDER_ORIGIN_OFFSET = np.array([100.0, 0.0, 150.0]) # In SDK CS: X-Front, Y-Up, Z-Right

TARGET_FPS = 25  # Target FPS for uArm command sending

HUMAN_ROOT_JOINT_IDX = 0  # Typically Pelvis (index 0) for SMPL-like skeletons
UARM_PLACEMENT_MODE = 'side_mounted_native_x_cw90'  # Options: 'upright', 'side_mounted_native_x_cw90'

# --- Helper: Path Setup for custom modules and SDK ---
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

    from kinematics.skeleton_utils import get_skeleton_parents
    from uarm.wrapper import SwiftAPI

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

class UArmMimicController:
    """
    Controls the uArm to mimic human arm movements loaded from pose data.
    """
    def __init__(self, port: Optional[str] = UARM_SERIAL_PORT):
        self.swift: Optional[SwiftAPI] = None
        self.port: Optional[str] = port
        self.is_connected_and_ready: bool = False
        self.key_joint_indices: Optional[dict[str, int]] = None

        self.raw_human_pose_sequence: Optional[np.ndarray] = None
        # self.human_pose_sequence_for_viz: Optional[np.ndarray] = None # Removed for no-viz
        self.initial_human_root_pos_raw: Optional[np.ndarray] = None
        self.initial_human_shoulder_pos_raw: Optional[np.ndarray] = None

        self.uarm_control_thread: Optional[threading.Thread] = None
        self.stop_thread_flag = threading.Event()
        self.current_human_frame_idx: int = 0
        self.latest_uarm_target_abs_mm = collections.deque(maxlen=1) # Still useful for internal state/logging
        # self.uarm_target_trail_mm = collections.deque(maxlen=UARM_TARGET_TRAIL_LENGTH) # Removed for no-viz
        # self.human_wrist_trail_for_viz = collections.deque(maxlen=HUMAN_WRIST_TRAIL_LENGTH) # Removed for no-viz
        self.human_skeleton_parents: Optional[np.ndarray] = None
        self.last_calculated_dynamic_scale_factor: float = FALLBACK_SCALE_FACTOR_M_TO_MM

    def connect_uarm(self) -> bool:
        """Attempts to connect to the uArm and initialize it."""
        print(f"Attempting to connect to uArm on port: {self.port if self.port else 'Auto-detect'}...")
        try:
            self.swift = SwiftAPI(port=self.port,
                                  臂吸펌웨어_업그레이드=False)
            time.sleep(2.0)

            if not self.swift.connected:
                print("Error: SwiftAPI serial connection failed.")
                self.swift = None
                return False

            _ = self.swift.waiting_ready(timeout=20)
            power_status = self.swift.get_power_status(wait=True, timeout=5)
            if power_status:
                print("uArm power status is ON.")
            else:
                print("Warning: uArm power status is OFF or could not be determined. Please ensure it's powered on.")

            device_info = self.swift.get_device_info(timeout=10)
            if device_info:
                print(f"Device Info: {device_info}")
            else:
                print("Warning: Failed to get device info.")

            current_mode = self.swift.get_mode(wait=True, timeout=10)
            if current_mode != 0 and current_mode is not None:
                print(f"uArm is in mode {current_mode}, setting to mode 0 (Normal).")
                self.swift.set_mode(0, wait=True, timeout=10)

            print("Resetting uArm on connect (standard SDK reset)...")
            self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=25)
            pos_after_reset = self.swift.get_position(wait=True, timeout=10)
            print(f"uArm position after SDK reset: {pos_after_reset}")

            if isinstance(pos_after_reset, list) and \
                    (190 < pos_after_reset[0] < 210 and \
                     -10 < pos_after_reset[1] < 10 and \
                     140 < pos_after_reset[2] < 160):
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
            if self.swift: self.swift.disconnect()
            self.swift = None
            return False

    def load_human_data(self, npz_file_path: str, skeleton_def: str, arm_choice: str) -> bool:
        """Loads human pose data and prepares it for mimicry."""
        self.raw_human_pose_sequence = load_pose_sequence_from_npz(npz_file_path)
        if self.raw_human_pose_sequence is None or self.raw_human_pose_sequence.shape[0] == 0:
            print(f"Error: Failed to load human pose data from '{npz_file_path}' or sequence is empty.")
            return False

        self.key_joint_indices = get_arm_joint_indices(skeleton_type=skeleton_def, arm_to_track=arm_choice)
        if not self.key_joint_indices:
            print(f"Error: Could not get key joint indices for skeleton '{skeleton_def}' and arm '{arm_choice}'.")
            return False

        self.human_skeleton_parents = get_skeleton_parents(skeleton_def) # Still needed if you ever re-add viz or other skeleton logic
        if self.human_skeleton_parents is None:
            print(f"Error: Could not get skeleton parents for '{skeleton_def}'.")
            # return False # Not strictly critical if no visualization using it

        num_joints_in_data = self.raw_human_pose_sequence.shape[1]
        max_expected_joint_idx = max(self.key_joint_indices.values())
        if not (0 <= HUMAN_ROOT_JOINT_IDX < num_joints_in_data and max_expected_joint_idx < num_joints_in_data):
            print(f"Error: Joint indices (Root: {HUMAN_ROOT_JOINT_IDX}, MaxArm: {max_expected_joint_idx}) "
                  f"are out of bounds for the loaded skeleton with {num_joints_in_data} joints.")
            return False

        self.initial_human_root_pos_raw = self.raw_human_pose_sequence[0, HUMAN_ROOT_JOINT_IDX, :].copy()
        self.initial_human_shoulder_pos_raw = self.raw_human_pose_sequence[0, self.key_joint_indices['shoulder'], :].copy()

        print(f"Raw initial human shoulder (frame 0, Index {self.key_joint_indices['shoulder']}): {self.initial_human_shoulder_pos_raw}")
        print(f"Raw initial human root (frame 0, Index {HUMAN_ROOT_JOINT_IDX}): {self.initial_human_root_pos_raw}")

        # self.human_pose_sequence_for_viz = self.raw_human_pose_sequence - self.initial_human_root_pos_raw[np.newaxis, np.newaxis, :] # Removed for no-viz

        print(f"Human pose data loaded: {self.raw_human_pose_sequence.shape[0]} frames.")
        return True

    def move_uarm_to_initial_human_pose(self, initial_frame_idx: int = 0,
                                        speed_mm_per_min: Optional[int] = None) -> bool:
        if not self.is_connected_and_ready or not self.swift:
            print("Error: uArm not connected/ready for initial pose movement.")
            return False

        print(f"\nMoving uArm to correspond with human pose at frame {initial_frame_idx}...")
        targets = self.calculate_uarm_target_for_frame(initial_frame_idx, is_initial_move=True)
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
            print(f"  uArm pos after initial move attempt: {final_pos}. Pos cmd success: {pos_result}, Wrist cmd success: {wrist_result}")

            if isinstance(final_pos, list) and np.allclose(final_pos, uarm_target_abs_mm, atol=5.0):
                print("  uArm successfully moved to initial human pose.")
                if len(self.latest_uarm_target_abs_mm) > 0: self.latest_uarm_target_abs_mm.popleft()
                self.latest_uarm_target_abs_mm.append(uarm_target_abs_mm.copy())
                # self.uarm_target_trail_mm.append(uarm_target_abs_mm.copy()) # Removed for no-viz
                return True
            else:
                print("  Warning: uArm final position doesn't match target for initial pose.");
                return False
        else:
            print("  Error: Could not calculate uArm target for initial human pose.");
            return False

    def calculate_uarm_target_for_frame(self, frame_idx: int, is_initial_move: bool = False) -> Optional[
        Tuple[np.ndarray, float]]:
        """
        Calculates the uArm target position and wrist angle for a given human pose frame.
        Human CS (Left-Handed): H_x: Front(+)/Back(-), H_y: Left(+)/Right(-), H_z: Up(+)/Down(-)
        uArm SDK CS (Right-Handed): SDK_X: Front(+), SDK_Y: Up(+), SDK_Z: Right(+)
        """
        if self.raw_human_pose_sequence is None or self.key_joint_indices is None: return None
        if not (0 <= frame_idx < self.raw_human_pose_sequence.shape[0]): return None

        shoulder_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['shoulder']]
        elbow_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['elbow']]
        wrist_pos_raw = self.raw_human_pose_sequence[frame_idx, self.key_joint_indices['wrist']]

        upper_arm_vec_raw = elbow_pos_raw - shoulder_pos_raw
        forearm_vec_raw = wrist_pos_raw - elbow_pos_raw
        human_upper_arm_length_m = np.linalg.norm(upper_arm_vec_raw)
        human_forearm_length_m = np.linalg.norm(forearm_vec_raw)
        total_human_arm_length_m = human_upper_arm_length_m + human_forearm_length_m

        wrist_vec_in_raw_human_shoulder_frame = wrist_pos_raw - shoulder_pos_raw
        human_shoulder_to_wrist_vec_length_m = np.linalg.norm(wrist_vec_in_raw_human_shoulder_frame)

        current_dynamic_scale_factor: float
        if total_human_arm_length_m < MIN_HUMAN_ARM_LENGTH_FOR_SCALING_M or abs(total_human_arm_length_m) < 1e-6:
            current_dynamic_scale_factor = FALLBACK_SCALE_FACTOR_M_TO_MM
        else:
            current_dynamic_scale_factor = UARM_TARGET_MAPPED_ARM_LENGTH_MM / total_human_arm_length_m
        self.last_calculated_dynamic_scale_factor = current_dynamic_scale_factor

        scaled_human_s_to_w_vec_mm = wrist_vec_in_raw_human_shoulder_frame * current_dynamic_scale_factor
        H_x_sw, H_y_sw, H_z_sw = scaled_human_s_to_w_vec_mm[0], scaled_human_s_to_w_vec_mm[1], scaled_human_s_to_w_vec_mm[2]

        sdk_x_rel, sdk_y_rel, sdk_z_rel = 0.0, 0.0, 0.0
        if UARM_PLACEMENT_MODE == 'upright':
            if TRACKED_ARM == 'right':
                sdk_x_rel, sdk_y_rel, sdk_z_rel = H_x_sw, H_z_sw, -H_y_sw
            elif TRACKED_ARM == 'left':
                sdk_x_rel, sdk_y_rel, sdk_z_rel = H_x_sw, H_z_sw, -H_y_sw # Assuming mirroring
            else:
                print(f"Error: Invalid TRACKED_ARM '{TRACKED_ARM}'."); return None
        elif UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
            # This mapping depends on how the uArm SDK axes (X-Front, Y-Up, Z-Right)
            # are oriented in the world when the arm is side-mounted.
            # Assuming the previous interpretation where SDK's Y becomes world vertical,
            # SDK's X becomes world reach, and SDK's Z becomes world side-to-side.
            if TRACKED_ARM == 'right':
                 # Human Front (H_x) -> Robot Reach (SDK X)
                 # Human Up    (H_z) -> Robot Vertical (SDK Y)
                 # Human Left  (H_y) -> Robot Left (SDK -Z, if Z is robot's right)
                sdk_x_rel = H_x_sw
                sdk_y_rel = H_z_sw
                sdk_z_rel = -H_y_sw
            elif TRACKED_ARM == 'left':
                sdk_x_rel = H_x_sw
                sdk_y_rel = H_z_sw
                sdk_z_rel = -H_y_sw # Assuming mirroring
            else:
                print(f"Error: Invalid TRACKED_ARM '{TRACKED_ARM}'."); return None
        else:
            print(f"Error: Unknown UARM_PLACEMENT_MODE '{UARM_PLACEMENT_MODE}'."); return None

        uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec = np.array([sdk_x_rel, sdk_y_rel, sdk_z_rel])

        print_debug_this_frame = is_initial_move or (frame_idx % (TARGET_FPS * 5) == 0) # Print less often without viz
        if print_debug_this_frame:
            print(f"\n--- Frame {frame_idx} Debug Info {'(Initial Move)' if is_initial_move else ''} ---")
            print(f"  Human CS (Input - Left-Handed: X-Front, Y-Left, Z-Up)")
            print(f"    Shoulder-to-Wrist Vec (Human CS, m): ({wrist_vec_in_raw_human_shoulder_frame[0]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[1]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[2]:.3f})")
            print(f"    Shoulder-to-Wrist Length (m): {human_shoulder_to_wrist_vec_length_m:.4f}")
            print(f"  Scaling & Mapping")
            print(f"    Dynamic Scale Factor (mm/m): {current_dynamic_scale_factor:.2f}")
            print(f"    Scaled Human S-to-W Vec (Human CS, mm): (H_x_sw:{H_x_sw:.1f}, H_y_sw:{H_y_sw:.1f}, H_z_sw:{H_z_sw:.1f})")
            print(f"  uArm SDK CS (Output - Right-Handed: X-Front, Y-Up, Z-Right)")
            print(f"    Conceptual Shoulder Offset (SDK CS, mm): {UARM_SHOULDER_ORIGIN_OFFSET}")
            print(f"    Effector Relative to Conceptual Shoulder (SDK CS, mm): (sdk_x_rel:{sdk_x_rel:.1f}, sdk_y_rel:{sdk_y_rel:.1f}, sdk_z_rel:{sdk_z_rel:.1f})")

        uarm_target_abs_mm_before_clip = UARM_SHOULDER_ORIGIN_OFFSET + uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec
        uarm_target_abs_mm = uarm_target_abs_mm_before_clip.copy()

        min_x, max_x = 50, 320
        min_y, max_y = -180, 220 # uArm's Y (up/down)
        min_z, max_z = -180, 180 # uArm's Z (left/right)

        uarm_target_abs_mm[0] = np.clip(uarm_target_abs_mm[0], min_x, max_x)
        uarm_target_abs_mm[1] = np.clip(uarm_target_abs_mm[1], min_y, max_y)
        uarm_target_abs_mm[2] = np.clip(uarm_target_abs_mm[2], min_z, max_z)

        if print_debug_this_frame:
            if not np.array_equal(uarm_target_abs_mm, uarm_target_abs_mm_before_clip):
                print(f"  Clipping Applied:")
                print(f"    Target BEFORE clip (SDK CS, mm): ({uarm_target_abs_mm_before_clip[0]:.1f}, {uarm_target_abs_mm_before_clip[1]:.1f}, {uarm_target_abs_mm_before_clip[2]:.1f})")
                print(f"    Target AFTER clip  (SDK CS, mm): ({uarm_target_abs_mm[0]:.1f}, {uarm_target_abs_mm[1]:.1f}, {uarm_target_abs_mm[2]:.1f})")
            else:
                print(f"    Target (SDK CS, mm): ({uarm_target_abs_mm[0]:.1f}, {uarm_target_abs_mm[1]:.1f}, {uarm_target_abs_mm[2]:.1f}) (No clipping needed)")

        # if self.human_pose_sequence_for_viz is not None and 0 <= frame_idx < self.human_pose_sequence_for_viz.shape[0]: # Removed for no-viz
        #     current_human_wrist_pos_viz = self.human_pose_sequence_for_viz[frame_idx, self.key_joint_indices['wrist']]
        #     self.human_wrist_trail_for_viz.append(current_human_wrist_pos_viz.copy()) # Removed for no-viz

        wrist_angle_uarm_deg = 90.0
        return uarm_target_abs_mm, wrist_angle_uarm_deg

    def _uarm_control_loop(self, start_frame_offset: int = 0):
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
            self.current_human_frame_idx = frame_idx # Still useful for logging/status

            targets = self.calculate_uarm_target_for_frame(frame_idx)

            if targets:
                uarm_target_abs_mm, wrist_angle_uarm_deg = targets

                if len(self.latest_uarm_target_abs_mm) > 0: self.latest_uarm_target_abs_mm.popleft()
                self.latest_uarm_target_abs_mm.append(uarm_target_abs_mm.copy())
                # self.uarm_target_trail_mm.append(uarm_target_abs_mm.copy()) # Removed for no-viz

                self.swift.set_position(x=uarm_target_abs_mm[0], y=uarm_target_abs_mm[1], z=uarm_target_abs_mm[2],
                                        speed=MOVEMENT_SPEED_MMPM, wait=False)
                self.swift.set_wrist(angle=wrist_angle_uarm_deg, speed=WRIST_SPEED_DEGPM, wait=False)

            if frame_idx % (TARGET_FPS * 2) == 0:
                log_msg_ctrl = f"[uArm Thread] Sent cmd for Frame {frame_idx}/{num_frames}."
                if targets: log_msg_ctrl += f" Target uArm (SDK mm): ({targets[0][0]:.0f}, {targets[0][1]:.0f}, {targets[0][2]:.0f})"
                print(log_msg_ctrl)

            elapsed_time = time.perf_counter() - loop_start_time
            time_to_wait = (1.0 / TARGET_FPS) - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        print("[uArm Thread] uArm control loop finished.")
        if self.swift and self.swift.connected:
            self.swift.flush_cmd(timeout=5, wait_stop=True)
        self.stop_thread_flag.set() # Signal main thread if loop finishes naturally

    def start_mimicry(self, start_from_frame_one: bool = False):
        if not self.is_connected_and_ready or self.raw_human_pose_sequence is None:
            print("Error: Cannot start mimicry. uArm not ready or human data not loaded.")
            return False
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Info: Mimicry thread already running.");
            return True

        self.stop_thread_flag.clear()
        loop_start_offset = 1 if start_from_frame_one else 0

        self.uarm_control_thread = threading.Thread(target=self._uarm_control_loop, args=(loop_start_offset,),
                                                    daemon=True) # Daemon thread will exit if main exits
        self.uarm_control_thread.start()
        print(f"uArm control thread started (loop will begin at frame index {loop_start_offset}).")
        return True

    def stop_mimicry(self):
        print("Attempting to stop uArm mimicry...")
        self.stop_thread_flag.set()
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            self.uarm_control_thread.join(timeout=7)
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Warning: uArm control thread did not join in time.")
        self.uarm_control_thread = None
        print("Mimicry stop sequence initiated.")

    def cleanup(self):
        self.stop_mimicry()
        if self.swift and self.swift.connected:
            print("\nResetting arm before disconnecting...")
            try:
                self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=15)
                self.swift.disconnect()
            except Exception as e:
                print(f"Error during uArm cleanup (reset/disconnect): {e}")
        self.is_connected_and_ready = False
        self.swift = None
        print("Cleanup complete.")


def main():
    print(f"--- Human Arm to uArm Mimicry (No Visualization, Placement: {UARM_PLACEMENT_MODE}, Scaling: Dynamic) ---")
    controller: Optional[UArmMimicController] = None # Type hint for clarity
    try:
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root_dir = os.getcwd()
    npz_full_path = os.path.join(project_root_dir, NPZ_FILE_RELATIVE_PATH)

    if not os.path.exists(npz_full_path):
        print(f"Error: NPZ data file not found: '{npz_full_path}'")
        return

    controller = UArmMimicController(port=UARM_SERIAL_PORT)
    try:
        if not controller.load_human_data(npz_full_path, SKELETON_TYPE, TRACKED_ARM):
            print("Failed to load human data. Exiting.")
            return

        uarm_successfully_connected = controller.connect_uarm()
        if not uarm_successfully_connected:
            print("CRITICAL: uArm connection failed. Cannot proceed with mimicry.")
            controller.cleanup() # Ensure potential partial connections are closed
            return # Exit if uArm cannot be controlled

        # Pre-calculate and move to initial pose
        if controller.raw_human_pose_sequence is not None and \
           controller.raw_human_pose_sequence.shape[0] > 0:
            # The calculate_uarm_target_for_frame for frame 0 is called within move_uarm_to_initial_human_pose
            if not controller.move_uarm_to_initial_human_pose(initial_frame_idx=0):
                print("Warning: Failed to accurately move uArm to initial human pose. Continuing...")
            else:
                print("uArm successfully positioned to initial human pose (frame 0).")
            controller.current_human_frame_idx = 0
            time.sleep(1) # Pause after initial uArm positioning
        else:
            print("Error: No human data loaded; cannot run uArm control.")
            if controller: controller.cleanup(); return

        print("\nStarting uArm control thread for mimicry...")
        if controller.start_mimicry(start_from_frame_one=True): # Start uArm control from frame 1
            print("Mimicry started. Press Ctrl+C to stop.")
            # Keep main thread alive while the daemon control thread runs
            while controller.uarm_control_thread and controller.uarm_control_thread.is_alive() and not controller.stop_thread_flag.is_set():
                time.sleep(0.5) # Check periodically
            if controller.stop_thread_flag.is_set() and not (controller.uarm_control_thread and controller.uarm_control_thread.is_alive()):
                 print("Mimicry loop finished naturally.")
        else:
            print("Failed to start uArm control thread.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            print("Initiating cleanup sequence...")
            controller.cleanup()
        print("\n--- Program Finished ---")

if __name__ == '__main__':
    main()