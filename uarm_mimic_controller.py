# uarm_mimic_controller.py

import os
import sys
import time
import numpy as np
import threading
import collections
from typing import Optional, Tuple, List, Dict # Python 3.9+ can use dict

# Imports from other refactored modules
from config_arm_mapper import (
    UARM_SERIAL_PORT, INITIAL_UARM_RESET_SPEED, MOVEMENT_SPEED_MMPM, WRIST_SPEED_DEGPM,
    UARM_TARGET_MAPPED_ARM_LENGTH_MM, MIN_HUMAN_ARM_LENGTH_FOR_SCALING_M,
    FALLBACK_SCALE_FACTOR_M_TO_MM, UARM_SHOULDER_ORIGIN_OFFSET, TARGET_FPS,
    UARM_TARGET_TRAIL_LENGTH, HUMAN_WRIST_TRAIL_LENGTH, HUMAN_ROOT_JOINT_IDX,
    SKELETON_TYPE, TRACKED_ARM, UARM_PLACEMENT_MODE
)
from arm_mapping_utils import load_pose_sequence_from_npz, get_arm_joint_indices

# Ensure uArm SDK and kinematics utils are on path
# The path setup should ideally be done once by the main entry point,
# but including it here defensively or relying on arm_mapping_utils's setup.
try:
    # This relies on arm_mapping_utils.py already having run its path setup if it's imported first.
    # Or, if this file is run standalone for testing, it might need its own setup.
    # For a modular design, it's better if the main script ensures paths are set.
    from uarm.wrapper import SwiftAPI
    # Assuming 'src' is in sys.path due to arm_mapping_utils or main_mapper.py
    from kinematics.skeleton_utils import get_skeleton_parents
except ImportError as e:
    print(f"Error importing SDK/kinematics in uarm_mimic_controller.py: {e}")
    print("Ensure arm_mapping_utils.py ran its path setup or paths are set by the main script.")
    # Fallback path setup if run directly or arm_mapping_utils wasn't imported/run first
    _controller_script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root_controller = _controller_script_dir

    _src_path_controller = os.path.join(_project_root_controller, "src")
    if os.path.isdir(_src_path_controller) and _src_path_controller not in sys.path:
        sys.path.insert(0, _src_path_controller)

    _sdk_base_dir_controller = os.path.join(_project_root_controller, "uarm-python-sdk")
    _sdk_actual_path_controller = os.path.join(_sdk_base_dir_controller, "uArm-Python-SDK-2.0")
    if os.path.isdir(os.path.join(_sdk_actual_path_controller, "uarm")) and _sdk_actual_path_controller not in sys.path:
        sys.path.insert(0, _sdk_actual_path_controller)

    try:
        from uarm.wrapper import SwiftAPI
        from kinematics.skeleton_utils import get_skeleton_parents
    except ImportError:
        print("Fallback path setup in uarm_mimic_controller.py also failed. Critical dependency missing.")
        sys.exit(1)


class UArmMimicController: # Renamed from UArmMimicControllerWithViz for generality
    """
    Controls the uArm to mimic human arm movements loaded from pose data.
    Also prepares data for optional 3D visualization.
    """
    def __init__(self, port: Optional[str] = UARM_SERIAL_PORT):
        self.swift: Optional[SwiftAPI] = None
        self.port: Optional[str] = port
        self.is_connected_and_ready: bool = False
        self.key_joint_indices: Optional[Dict[str, int]] = None

        self.raw_human_pose_sequence: Optional[np.ndarray] = None
        self.human_pose_sequence_for_viz: Optional[np.ndarray] = None # Root-pinned version for visualization
        self.initial_human_root_pos_raw: Optional[np.ndarray] = None
        self.initial_human_shoulder_pos_raw: Optional[np.ndarray] = None # Shoulder of the tracked arm

        self.uarm_control_thread: Optional[threading.Thread] = None
        self.stop_thread_flag = threading.Event()
        self.current_human_frame_idx: int = 0
        self.latest_uarm_target_abs_mm = collections.deque(maxlen=1)
        self.uarm_target_trail_mm = collections.deque(maxlen=UARM_TARGET_TRAIL_LENGTH)
        self.human_wrist_trail_for_viz = collections.deque(maxlen=HUMAN_WRIST_TRAIL_LENGTH)
        self.human_skeleton_parents: Optional[np.ndarray] = None
        self.last_calculated_dynamic_scale_factor: float = FALLBACK_SCALE_FACTOR_M_TO_MM

    def connect_uarm(self) -> bool:
        """Attempts to connect to the uArm and initialize it."""
        print(f"Attempting to connect to uArm on port: {self.port if self.port else 'Auto-detect'}...")
        try:
            self.swift = SwiftAPI(port=self.port, enable_handle_thread=True) # Added enable_handle_thread
            time.sleep(2.0)

            if not self.swift.connected:
                print("Error: SwiftAPI serial connection failed.")
                self.swift = None
                return False

            # Call waiting_ready, but also check power_status as it can be more reliable for some firmware
            _ = self.swift.waiting_ready(timeout=20) # Firmware 3.2.0 might return None on success
            power_status = self.swift.get_power_status(wait=True, timeout=5)

            if power_status: # True if powered, False if not, None on error
                print("uArm power status is ON.")
            else: # This includes power_status being False or None
                print(f"Warning: uArm power status is {power_status}. Please ensure it's powered on. Problems may occur.")
                # For some SDK/firmware versions, subsequent commands might still work if power_status is initially None or False.
                # If minimal_test.py showed it worked, we can be more lenient here.

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
            # The SDK's reset may not return a useful string status for all firmware versions.
            # The physical reset and subsequent position check are more important.
            self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=25)
            time.sleep(1.0) # Allow time for physical reset
            pos_after_reset = self.swift.get_position(wait=True, timeout=10)
            print(f"uArm position after SDK reset: {pos_after_reset}")

            # Check if the arm is near its typical home position after reset
            if isinstance(pos_after_reset, list) and \
               (190 < pos_after_reset[0] < 210 and \
                -10 < pos_after_reset[1] < 10 and \
                140 < pos_after_reset[2] < 160): # uArm SDK: X-Front, Y-Up, Z-Right (for default orientation)
                                                 # Note: Your original code had Y and Z swapped in check
                print("Physical reset to home position confirmed by SDK reset.")
                self.is_connected_and_ready = True
                return True
            else:
                print(f"Warning: Physical reset not confirmed by SDK reset. Position is {pos_after_reset}, expected near [200,0,150].")
                # Optionally, disconnect if reset is not confirmed, or allow proceeding with caution
                # For safety, let's consider it unstable if reset not confirmed
                if self.swift: self.swift.disconnect()
                self.swift = None
                self.is_connected_and_ready = False
                return False
        except Exception as e:
            print(f"Error connecting to uArm: {e}")
            import traceback
            traceback.print_exc()
            if self.swift:
                try: self.swift.disconnect()
                except: pass
            self.swift = None
            self.is_connected_and_ready = False
            return False

    def load_human_data(self, npz_file_path: str, skeleton_def: str = SKELETON_TYPE, arm_choice: str = TRACKED_ARM) -> bool:
        """Loads human pose data and prepares it for mimicry and visualization."""
        self.raw_human_pose_sequence = load_pose_sequence_from_npz(npz_file_path)
        if self.raw_human_pose_sequence is None or self.raw_human_pose_sequence.shape[0] == 0:
            print(f"Error (controller): Failed to load human pose data from '{npz_file_path}' or sequence is empty.")
            return False

        self.key_joint_indices = get_arm_joint_indices(skeleton_type=skeleton_def, arm_to_track=arm_choice)
        if not self.key_joint_indices:
            print(f"Error (controller): Could not get key joint indices for skeleton '{skeleton_def}' and arm '{arm_choice}'.")
            return False

        self.human_skeleton_parents = get_skeleton_parents(skeleton_def)
        if self.human_skeleton_parents is None:
            print(f"Error (controller): Could not get skeleton parents for '{skeleton_def}'. Visualization might be affected.")
            # Depending on whether visualization is critical, you might return False or just warn.

        num_joints_in_data = self.raw_human_pose_sequence.shape[1]
        max_expected_joint_idx = max(self.key_joint_indices.values())
        if not (0 <= HUMAN_ROOT_JOINT_IDX < num_joints_in_data and max_expected_joint_idx < num_joints_in_data):
            print(f"Error (controller): Joint indices (Root: {HUMAN_ROOT_JOINT_IDX}, MaxArm: {max_expected_joint_idx}) "
                  f"are out of bounds for the loaded skeleton with {num_joints_in_data} joints.")
            return False

        self.initial_human_root_pos_raw = self.raw_human_pose_sequence[0, HUMAN_ROOT_JOINT_IDX, :].copy()
        self.initial_human_shoulder_pos_raw = self.raw_human_pose_sequence[0, self.key_joint_indices['shoulder'], :].copy()

        print(f"Raw initial human shoulder (frame 0, Index {self.key_joint_indices['shoulder']}): {self.initial_human_shoulder_pos_raw}")
        print(f"Raw initial human root (frame 0, Index {HUMAN_ROOT_JOINT_IDX}): {self.initial_human_root_pos_raw}")

        self.human_pose_sequence_for_viz = self.raw_human_pose_sequence - self.initial_human_root_pos_raw[np.newaxis, np.newaxis, :]

        print(f"Human pose data loaded: {self.raw_human_pose_sequence.shape[0]} frames.")
        print(f"Visualization data (human_pose_sequence_for_viz) created and is root-relative.")
        return True

    def move_uarm_to_initial_human_pose(self, initial_frame_idx: int = 0, speed_mm_per_min: Optional[int] = None) -> bool:
        """Calculates and moves uArm to correspond with the initial human pose frame."""
        if not self.is_connected_and_ready or not self.swift:
            print("Error (controller): uArm not connected/ready for initial pose movement.")
            return False

        print(f"\nMoving uArm to correspond with human pose at frame {initial_frame_idx}...")
        targets = self.calculate_uarm_target_for_frame(initial_frame_idx, is_initial_move=True)
        if targets:
            uarm_target_abs_mm, wrist_angle_uarm_deg = targets
            positioning_speed = speed_mm_per_min if speed_mm_per_min is not None else MOVEMENT_SPEED_MMPM

            print(f"  Targeting initial uArm pos (SDK CS, mm): {uarm_target_abs_mm}, Wrist: {wrist_angle_uarm_deg} deg")
            print(f"  Movement speed: {positioning_speed} mm/min")

            pos_result = self.swift.set_position(
                x=uarm_target_abs_mm[0], y=uarm_target_abs_mm[1], z=uarm_target_abs_mm[2],
                speed=positioning_speed, wait=True, timeout=20
            )
            wrist_result = self.swift.set_wrist(
                angle=wrist_angle_uarm_deg, speed=WRIST_SPEED_DEGPM, wait=True, timeout=10
            )
            final_pos = self.swift.get_position(wait=True, timeout=5)
            print(f"  uArm pos after initial move attempt: {final_pos}. Pos cmd status: {pos_result}, Wrist cmd status: {wrist_result}")

            if isinstance(final_pos, list) and np.allclose(final_pos, uarm_target_abs_mm, atol=5.0): # Check if close to target
                print("  uArm successfully moved to initial human pose.")
                if len(self.latest_uarm_target_abs_mm) > 0: self.latest_uarm_target_abs_mm.popleft()
                self.latest_uarm_target_abs_mm.append(uarm_target_abs_mm.copy())
                self.uarm_target_trail_mm.append(uarm_target_abs_mm.copy()) # Start trail for viz
                return True
            else:
                print("  Warning: uArm final position doesn't match target for initial pose or get_position failed.");
                return False
        else:
            print("  Error: Could not calculate uArm target for initial human pose.");
            return False

    def calculate_uarm_target_for_frame(self, frame_idx: int, is_initial_move: bool = False) -> Optional[Tuple[np.ndarray, float]]:
        """
        Calculates the uArm target position (in uArm SDK CS) and wrist angle for a given human pose frame.
        Human CS (Left-Handed, from data): H_x: Front(+)/Back(-), H_y: Left(+)/Right(-), H_z: Up(+)/Down(-)
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
                sdk_x_rel = H_x_sw    # Human Front (+H_x) -> uArm Front (+SDK_X)
                sdk_y_rel = H_z_sw    # Human Up    (+H_z) -> uArm Up   (+SDK_Y)
                sdk_z_rel = -H_y_sw   # Human Left  (+H_y) maps to uArm Left, which is -SDK_Z (as SDK_Z+ is Right)
            elif TRACKED_ARM == 'left':
                sdk_x_rel = H_x_sw
                sdk_y_rel = H_z_sw
                sdk_z_rel = -H_y_sw # Assuming mirrored setup
            else:
                print(f"Error: Invalid TRACKED_ARM '{TRACKED_ARM}'."); return None
        elif UARM_PLACEMENT_MODE == 'side_mounted_native_x_cw90':
            # Human CS: H_x (Front), H_y (Left), H_z (Up)
            # uArm SDK CS (Relative to its own base): SDK_X (Front/Reach), SDK_Y (Up), SDK_Z (Right)
            # Physical interpretation for side_mounted_native_x_cw90:
            #   Robot's physical "Up/Down" movement is along its SDK Y-axis.
            #   Robot's physical "Reach" movement is along its SDK X-axis.
            #   Robot's physical "Side-to-Side" movement is along its SDK Z-axis.
            # Desired Mapping (from human perspective to robot's physical actions):
            #   Human Z (Up/Down) should control Robot's physical Up/Down (SDK Y).
            #   Human X (Front/Back) should control Robot's physical Reach (SDK X).
            #   Human Y (Left/Right) should control Robot's physical Side-to-Side (SDK Z).
            if TRACKED_ARM == 'right':
                sdk_x_rel = H_x_sw    # Human Front (+H_x) -> Robot Reach (+SDK_X)
                sdk_y_rel = H_z_sw    # Human Up    (+H_z) -> Robot Vertical (+SDK_Y)
                sdk_z_rel = -H_y_sw   # Human Left  (+H_y) -> Robot to its Left (-SDK_Z)
            elif TRACKED_ARM == 'left': # Mirroring for left arm
                sdk_x_rel = H_x_sw
                sdk_y_rel = H_z_sw
                sdk_z_rel = -H_y_sw # If human left is +H_y, robot left is -SDK_Z
            else:
                print(f"Error: Invalid TRACKED_ARM '{TRACKED_ARM}'."); return None
        else:
            print(f"Error: Unknown UARM_PLACEMENT_MODE '{UARM_PLACEMENT_MODE}'."); return None

        uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec = np.array([sdk_x_rel, sdk_y_rel, sdk_z_rel])

        print_debug_this_frame = is_initial_move or (frame_idx % (TARGET_FPS * 5) == 0) # Print less often for non-initial moves
        if print_debug_this_frame:
            print(f"\n--- Frame {frame_idx} Mapping Debug {'(Initial Move)' if is_initial_move else ''} ---")
            print(f"  Human CS (Input - Left-Handed: X-Front, Y-Left, Z-Up)")
            print(f"    S-to-W Vec (Human CS, m): ({wrist_vec_in_raw_human_shoulder_frame[0]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[1]:.3f}, {wrist_vec_in_raw_human_shoulder_frame[2]:.3f}), Length(m): {human_shoulder_to_wrist_vec_length_m:.4f}")
            print(f"  Scaling & Mapping")
            print(f"    Dynamic Scale Factor (mm/m): {current_dynamic_scale_factor:.2f}, Total Human Arm Length (m): {total_human_arm_length_m:.3f}")
            print(f"    Scaled Human S-to-W Vec (Human CS orientation, mm): (H_x_sw:{H_x_sw:.1f}, H_y_sw:{H_y_sw:.1f}, H_z_sw:{H_z_sw:.1f})")
            print(f"  uArm SDK CS (Output - Right-Handed: X-Front, Y-Up, Z-Right)")
            print(f"    Conceptual Shoulder Offset (SDK CS, mm): {UARM_SHOULDER_ORIGIN_OFFSET}")
            print(f"    Effector Relative to Conceptual Shoulder (SDK CS, mm): (sdk_x_rel:{sdk_x_rel:.1f}, sdk_y_rel:{sdk_y_rel:.1f}, sdk_z_rel:{sdk_z_rel:.1f})")

        uarm_target_abs_mm_before_clip = UARM_SHOULDER_ORIGIN_OFFSET + uarm_effector_relative_to_conceptual_shoulder_sdk_mm_vec
        uarm_target_abs_mm = uarm_target_abs_mm_before_clip.copy()

        min_x, max_x = 50, 320    # SDK X (Front/Reach)
        min_y, max_y = -180, 220  # SDK Y (Up/Down for upright, Side-to-Side for side_mounted_native_x_cw90 if Y becomes horizontal)
                                  # For side_mounted_native_x_cw90 where SDK_Y is physical_up: range might be e.g. 0 to 200.
        min_z, max_z = -180, 180  # SDK Z (Right/Left for upright, Reach for side_mounted_native_x_cw90 if Z becomes horizontal)
                                  # For side_mounted_native_x_cw90 where SDK_Z is physical_side_to_side: range might be e.g. -100 to 100.

        # Adjust clipping based on actual workspace limits in SDK CS
        # These are generic limits, specific mountings might require different ones.
        uarm_target_abs_mm[0] = np.clip(uarm_target_abs_mm[0], min_x, max_x) # Clip SDK X
        uarm_target_abs_mm[1] = np.clip(uarm_target_abs_mm[1], min_y, max_y) # Clip SDK Y
        uarm_target_abs_mm[2] = np.clip(uarm_target_abs_mm[2], min_z, max_z) # Clip SDK Z

        if print_debug_this_frame:
            if not np.array_equal(uarm_target_abs_mm, uarm_target_abs_mm_before_clip):
                print(f"  Clipping Applied:")
                print(f"    Target BEFORE clip (SDK CS, mm): ({uarm_target_abs_mm_before_clip[0]:.1f}, {uarm_target_abs_mm_before_clip[1]:.1f}, {uarm_target_abs_mm_before_clip[2]:.1f})")
                print(f"    Target AFTER clip  (SDK CS, mm): ({uarm_target_abs_mm[0]:.1f}, {uarm_target_abs_mm[1]:.1f}, {uarm_target_abs_mm[2]:.1f})")
            else:
                print(f"    Target (SDK CS, mm): ({uarm_target_abs_mm[0]:.1f}, {uarm_target_abs_mm[1]:.1f}, {uarm_target_abs_mm[2]:.1f}) (No clipping needed)")

        if self.human_pose_sequence_for_viz is not None and 0 <= frame_idx < self.human_pose_sequence_for_viz.shape[0]:
            current_human_wrist_pos_viz = self.human_pose_sequence_for_viz[frame_idx, self.key_joint_indices['wrist']]
            self.human_wrist_trail_for_viz.append(current_human_wrist_pos_viz.copy())

        wrist_angle_uarm_deg = 90.0
        return uarm_target_abs_mm, wrist_angle_uarm_deg

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
            self.current_human_frame_idx = frame_idx

            targets = self.calculate_uarm_target_for_frame(frame_idx)

            if targets:
                uarm_target_abs_mm, wrist_angle_uarm_deg = targets

                if len(self.latest_uarm_target_abs_mm) > 0: self.latest_uarm_target_abs_mm.popleft()
                self.latest_uarm_target_abs_mm.append(uarm_target_abs_mm.copy())
                self.uarm_target_trail_mm.append(uarm_target_abs_mm.copy())

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
            self.swift.flush_cmd(timeout=7, wait_stop=True) # Increased timeout for flush
        self.stop_thread_flag.set()

    def start_mimicry(self, start_from_frame_one: bool = False):
        """Starts the uArm control thread."""
        if not self.is_connected_and_ready or self.raw_human_pose_sequence is None:
            print("Error (controller): Cannot start mimicry. uArm not ready or human data not loaded.")
            return False
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Info (controller): Mimicry thread already running.");
            return True

        self.stop_thread_flag.clear()
        loop_start_offset = 1 if start_from_frame_one else 0

        self.uarm_control_thread = threading.Thread(target=self._uarm_control_loop, args=(loop_start_offset,), daemon=True)
        self.uarm_control_thread.start()
        print(f"uArm control thread started (loop will begin at frame index {loop_start_offset}).")
        return True

    def stop_mimicry(self):
        """Stops the uArm control thread."""
        print("Attempting to stop uArm mimicry...")
        self.stop_thread_flag.set()
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            self.uarm_control_thread.join(timeout=7)
        if self.uarm_control_thread and self.uarm_control_thread.is_alive():
            print("Warning (controller): uArm control thread did not join in time.")
        self.uarm_control_thread = None
        print("Mimicry stop sequence initiated.")

    def cleanup(self):
        """Stops mimicry, resets uArm, and disconnects."""
        self.stop_mimicry()
        if self.swift and self.swift.connected:
            print("\nResetting arm before disconnecting...")
            try:
                # It might be safer to move to a known safe position before reset
                # self.swift.set_position(x=200, y=0, z=150, speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=10)
                self.swift.reset(speed=INITIAL_UARM_RESET_SPEED, wait=True, timeout=20) # Increased timeout
                final_pos = self.swift.get_position(wait=True, timeout=5)
                print(f"Arm reset. Final position: {final_pos}")
                self.swift.disconnect()
                print("Disconnected from uArm.")
            except Exception as e:
                print(f"Error during uArm cleanup (reset/disconnect): {e}")
        elif self.swift: # Swift object exists but not marked as connected
             print("Swift object exists but not connected, attempting disconnect.")
             try:
                 self.swift.disconnect()
             except: pass # Ignore errors
        self.is_connected_and_ready = False
        self.swift = None
        print("Cleanup complete.")