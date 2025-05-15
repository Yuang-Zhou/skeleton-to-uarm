# config_arm_mapper.py

import os
import numpy as np

# --- Constants and Configuration ---
# Path to the NPZ file containing pose data
# Relative to the main script's directory (e.g., main_mapper.py)
NPZ_FILE_RELATIVE_PATH = os.path.join("data", "00", "joints_drc_smooth.npz")

# Skeleton definition type
SKELETON_TYPE = 'smpl_24'
# Which arm to track ('right' or 'left')
TRACKED_ARM = 'right'

# uArm serial port configuration
# macOS example: '/dev/cu.usbmodemXXXXX'
# Windows example: 'COMX' (e.g., 'COM3')
# Linux example: '/dev/ttyACMX' or '/dev/ttyUSBX'
# Set to None for auto-detection by the SDK
UARM_SERIAL_PORT = '/dev/cu.usbmodem144301' # !!! USER: MODIFY THIS !!!

# uArm operational speeds
INITIAL_UARM_RESET_SPEED = 3000  # mm/min, for resetting the arm
MOVEMENT_SPEED_MMPM = 7000       # mm/min, for general set_position commands
WRIST_SPEED_DEGPM = 1200         # deg/min, for set_wrist commands

# Mapping parameters from human motion to uArm space
# Target effective length of the uArm's arm when the human arm is conceptually fully extended
UARM_TARGET_MAPPED_ARM_LENGTH_MM = 400.0 # (mm)
# Minimum human arm length (sum of upper arm and forearm, in meters) for stable dynamic scaling.
# If the calculated human arm length is below this, FALLBACK_SCALE_FACTOR_M_TO_MM is used.
MIN_HUMAN_ARM_LENGTH_FOR_SCALING_M = 0.1 # (meters)
# Fallback scaling factor if human arm length is too small or zero. (human meters to uArm mm)
FALLBACK_SCALE_FACTOR_M_TO_MM = 180.0

# Offset for the conceptual human shoulder in the uArm's NATIVE coordinate system (in mm).
# This point acts as the origin for mapped human arm movements relative to the uArm.
# The uArm's NATIVE coordinate system is typically: X (forward/reach), Y (left/right), Z (up/down)
# If side_mounted_native_x_cw90, this changes: X (robot physical up), Y (robot physical left/right), Z (robot physical reach)
# This UARM_SHOULDER_ORIGIN_OFFSET is in the uArm's NATIVE SDK CS.
UARM_SHOULDER_ORIGIN_OFFSET = np.array([0.0, 0.0, 20.0]) # (mm) Example: X=100, Y=0, Z=150 in SDK CS

# Target frames per second for sending commands to uArm and for animation updates
TARGET_FPS = 25

# Trail lengths for visualization
UARM_TARGET_TRAIL_LENGTH = 60  # Number of past uArm target points to display
HUMAN_WRIST_TRAIL_LENGTH = 60  # Number of past human wrist points to display

# Index of the root joint in the human pose data (e.g., Pelvis for SMPL)
HUMAN_ROOT_JOINT_IDX = 0

# Defines how the uArm is physically placed and how human coordinates map to it.
# Options:
#   'upright': Standard uArm orientation. Human CS (X-front, Y-left, Z-up) maps to
#              uArm SDK CS (X-front, Y-up, Z-right) with appropriate axis swaps and inversions.
#   'side_mounted_native_x_cw90': uArm is on its side. Human CS maps to uArm SDK CS
#                                 considering this rotated placement.
#                                 Mapping for 'right' arm:
#                                 Human Front (+H_x) -> uArm SDK X (Reach direction of the physical robot)
#                                 Human Up    (+H_z) -> uArm SDK Y (Vertical movement of the physical robot arm)
#                                 Human Left  (+H_y) -> uArm SDK -Z (Side-to-side, to the physical robot's left)
UARM_PLACEMENT_MODE = 'side_mounted_native_x_cw90'

# --- Skeleton Definitions ---
# SMPL 24-joint definition
SMPL24_JOINT_MAPPING = {
    'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3, 'L_Knee': 4, 'R_Knee': 5,
    'Spine2': 6, 'L_Ankle': 7, 'R_Ankle': 8, 'Spine3': 9, 'L_Foot': 10, 'R_Foot': 11,
    'Neck': 12, 'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
    'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19,
    'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23
}

# Key joints for arm tracking within the SMPL24 skeleton
SMPL24_ARM_KEY_JOINTS = {
    'right': {
        'shoulder': SMPL24_JOINT_MAPPING['R_Shoulder'],
        'elbow': SMPL24_JOINT_MAPPING['R_Elbow'],
        'wrist': SMPL24_JOINT_MAPPING['R_Wrist']
    },
    'left': {
        'shoulder': SMPL24_JOINT_MAPPING['L_Shoulder'],
        'elbow': SMPL24_JOINT_MAPPING['L_Elbow'],
        'wrist': SMPL24_JOINT_MAPPING['L_Wrist']
    }
}

# Visualization coordinate system transformation matrix placeholder.
# This will be calculated dynamically in visualization based on UARM_PLACEMENT_MODE.
# R_NATIVE_TO_VIZ = np.eye(3) # Placeholder, calculated in get_rotation_matrix_native_to_viz