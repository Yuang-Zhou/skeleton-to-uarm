# arm_visualization.py

import matplotlib

matplotlib.use('TkAgg')  # Using TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional, List, TYPE_CHECKING

# Imports from other refactored modules
from config_arm_mapper import (
    TRACKED_ARM, UARM_PLACEMENT_MODE, TARGET_FPS,
    UARM_SHOULDER_ORIGIN_OFFSET  # Used for conceptual positioning
)
from arm_mapping_utils import get_rotation_matrix_native_to_viz

if TYPE_CHECKING:  # To avoid circular import issues for type hinting
    from uarm_mimic_controller import UArmMimicController

# --- Matplotlib Animation Global Variables ---
# These are module-level variables that will be accessed by init and update functions.
# Consider encapsulating them in a class if this module grows more complex.
fig_anim_viz: Optional[plt.Figure] = None
ax_anim_viz: Optional[Axes3D] = None
# Human skeleton artists
human_full_scatter_viz: Optional[plt.Artist] = None
human_full_lines_viz: List[plt.Line2D] = []
# Tracked arm artists (human)
tracked_arm_scatter_viz: Optional[plt.Artist] = None
tracked_arm_lines_viz: List[plt.Line2D] = []
# uArm visualization artists
uarm_base_viz: Optional[plt.Artist] = None
uarm_target_point_viz: Optional[plt.Line2D] = None
uarm_trail_line_viz: Optional[plt.Line2D] = None
# Human wrist trail artist
human_wrist_trail_line_viz: Optional[plt.Line2D] = None


def init_animation_artists(passed_controller: 'UArmMimicController'):
    """
    Initializes all artists for the Matplotlib animation.
    passed_controller: An instance of UArmMimicController to access data.
    """
    global human_full_scatter_viz, human_full_lines_viz, tracked_arm_scatter_viz, tracked_arm_lines_viz
    global uarm_base_viz, uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz
    global fig_anim_viz, ax_anim_viz  # Ensure these are the global figure/axes for this module

    if passed_controller is None or \
            passed_controller.human_pose_sequence_for_viz is None or \
            passed_controller.key_joint_indices is None or \
            passed_controller.initial_human_root_pos_raw is None or \
            passed_controller.initial_human_shoulder_pos_raw is None or \
            passed_controller.human_skeleton_parents is None:
        print("Warning (Viz Init): Controller or necessary data not fully ready.")
        return []

    ax_anim_viz.clear()
    viz_data = passed_controller.human_pose_sequence_for_viz

    margin = 0.3
    x_min, x_max = viz_data[..., 0].min() - margin, viz_data[..., 0].max() + margin
    y_min, y_max = viz_data[..., 1].min() - margin, viz_data[..., 1].max() + margin
    z_min, z_max = viz_data[..., 2].min() - margin, viz_data[..., 2].max() + margin
    ax_anim_viz.set_xlim(x_min, x_max);
    ax_anim_viz.set_ylim(y_min, y_max);
    ax_anim_viz.set_zlim(z_min, z_max)

    axis_ranges = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_r = axis_ranges[axis_ranges > 1e-6].max() if (axis_ranges > 1e-6).any() else 1.0
    mid_x, mid_y, mid_z = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5, (z_max + z_min) * 0.5
    ax_anim_viz.set_xlim(mid_x - max_r / 2, mid_x + max_r / 2)
    ax_anim_viz.set_ylim(mid_y - max_r / 2, mid_y + max_r / 2)
    ax_anim_viz.set_zlim(mid_z - max_r / 2, mid_z + max_r / 2)

    ax_anim_viz.set_xlabel("X (Pinned Human / Viz)");
    ax_anim_viz.set_ylabel("Y (Pinned Human / Viz)");
    ax_anim_viz.set_zlabel("Z (Pinned Human / Viz)")
    ax_anim_viz.set_title("Human (Pinned) & uArm Conceptualization");
    ax_anim_viz.view_init(elev=20., azim=-75)

    initial_pose_viz = passed_controller.human_pose_sequence_for_viz[0]
    human_full_scatter_viz = ax_anim_viz.scatter(
        initial_pose_viz[:, 0], initial_pose_viz[:, 1], initial_pose_viz[:, 2],
        s=25, c='blue', alpha=0.6, label='Full Human (Pinned)'
    )
    human_full_lines_viz.clear()  # Ensure list is empty before repopulating
    for i in range(initial_pose_viz.shape[0]):
        if passed_controller.human_skeleton_parents[i] != -1:
            p_idx = passed_controller.human_skeleton_parents[i]
            line, = ax_anim_viz.plot(initial_pose_viz[[i, p_idx], 0], initial_pose_viz[[i, p_idx], 1],
                                     initial_pose_viz[[i, p_idx], 2], 'k-', alpha=0.2, lw=1.5)
            human_full_lines_viz.append(line)

    s_idx = passed_controller.key_joint_indices['shoulder']
    e_idx = passed_controller.key_joint_indices['elbow']
    w_idx = passed_controller.key_joint_indices['wrist']
    tracked_arm_joint_indices = [s_idx, e_idx, w_idx]
    tracked_arm_viz_data = initial_pose_viz[tracked_arm_joint_indices]
    tracked_arm_scatter_viz = ax_anim_viz.scatter(
        tracked_arm_viz_data[:, 0], tracked_arm_viz_data[:, 1], tracked_arm_viz_data[:, 2],
        s=40, c='red', label=f'Tracked {TRACKED_ARM} Arm (Pinned)', zorder=5
    )
    tracked_arm_lines_viz.clear()  # Ensure list is empty
    line_shoulder_elbow, = ax_anim_viz.plot(initial_pose_viz[[s_idx, e_idx], 0], initial_pose_viz[[s_idx, e_idx], 1],
                                            initial_pose_viz[[s_idx, e_idx], 2], 'r-', lw=4, zorder=4)
    line_elbow_wrist, = ax_anim_viz.plot(initial_pose_viz[[e_idx, w_idx], 0], initial_pose_viz[[e_idx, w_idx], 1],
                                         initial_pose_viz[[e_idx, w_idx], 2], 'r-', lw=4, zorder=4)
    tracked_arm_lines_viz.extend([line_shoulder_elbow, line_elbow_wrist])

    current_viz_scale_factor_init = passed_controller.last_calculated_dynamic_scale_factor
    if abs(
        current_viz_scale_factor_init) < 1e-9: current_viz_scale_factor_init = passed_controller.FALLBACK_SCALE_FACTOR_M_TO_MM  # Use attribute from controller

    human_shoulder_in_pinned_viz = passed_controller.initial_human_shoulder_pos_raw - passed_controller.initial_human_root_pos_raw
    R_native_to_viz = get_rotation_matrix_native_to_viz()

    uarm_conceptual_shoulder_offset_in_native_cs = UARM_SHOULDER_ORIGIN_OFFSET  # This is in native SDK CS
    uarm_conceptual_shoulder_offset_in_viz_cs_mm = R_native_to_viz @ uarm_conceptual_shoulder_offset_in_native_cs
    uarm_conceptual_shoulder_offset_in_viz_cs_human_units = uarm_conceptual_shoulder_offset_in_viz_cs_mm / current_viz_scale_factor_init

    # The uArm's physical base (0,0,0 in its own SDK CS) is located such that its conceptual shoulder matches the human's.
    # So, P_human_shoulder_viz = P_uarm_base_viz + R_native_to_viz @ P_uarm_conceptual_shoulder_native
    # Therefore, P_uarm_base_viz = P_human_shoulder_viz - (R_native_to_viz @ P_uarm_conceptual_shoulder_native_mm) / scale_factor
    conceptual_uarm_physical_base_in_viz_init = human_shoulder_in_pinned_viz - uarm_conceptual_shoulder_offset_in_viz_cs_human_units

    uarm_base_viz = ax_anim_viz.scatter(
        [conceptual_uarm_physical_base_in_viz_init[0]], [conceptual_uarm_physical_base_in_viz_init[1]],
        [conceptual_uarm_physical_base_in_viz_init[2]],
        s=150, c='purple', marker='s', label='uArm Base (Conceptual)', zorder=9, edgecolors='k'
    )

    initial_uarm_effector_plot_point_viz = conceptual_uarm_physical_base_in_viz_init  # Default
    if passed_controller.latest_uarm_target_abs_mm:
        uarm_target_effector_native_mm = passed_controller.latest_uarm_target_abs_mm[0]
        # Effector position in VIZ CS, relative to uArm base in VIZ CS
        uarm_target_effector_in_viz_cs_mm = R_native_to_viz @ uarm_target_effector_native_mm
        uarm_target_effector_in_viz_cs_human_units = uarm_target_effector_in_viz_cs_mm / current_viz_scale_factor_init
        # Absolute position in VIZ CS for plotting
        initial_uarm_effector_plot_point_viz = conceptual_uarm_physical_base_in_viz_init + uarm_target_effector_in_viz_cs_human_units

    uarm_target_point_viz, = ax_anim_viz.plot(
        [initial_uarm_effector_plot_point_viz[0]], [initial_uarm_effector_plot_point_viz[1]],
        [initial_uarm_effector_plot_point_viz[2]],
        'go', ms=12, mec='k', label='uArm Effector (Conceptual)', zorder=10
    )

    uarm_trail_line_viz, = ax_anim_viz.plot([], [], [], 'g--', alpha=0.7, lw=1.5, label='uArm Effector Trail', zorder=3)
    human_wrist_trail_line_viz, = ax_anim_viz.plot([], [], [], 'm:', alpha=0.7, lw=1.5,
                                                   label=f'Human {TRACKED_ARM} Wrist Trail (Pinned)', zorder=3)

    ax_anim_viz.legend(loc='upper left', fontsize='small');
    # plt.tight_layout() # Called in main usually

    artists = ([human_full_scatter_viz] + human_full_lines_viz +
               [tracked_arm_scatter_viz] + tracked_arm_lines_viz +
               [human_wrist_trail_line_viz])
    if uarm_base_viz: artists.append(uarm_base_viz)
    if uarm_target_point_viz: artists.append(uarm_target_point_viz)
    if uarm_trail_line_viz: artists.append(uarm_trail_line_viz)
    return artists


def update_animation_frame(frame_num_anim: int, passed_controller: 'UArmMimicController'):
    """
    Updates all artists for the current animation frame.
    frame_num_anim: Frame number from FuncAnimation.
    passed_controller: An instance of UArmMimicController to access data.
    """
    global human_full_scatter_viz, human_full_lines_viz, tracked_arm_scatter_viz, tracked_arm_lines_viz
    global uarm_base_viz, uarm_target_point_viz, uarm_trail_line_viz, human_wrist_trail_line_viz
    global ax_anim_viz  # Access the module-level axis

    if not all([passed_controller,
                passed_controller.human_pose_sequence_for_viz is not None,
                passed_controller.key_joint_indices,
                passed_controller.initial_human_root_pos_raw is not None,
                passed_controller.initial_human_shoulder_pos_raw is not None,
                passed_controller.human_skeleton_parents is not None]):
        current_artists = [human_full_scatter_viz] + human_full_lines_viz if human_full_scatter_viz else []
        # ... (add other existing artists to current_artists before returning)
        return filter(None, current_artists)

    current_data_frame_idx = passed_controller.current_human_frame_idx
    max_viz_frames = passed_controller.human_pose_sequence_for_viz.shape[0]

    if not (0 <= current_data_frame_idx < max_viz_frames):
        if passed_controller.stop_thread_flag.is_set() and current_data_frame_idx >= max_viz_frames - 1:
            current_data_frame_idx = max_viz_frames - 1
        else:
            current_artists = [human_full_scatter_viz] + human_full_lines_viz if human_full_scatter_viz else []
            # ... (add other existing artists)
            return filter(None, current_artists)

    current_pose_human_viz = passed_controller.human_pose_sequence_for_viz[current_data_frame_idx]
    human_full_scatter_viz._offsets3d = (
    current_pose_human_viz[:, 0], current_pose_human_viz[:, 1], current_pose_human_viz[:, 2])

    # Efficiently update lines
    line_artist_idx = 0
    for joint_idx_loop in range(passed_controller.human_pose_sequence_for_viz.shape[1]):
        parent_idx_loop = passed_controller.human_skeleton_parents[joint_idx_loop]
        if parent_idx_loop != -1:
            if line_artist_idx < len(human_full_lines_viz):  # Check bounds
                human_full_lines_viz[line_artist_idx].set_data_3d(
                    current_pose_human_viz[[joint_idx_loop, parent_idx_loop], 0],
                    current_pose_human_viz[[joint_idx_loop, parent_idx_loop], 1],
                    current_pose_human_viz[[joint_idx_loop, parent_idx_loop], 2]
                )
            line_artist_idx += 1

    s_idx = passed_controller.key_joint_indices['shoulder']
    e_idx = passed_controller.key_joint_indices['elbow']
    w_idx = passed_controller.key_joint_indices['wrist']
    tracked_arm_data_viz = current_pose_human_viz[[s_idx, e_idx, w_idx]]
    tracked_arm_scatter_viz._offsets3d = (
    tracked_arm_data_viz[:, 0], tracked_arm_data_viz[:, 1], tracked_arm_data_viz[:, 2])
    tracked_arm_lines_viz[0].set_data_3d(current_pose_human_viz[[s_idx, e_idx], 0],
                                         current_pose_human_viz[[s_idx, e_idx], 1],
                                         current_pose_human_viz[[s_idx, e_idx], 2])
    tracked_arm_lines_viz[1].set_data_3d(current_pose_human_viz[[e_idx, w_idx], 0],
                                         current_pose_human_viz[[e_idx, w_idx], 1],
                                         current_pose_human_viz[[e_idx, w_idx], 2])

    current_viz_scale_factor_update = passed_controller.last_calculated_dynamic_scale_factor
    if abs(
        current_viz_scale_factor_update) < 1e-9: current_viz_scale_factor_update = passed_controller.FALLBACK_SCALE_FACTOR_M_TO_MM

    human_shoulder_in_pinned_viz = passed_controller.initial_human_shoulder_pos_raw - passed_controller.initial_human_root_pos_raw
    R_native_to_viz_update = get_rotation_matrix_native_to_viz()
    uarm_conceptual_shoulder_offset_in_native_cs = UARM_SHOULDER_ORIGIN_OFFSET
    uarm_conceptual_shoulder_offset_in_viz_cs_mm = R_native_to_viz_update @ uarm_conceptual_shoulder_offset_in_native_cs
    uarm_conceptual_shoulder_offset_in_viz_cs_human_units = uarm_conceptual_shoulder_offset_in_viz_cs_mm / current_viz_scale_factor_update
    conceptual_uarm_physical_base_in_viz_update = human_shoulder_in_pinned_viz - uarm_conceptual_shoulder_offset_in_viz_cs_human_units

    if uarm_base_viz:
        uarm_base_viz._offsets3d = (
        [conceptual_uarm_physical_base_in_viz_update[0]], [conceptual_uarm_physical_base_in_viz_update[1]],
        [conceptual_uarm_physical_base_in_viz_update[2]])

    if uarm_target_point_viz and passed_controller.latest_uarm_target_abs_mm:
        uarm_target_effector_native_mm = passed_controller.latest_uarm_target_abs_mm[0]
        uarm_target_effector_in_viz_cs_mm = R_native_to_viz_update @ uarm_target_effector_native_mm
        uarm_target_effector_in_viz_cs_human_units = uarm_target_effector_in_viz_cs_mm / current_viz_scale_factor_update
        plot_point_uarm_effector_viz = conceptual_uarm_physical_base_in_viz_update + uarm_target_effector_in_viz_cs_human_units
        uarm_target_point_viz.set_data_3d([plot_point_uarm_effector_viz[0]], [plot_point_uarm_effector_viz[1]],
                                          [plot_point_uarm_effector_viz[2]])

    if uarm_trail_line_viz and passed_controller.uarm_target_trail_mm:
        trail_data_native_mm = np.array(list(passed_controller.uarm_target_trail_mm))
        if trail_data_native_mm.size > 0:
            trail_plot_points_viz = []
            for point_native_mm in trail_data_native_mm:
                point_effector_in_viz_cs_mm = R_native_to_viz_update @ point_native_mm
                point_effector_in_viz_cs_human_units = point_effector_in_viz_cs_mm / current_viz_scale_factor_update
                plot_point = conceptual_uarm_physical_base_in_viz_update + point_effector_in_viz_cs_human_units
                trail_plot_points_viz.append(plot_point)
            if trail_plot_points_viz:
                trail_plot_points_viz_np = np.array(trail_plot_points_viz)
                uarm_trail_line_viz.set_data_3d(trail_plot_points_viz_np[:, 0], trail_plot_points_viz_np[:, 1],
                                                trail_plot_points_viz_np[:, 2])

    if human_wrist_trail_line_viz and passed_controller.human_wrist_trail_for_viz:
        wrist_trail_data_viz = np.array(list(passed_controller.human_wrist_trail_for_viz))
        if wrist_trail_data_viz.size > 0:
            human_wrist_trail_line_viz.set_data_3d(wrist_trail_data_viz[:, 0], wrist_trail_data_viz[:, 1],
                                                   wrist_trail_data_viz[:, 2])

    ax_anim_viz.set_title(f"Frame: {current_data_frame_idx} / {max_viz_frames - 1}")

    artists_to_return = ([human_full_scatter_viz] + human_full_lines_viz +
                         [tracked_arm_scatter_viz] + tracked_arm_lines_viz)
    if uarm_base_viz: artists_to_return.append(uarm_base_viz)
    if uarm_target_point_viz: artists_to_return.append(uarm_target_point_viz)
    if uarm_trail_line_viz: artists_to_return.append(uarm_trail_line_viz)
    if human_wrist_trail_line_viz: artists_to_return.append(human_wrist_trail_line_viz)
    return filter(None, artists_to_return)