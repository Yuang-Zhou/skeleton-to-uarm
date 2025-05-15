# main_mapper.py

import os
import sys
import time
import numpy as np # Keep for potential future use, though not directly used now in main
import matplotlib.pyplot as plt # Keep for plt.show() and figure creation
from matplotlib.animation import FuncAnimation # Required for animation
from mpl_toolkits.mplot3d import Axes3D # Required for 3D subplot

# Imports from our refactored modules
from config_arm_mapper import (
    NPZ_FILE_RELATIVE_PATH, SKELETON_TYPE, TRACKED_ARM, UARM_SERIAL_PORT, TARGET_FPS,
    UARM_PLACEMENT_MODE
)
from uarm_mimic_controller import UArmMimicController
from arm_visualization import (
    init_animation_artists, update_animation_frame,
    fig_anim_viz, ax_anim_viz # Import the figure and axes defined in visualization
)
# arm_mapping_utils are used by uarm_mimic_controller, no direct import needed here
# unless you want to call util functions directly from main.

def run_arm_mimicry_with_visualization():
    """
    Main function to run the uArm mimicry with visualization.
    Orchestrates the controller and visualization components.
    """
    # --- Print configuration ---
    print(f"--- Human Arm to uArm Mimicry (Configured via config_arm_mapper.py) ---")
    print(f"    Data File: '{NPZ_FILE_RELATIVE_PATH}'")
    print(f"    Skeleton: '{SKELETON_TYPE}', Tracked Arm: '{TRACKED_ARM}'")
    print(f"    uArm Port: '{UARM_SERIAL_PORT if UARM_SERIAL_PORT else 'Auto-Detect'}'")
    print(f"    Target FPS: {TARGET_FPS}")
    print(f"    uArm Placement Mode: {UARM_PLACEMENT_MODE}") # From config
    print(f"--- End Configuration ---")

    global fig_anim_viz, ax_anim_viz # Allow main to assign to these global fig/ax from arm_visualization.py

    # --- Setup ---
    try:
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root_dir = os.getcwd()
        print(f"Warning (main): __file__ not defined. Using current working directory as project root: {project_root_dir}")

    npz_full_path = os.path.join(project_root_dir, NPZ_FILE_RELATIVE_PATH)

    if not os.path.exists(npz_full_path):
        print(f"Error (main): NPZ data file not found: '{npz_full_path}'")
        print(f"Please ensure '{NPZ_FILE_RELATIVE_PATH}' exists relative to the script or project root.")
        return

    # Instantiate the controller
    controller_instance = UArmMimicController(port=UARM_SERIAL_PORT)

    try:
        # --- Load Data ---
        if not controller_instance.load_human_data(npz_full_path, SKELETON_TYPE, TRACKED_ARM):
            print("Failed to load human data. Exiting.")
            return # Exit if data loading fails

        # --- Connect to uArm ---
        uarm_successfully_connected = controller_instance.connect_uarm()
        if not uarm_successfully_connected:
            print("Warning (main): uArm connection failed. Proceeding with visualization only (if possible).")
            controller_instance.is_connected_and_ready = False # Ensure state reflects reality

        # --- Initial uArm Pose (if connected) and Visualization Prep ---
        if controller_instance.raw_human_pose_sequence is not None and \
           controller_instance.raw_human_pose_sequence.shape[0] > 0:
            # Calculate target for frame 0 to populate initial scale factor and trails for viz
            # This is done even if uArm is not connected, for visualization consistency.
            controller_instance.calculate_uarm_target_for_frame(0, is_initial_move=True)

            if controller_instance.is_connected_and_ready:
                if not controller_instance.move_uarm_to_initial_human_pose(initial_frame_idx=0):
                    print("Warning (main): Failed to accurately move uArm to initial human pose. Continuing...")
                else:
                    print("uArm successfully positioned to initial human pose (frame 0).")
                controller_instance.current_human_frame_idx = 0 # Sync frame index
                time.sleep(1) # Pause after initial uArm positioning
        else:
            print("Error (main): No human data loaded; cannot run animation or uArm control.")
            if controller_instance: controller_instance.cleanup(); return

        # --- Setup Visualization ---
        print("\nPreparing visualization...")
        # Assign to the global fig and ax in arm_visualization module
        # This is a bit unconventional; usually, you'd have a class in arm_visualization
        # that manages its own fig/ax, or these functions return them.
        # For now, we directly use/assign the ones from arm_visualization.
        # If arm_visualization.fig_anim_viz is None, create it.
        if sys.modules['arm_visualization'].fig_anim_viz is None:
             sys.modules['arm_visualization'].fig_anim_viz = plt.figure(figsize=(16, 12))
             sys.modules['arm_visualization'].ax_anim_viz = sys.modules['arm_visualization'].fig_anim_viz.add_subplot(111, projection='3d')


        # --- Start uArm Control Thread (if connected) ---
        mimicry_control_started = False
        if controller_instance.is_connected_and_ready:
            print("Starting uArm control thread...")
            if controller_instance.start_mimicry(start_from_frame_one=True): # Start from frame 1 if frame 0 was for init
                mimicry_control_started = True
            else:
                print("Failed to start uArm control thread. Visualization will run without live uArm movement.")
        else:
            print("uArm not connected. Visualization will run; uArm control thread will not start.")

        # --- Start Animation ---
        if controller_instance.raw_human_pose_sequence is None:
            print("Error (main): Human pose sequence is None before starting animation.")
            return
        num_animation_frames = controller_instance.raw_human_pose_sequence.shape[0]
        animation_interval_ms = int(1000 / TARGET_FPS)

        print(f"Starting Matplotlib animation for {num_animation_frames} frames at {TARGET_FPS} FPS ({animation_interval_ms}ms interval)...")
        # Pass the controller instance to init and update functions via fargs
        ani = FuncAnimation(sys.modules['arm_visualization'].fig_anim_viz,
                            update_animation_frame,
                            frames=num_animation_frames,
                            init_func=lambda: init_animation_artists(controller_instance), # Use lambda to pass controller
                            fargs=(controller_instance,), # Pass controller to update_animation_frame
                            blit=False, # Blit=False is often more robust for 3D Matplotlib
                            interval=animation_interval_ms,
                            repeat=False) # Do not repeat animation automatically

        plt.tight_layout() # Adjust layout to prevent overlapping elements
        plt.show() # This call blocks until the animation window is closed.

        print("Plot window closed or animation finished.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller_instance' in locals() and controller_instance:
            print("Initiating cleanup sequence in main...")
            controller_instance.cleanup()
        print("\n--- Program Finished ---")

if __name__ == '__main__':
    run_arm_mimicry_with_visualization()