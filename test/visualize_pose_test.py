import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
from matplotlib.animation import FuncAnimation  # 导入动画模块
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 假设您的脚本 (visualize_pose_test.py) 与 src 目录在同一级别
try:
    from src.datasets.amass_dataset import AMASSSubsetDataset
    from src.kinematics.skeleton_utils import get_skeleton_parents, get_num_joints

    print("模块导入成功!")
except ImportError as e:
    print(f"模块导入失败: {e}")
    # ... (之前的错误处理) ...
    exit()

# 全局变量，用于存储绘图艺术家对象，以便在 update 函数中访问和更新
scatter_plot = None
bone_lines = []
title_obj = None
skeleton_parents_global = None  # 用于在 update 函数中访问


def init_animation_plot(ax, first_pose_data_np, skeleton_type):
    """初始化动画的绘图区域，绘制第一帧"""
    global scatter_plot, bone_lines, title_obj, skeleton_parents_global

    num_joints = first_pose_data_np.shape[0]
    if first_pose_data_np.shape != (num_joints, 3):
        print(f"错误的姿态数据形状: {first_pose_data_np.shape}. 期望 ({num_joints}, 3)")
        return False

    skeleton_parents_global = get_skeleton_parents(skeleton_type)

    # 绘制关节点
    # 注意：这里将 scatter_plot 赋值给全局变量
    scatter_plot = ax.scatter(first_pose_data_np[:, 0], first_pose_data_np[:, 1], first_pose_data_np[:, 2],
                              c='deepskyblue', marker='o', s=60, edgecolors='black', linewidth=0.5, depthshade=True)

    # 绘制骨骼
    # 清空之前的 bone_lines (如果再次调用此函数)
    for line in bone_lines:
        line.remove()
    bone_lines.clear()

    for i, p in enumerate(skeleton_parents_global):
        if p != -1:
            line, = ax.plot([first_pose_data_np[i, 0], first_pose_data_np[p, 0]],
                            [first_pose_data_np[i, 1], first_pose_data_np[p, 1]],
                            [first_pose_data_np[i, 2], first_pose_data_np[p, 2]],
                            'r-', linewidth=2.5)
            bone_lines.append(line)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 注意：这里将 title_obj 赋值给全局变量
    title_obj = ax.set_title("Pose Animation - Frame 0")

    # 设置固定的坐标轴范围，防止动画过程中抖动
    # 您可以根据您的数据动态计算这些值，或者预设一个较大的范围
    all_frames_data = first_pose_data_np  # 这里仅用第一帧来估计范围，更好的做法是传入整个序列
    x_min, x_max = all_frames_data[:, 0].min() - 0.5, all_frames_data[:, 0].max() + 0.5
    y_min, y_max = all_frames_data[:, 1].min() - 0.5, all_frames_data[:, 1].max() + 0.5
    z_min, z_max = all_frames_data[:, 2].min() - 0.5, all_frames_data[:, 2].max() + 0.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # 尝试保持各轴比例一致
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.view_init(elev=20., azim=-45)  # 设置初始视角

    return True


def update_animation_frame(frame_num, all_frames_data_np):
    """更新动画的每一帧"""
    global scatter_plot, bone_lines, title_obj, skeleton_parents_global

    current_pose = all_frames_data_np[frame_num]

    # 更新关节点位置
    # 对于 scatter._offsets3d，matplotlib 期望 (xs, ys, zs) 的元组
    scatter_plot._offsets3d = (current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])

    # 更新骨骼连接
    line_idx = 0
    for i, p in enumerate(skeleton_parents_global):
        if p != -1:
            bone_lines[line_idx].set_data([current_pose[i, 0], current_pose[p, 0]],
                                          [current_pose[i, 1], current_pose[p, 1]])
            bone_lines[line_idx].set_3d_properties([current_pose[i, 2], current_pose[p, 2]])
            line_idx += 1

    title_obj.set_text(f'Pose Animation - Frame {frame_num} / {all_frames_data_np.shape[0] - 1}')

    # FuncAnimation 需要返回一个包含所有已更改艺术家的元组/列表
    return [scatter_plot] + bone_lines + [title_obj]


if __name__ == '__main__':
    data_dir = "../data/00"
    try:
        npz_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        if not npz_files:
            print(f"在 '{data_dir}' 目录下没有找到 .npz 文件。")
            exit()
        file_to_visualize = npz_files[0]
        print(f"尝试可视化文件: {file_to_visualize}")
    except FileNotFoundError:
        print(f"数据目录 '{data_dir}' 未找到。")
        exit()

    # --- 加载整个序列用于动画 ---
    # 为了加载整个序列，我们需要知道它的长度，或者设置一个足够大的 window_size
    # 让我们先读取 .npz 文件来获取序列长度
    try:
        raw_data = np.load(file_to_visualize)
        if 'poses_r3j' not in raw_data:
            print(f"错误: 文件 {file_to_visualize} 中缺少 'poses_r3j' 键。")
            exit()
        full_sequence_length = raw_data['poses_r3j'].shape[0]
        print(f"文件 {file_to_visualize} 中的完整序列长度为: {full_sequence_length} 帧。")
    except Exception as e:
        print(f"读取文件 {file_to_visualize} 以获取序列长度时出错: {e}")
        exit()

    # 可以选择动画化的帧数，例如，如果序列太长，可以只取一部分
    frames_to_animate = min(full_sequence_length, 300)  # 最多动画300帧，或者序列的全部长度

    skeleton_type = 'smpl_24'
    num_joints_expected = get_num_joints(skeleton_type)

    try:
        dataset = AMASSSubsetDataset(
            data_paths=[file_to_visualize],
            window_size=frames_to_animate,  # 设置 window_size 为我们希望动画的帧数
            skeleton_type=skeleton_type,
            is_train=False,
            center_around_root=True,  # 动画时通常中心化根节点效果较好
            gaussian_noise_std=0.0,
            temporal_noise_type='none',  # 确保加载干净数据
            # ... 其他噪声参数设为不激活 ...
        )
    except Exception as e:
        print(f"初始化 AMASSSubsetDataset 时出错: {e}")
        exit()

    if len(dataset) > 0:
        print(f"数据集成功加载，基于 window_size={frames_to_animate} 创建了 {len(dataset)} 个窗口。")
        try:
            # 我们期望只创建一个窗口，因为 window_size 设置为序列长度
            _, clean_window_torch, _ = dataset[0]
        except IndexError:
            print("错误：无法从数据集中获取样本。检查 window_size 和实际序列长度。")
            exit()

        print(f"加载的动画数据形状 (torch): {clean_window_torch.shape}")
        # 期望形状: (frames_to_animate, num_joints, 3)

        if clean_window_torch.shape[0] > 0 and \
                clean_window_torch.shape[1] == num_joints_expected and \
                clean_window_torch.shape[2] == 3:

            animation_data_np = clean_window_torch.numpy()

            # --- 创建 Matplotlib 图形和子图 ---
            fig = plt.figure(figsize=(10, 10))  # 可以调整图形大小
            ax = fig.add_subplot(111, projection='3d')

            # --- 初始化绘图（绘制第一帧）---
            if not init_animation_plot(ax, animation_data_np[0], skeleton_type):
                print("初始化动画绘图失败。")
                exit()

            # 调整坐标轴范围以适应整个动画序列
            x_min_anim, x_max_anim = animation_data_np[..., 0].min() - 0.5, animation_data_np[..., 0].max() + 0.5
            y_min_anim, y_max_anim = animation_data_np[..., 1].min() - 0.5, animation_data_np[..., 1].max() + 0.5
            z_min_anim, z_max_anim = animation_data_np[..., 2].min() - 0.5, animation_data_np[..., 2].max() + 0.5

            ax.set_xlim(x_min_anim, x_max_anim)
            ax.set_ylim(y_min_anim, y_max_anim)
            ax.set_zlim(z_min_anim, z_max_anim)

            max_range_anim = np.array([x_max_anim - x_min_anim, y_max_anim - y_min_anim, z_max_anim - z_min_anim]).max()
            mid_x_anim = (x_max_anim + x_min_anim) * 0.5
            mid_y_anim = (y_max_anim + y_min_anim) * 0.5
            mid_z_anim = (z_max_anim + z_min_anim) * 0.5
            ax.set_xlim(mid_x_anim - max_range_anim / 2, mid_x_anim + max_range_anim / 2)
            ax.set_ylim(mid_y_anim - max_range_anim / 2, mid_y_anim + max_range_anim / 2)
            ax.set_zlim(mid_z_anim - max_range_anim / 2, mid_z_anim + max_range_anim / 2)

            # --- 创建动画 ---
            # fargs 传递给 update_animation_frame 函数的额外参数
            # interval 是两帧之间的延迟（毫秒），例如 50ms -> 20 FPS
            ani = FuncAnimation(fig, update_animation_frame, frames=animation_data_np.shape[0],
                                fargs=(animation_data_np,), interval=50, blit=False, repeat=True)

            # --- 显示动画 ---
            print("正在显示动画...")
            plt.tight_layout()  # 调整布局
            plt.show()

            # --- (可选) 保存动画 ---
            # save_animation = True
            # if save_animation:
            #     try:
            #         animation_output_path = "pose_animation.mp4"
            #         print(f"正在保存动画到 {animation_output_path} (这可能需要一些时间)...")
            #         # 需要安装 ffmpeg: conda install ffmpeg -c conda-forge
            #         # 或者对于GIF: pip install imageio
            #         ani.save(animation_output_path, writer='ffmpeg', fps=20) # 或者 writer=PillowWriter(fps=20) for gif
            #         print("动画保存完毕。")
            #     except Exception as e:
            #         print(f"保存动画失败: {e}")
            #         print("确保已安装 ffmpeg (用于mp4) 或 imageio (用于gif)。")

        else:
            print(
                f"加载的动画数据形状不正确。期望: ({frames_to_animate} 或更小, {num_joints_expected}, 3)，实际: {clean_window_torch.shape}")
    else:
        print(f"未能从文件 {file_to_visualize} 创建任何窗口。")