import time
import sys
import os
import numpy as np # 用于 np.clip

# --- 帮助：uArm SDK的路径设置 (确保路径正确) ---
try:
    _current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # --- MODIFIED LINE: Correctly define the project root ---
    _project_root = os.path.dirname(_current_script_dir) # 项目根目录是脚本所在目录的上一级
    # --- END OF MODIFICATION ---

    _sdk_base_dir = os.path.join(_project_root, "uarm-python-sdk") # Path to the folder containing 'uArm-Python-SDK-2.0'
    _sdk_actual_path = os.path.join(_sdk_base_dir, "uArm-Python-SDK-2.0") # Path to the specific SDK version folder

    # This check ensures that inside _sdk_actual_path, there is a directory named 'uarm'
    # which is the actual Python package.
    if os.path.isdir(os.path.join(_sdk_actual_path, "uarm")) and _sdk_actual_path not in sys.path:
        sys.path.insert(0, _sdk_actual_path)
        print(f"信息: 已添加uArm SDK路径 '{_sdk_actual_path}' 到sys.path。")
    elif not os.path.isdir(os.path.join(_sdk_actual_path, "uarm")):
        print(f"错误: 在期望的SDK路径 '{_sdk_actual_path}' 下未找到 'uarm' 文件夹。请检查SDK结构。")
        # You might want to print the expected structure or sys.path for debugging
        print(f"DEBUG: _current_script_dir = {_current_script_dir}")
        print(f"DEBUG: _project_root = {_project_root}")
        print(f"DEBUG: _sdk_base_dir = {_sdk_base_dir}")
        print(f"DEBUG: _sdk_actual_path = {_sdk_actual_path}")
        sys.exit(1)


    from uarm.wrapper import SwiftAPI
    print("成功导入uArm SDK。")
except ImportError as e:
    print(f"严重导入错误: {e}。请确保uArm SDK已正确放置并可访问。")
    print(f"DEBUG (ImportError): _sdk_actual_path = {_sdk_actual_path if '_sdk_actual_path' in locals() else 'Not defined'}")
    print(f"DEBUG (ImportError): sys.path includes = {sys.path}")
    sys.exit(1)
except NameError as e: # Catch if _sdk_actual_path was not defined due to path issues before the check
    print(f"路径配置错误: {e}")
    sys.exit(1)
# --- 路径设置结束 ---


# --- uArm 配置 ---
# !!! 关键: 请替换为你的uArm实际串口号 !!!
# macOS 示例: '/dev/cu.usbmodemXXXXX' 或 '/dev/tty.usbmodemXXXXX'
# Windows 示例: 'COMX' (例如 'COM3')
# Linux 示例: '/dev/ttyACMX' 或 '/dev/ttyUSBX'
UARM_SERIAL_PORT = '/dev/cu.usbmodem144301' # <--- 请务必修改这里! 如果设为None，SDK会尝试自动检测

DEFAULT_MOVE_SPEED_MMPM = 1500  # 测试时移动速度 (mm/min) - 调低以确保安全
DEFAULT_RESET_SPEED_MMPM = 3000 # 重置时速度 (mm/min)

# 定义一个大致的归位/安全位置 (uArm Swift Pro重置后通常在此位置附近)
# SDK坐标系: X向前, Y向上, Z向右
HOME_POSITION_SDK = {'x': 200.0, 'y': 150.0, 'z': 0.0}

# 记录探索到的边界
explored_limits = {
    "min_x": float('inf'), "max_x": float('-inf'),
    "min_y": float('inf'), "max_y": float('-inf'),
    "min_z": float('inf'), "max_z": float('-inf'),
}

def connect_uarm(port: str | None) -> SwiftAPI | None:
    """尝试连接到uArm并准备就绪。"""
    print(f"\n正在尝试连接到uArm，串口: {port if port else '自动检测'}...")
    swift = None
    try:
        # 注意：根据你的 minimum_uarm_test.py, 你可能需要特定的参数来初始化 SwiftAPI
        # 例如 enable_handle_thread=True, enable_write_thread=False 等
        # 这里使用一个较为通用的初始化方式，如果连接不稳定，可以参考你的测试脚本进行调整
        swift = SwiftAPI(port=port,
                         enable_handle_thread=True, # 基于 minimum_uarm_test.py
                         enable_write_thread=False, # 基于 minimum_uarm_test.py
                         enable_handle_report_thread=False, # 基于 minimum_uarm_test.py
                         cmd_pend_size=5 # 基于 minimum_uarm_test.py
                         )
        print("等待SDK内部连接...")
        time.sleep(2.0) # 给SDK一些时间建立连接

        if not swift.connected:
            print("错误: SwiftAPI未能建立串口连接 (swift.connected is False)。")
            if swift: swift.disconnect() # 尝试清理
            return None

        print("串口已连接。获取设备信息...")
        device_info = swift.get_device_info(timeout=10)
        if device_info:
            print(f"设备信息: {device_info}")
        else:
            print("警告: 未能获取设备信息。")


        print("等待uArm准备就绪 (waiting_ready)...")
        is_ready = swift.waiting_ready(timeout=30) # 增加超时
        print(f"waiting_ready() 返回: {is_ready}")
        print(f"等待后的电源状态 (属性): {swift.power_status}")


        # 即使 waiting_ready 返回 None 或 False，也检查一下电源状态
        # 因为在你的 minimum_uarm_test.py 中，有时 arm 仍然可以工作
        if not swift.power_status:
            print("警告: uArm电源状态为关闭。请确保机械臂已上电。")
            # 你可以选择在这里返回 None 或者继续尝试，取决于你测试的结果
            # return None # 如果电源关闭则认为连接失败

        print("尝试设置舵机附加 (servo attach)...")
        attach_result = swift.set_servo_attach(wait=True, timeout=10)
        print(f"set_servo_attach() 返回: {attach_result}")
        if attach_result != 'OK':
            print("警告: set_servo_attach 未返回 'OK'。")
        time.sleep(0.5)

        print(f"尝试重置机械臂到初始位置 (速度: {DEFAULT_RESET_SPEED_MMPM} mm/min)...")
        reset_result = swift.reset(speed=DEFAULT_RESET_SPEED_MMPM, wait=True, timeout=30)
        print(f"swift.reset() 指令返回: {reset_result}")
        time.sleep(1.0) # 等待物理移动完成

        pos_after_reset = swift.get_position(wait=True, timeout=10)
        print(f"重置后的位置: {pos_after_reset}")

        if isinstance(pos_after_reset, list) and \
           (190 < pos_after_reset[0] < 210 and \
            140 < pos_after_reset[1] < 160 and \
            -10 < pos_after_reset[2] < 10): # 调整 Y 和 Z 的检查顺序以匹配 SDK X, Y (up), Z (right)
            print("uArm已连接、重置并准备就绪。")
            return swift
        elif isinstance(pos_after_reset, list):
            print("警告: 重置后机械臂位置不在预期的HOME附近。请手动确认其状态。")
            return swift # 仍然返回对象，让用户决定是否继续
        else:
            print("错误: 重置失败或未能获取到有效位置。")
            if swift.connected:
                swift.disconnect()
            return None

    except Exception as e:
        print(f"连接或初始化uArm时发生错误: {e}")
        import traceback
        traceback.print_exc()
        if swift and swift.connected:
            try:
                swift.disconnect()
            except: pass
        return None


def disconnect_uarm(swift: SwiftAPI | None, reset_on_exit: bool = True):
    """断开与uArm的连接，并可选择在退出前重置。"""
    if swift and swift.connected:
        print("\n正在断开连接...")
        if reset_on_exit:
            print(f"正在重置机械臂至初始位置 (速度: {DEFAULT_RESET_SPEED_MMPM} mm/min)...")
            try:
                reset_result = swift.reset(speed=DEFAULT_RESET_SPEED_MMPM, wait=True, timeout=30)
                print(f"重置指令返回: {reset_result}")
                print("机械臂已尝试重置。")
            except Exception as e:
                print(f"重置机械臂时发生错误: {e}")
        swift.disconnect()
        print("已与uArm断开连接。")
    elif swift:
        print("uArm对象存在但未连接。尝试执行断开操作...")
        try:
            swift.disconnect() # 清理资源
        except Exception as e:
            print(f"尝试断开未连接的uArm对象时出错: {e}")
    else:
        print("uArm未连接或对象不存在。")

def move_to_target_and_report(swift: SwiftAPI, x: float, y: float, z: float, speed: int) -> dict | None:
    """将uArm移动到指定的目标坐标并报告实际位置。"""
    if not swift or not swift.connected:
        print("错误: uArm未连接，无法移动。")
        return None

    print(f"\n指令移动到 (SDK坐标系): X={x:.1f} (前), Y={y:.1f} (上), Z={z:.1f} (右) | 速度: {speed} mm/min")
    try:
        move_result = swift.set_position(x=x, y=y, z=z, speed=speed, wait=True, timeout=30) # 增加超时
        print(f"set_position 指令返回: {move_result}")

        if move_result != 'OK' and move_result != 'E22': # E22: 超出工作范围，但也算执行了
             print(f"警告: 移动指令可能未成功执行 (返回: {move_result})。请检查机械臂状态。")

        time.sleep(0.2) # 短暂等待机械臂稳定

        current_pos_list = swift.get_position(wait=True, timeout=10)
        if current_pos_list and isinstance(current_pos_list, list) and len(current_pos_list) == 3:
            actual_x, actual_y, actual_z = current_pos_list[0], current_pos_list[1], current_pos_list[2]
            print(f"报告实际位置 (SDK坐标系): X={actual_x:.1f}, Y={actual_y:.1f}, Z={actual_z:.1f}")

            # 更新探索到的边界
            global explored_limits
            explored_limits["min_x"] = min(explored_limits["min_x"], actual_x)
            explored_limits["max_x"] = max(explored_limits["max_x"], actual_x)
            explored_limits["min_y"] = min(explored_limits["min_y"], actual_y)
            explored_limits["max_y"] = max(explored_limits["max_y"], actual_y)
            explored_limits["min_z"] = min(explored_limits["min_z"], actual_z)
            explored_limits["max_z"] = max(explored_limits["max_z"], actual_z)

            return {'x': actual_x, 'y': actual_y, 'z': actual_z}
        else:
            print(f"警告: 获取当前位置失败或格式不正确。收到: {current_pos_list}")
            return None
    except Exception as e:
        print(f"移动到目标位置时发生错误: {e}")
        try:
            err_pos = swift.get_position(wait=True, timeout=5)
            print(f"错误发生后，报告位置: {err_pos}")
        except:
            print("错误发生后，获取位置也失败。")
        return None

def get_user_coordinates_or_command(current_pos_dict: dict | None) -> tuple[str, float | None, float | None, float | None]:
    """获取用户输入的坐标值或特殊指令。"""
    prompt_parts = []
    if current_pos_dict:
        prompt_parts.append(f"当前约: X={current_pos_dict.get('x',0):.1f}, Y={current_pos_dict.get('y',0):.1f}, Z={current_pos_dict.get('z',0):.1f}")
    else:
        prompt_parts.append("当前位置未知")

    prompt_parts.append("\n输入指令: 目标坐标(X,Y,Z)")
    prompt_parts.append("  或 'h' (归位), 'r' (重置), 's' (显示已记录范围), 'q' (退出)")
    prompt_parts.append("  或 '+x VAL', '-x VAL', '+y VAL', ... 进行增量移动 (例如: +x 10)")
    prompt_str = "\n".join(prompt_parts) + "\n> "

    user_input = input(prompt_str).strip().lower()

    if user_input == 'q': return 'quit', None, None, None
    if user_input == 'h': return 'home', None, None, None
    if user_input == 'r': return 'reset', None, None, None
    if user_input == 's': return 'show_limits', None, None, None

    # 检查增量移动指令
    parts_incremental = user_input.split()
    if len(parts_incremental) == 2 and parts_incremental[0] in ['+x', '-x', '+y', '-y', '+z', '-z']:
        try:
            value = float(parts_incremental[1])
            axis = parts_incremental[0][1]
            direction = 1.0 if parts_incremental[0][0] == '+' else -1.0
            delta_x = value * direction if axis == 'x' else 0.0
            delta_y = value * direction if axis == 'y' else 0.0
            delta_z = value * direction if axis == 'z' else 0.0
            return 'incremental', delta_x, delta_y, delta_z
        except ValueError:
            print("增量值无效，必须是数字。")
            return 'error', None, None, None

    # 检查绝对坐标指令
    try:
        parts_absolute = user_input.split(',')
        if len(parts_absolute) == 3:
            x = float(parts_absolute[0])
            y = float(parts_absolute[1])
            z = float(parts_absolute[2])
            return 'move', x, y, z
        else:
            print("输入格式错误。请输入X,Y,Z坐标，或有效指令。")
            return 'error', None, None, None
    except ValueError:
        print("坐标值无效，必须是数字。")
        return 'error', None, None, None
    except Exception as e:
        print(f"处理输入时发生未知错误: {e}")
        return 'error', None, None, None

def print_explored_limits():
    print("\n--- 已探索的工作空间范围 (SDK坐标系) ---")
    if explored_limits["min_x"] == float('inf'):
        print("尚未记录任何有效移动。")
    else:
        print(f"  X轴 (前/后): 从 {explored_limits['min_x']:.1f} 到 {explored_limits['max_x']:.1f} mm")
        print(f"  Y轴 (上/下): 从 {explored_limits['min_y']:.1f} 到 {explored_limits['max_y']:.1f} mm")
        print(f"  Z轴 (左/右): 从 {explored_limits['min_z']:.1f} 到 {explored_limits['max_z']:.1f} mm")
    print("--------------------------------------")

def main():
    swift: SwiftAPI | None = None
    current_position_dict: dict | None = None # 初始化为None

    try:
        print("--- uArm机械臂工作空间手动探索脚本 ---")
        print(f"确保uArm已上电并通过USB连接。脚本将尝试连接到: {UARM_SERIAL_PORT if UARM_SERIAL_PORT else '自动检测'}")
        if UARM_SERIAL_PORT == '/dev/cu.usbmodem144201' or (UARM_SERIAL_PORT and UARM_SERIAL_PORT.upper() == 'COMX'):
             print("警告: 你可能需要修改脚本中的 UARM_SERIAL_PORT 变量为你的实际串口号!")
        input("准备好后按Enter键开始连接...")

        swift = connect_uarm(UARM_SERIAL_PORT)

        if not swift:
            print("未能连接到uArm。请检查连接和设置后重试。程序即将退出。")
            return

        # 连接成功后，获取一次当前位置作为初始current_position_dict
        pos_after_connect = swift.get_position(wait=True, timeout=10)
        if pos_after_connect and isinstance(pos_after_connect, list) and len(pos_after_connect) == 3:
            current_position_dict = {'x': pos_after_connect[0], 'y': pos_after_connect[1], 'z': pos_after_connect[2]}
        else: # 如果获取失败，使用HOME作为备用
            current_position_dict = HOME_POSITION_SDK.copy()
            print(f"未能获取准确初始位置，假设在HOME: {current_position_dict}")


        # 主测试循环
        while True:
            command, val1, val2, val3 = get_user_coordinates_or_command(current_position_dict)

            if command == 'quit':
                break
            elif command == 'home':
                print("指令: 移动到HOME位置...")
                current_position_dict = move_to_target_and_report(swift,
                                                                  HOME_POSITION_SDK['x'],
                                                                  HOME_POSITION_SDK['y'],
                                                                  HOME_POSITION_SDK['z'],
                                                                  DEFAULT_MOVE_SPEED_MMPM)
            elif command == 'reset':
                print("指令: 重置机械臂...")
                try:
                    reset_result = swift.reset(speed=DEFAULT_RESET_SPEED_MMPM, wait=True, timeout=30)
                    print(f"重置指令返回: {reset_result}")
                    time.sleep(1.0)
                    pos_after_reset = swift.get_position(wait=True, timeout=10)
                    if pos_after_reset and isinstance(pos_after_reset, list) and len(pos_after_reset) == 3:
                        current_position_dict = {'x': pos_after_reset[0], 'y': pos_after_reset[1], 'z': pos_after_reset[2]}
                        print(f"重置后位置: {current_position_dict}")
                    else:
                        current_position_dict = HOME_POSITION_SDK.copy() # 获取失败则假设在Home
                        print(f"重置后未能获取准确位置，假设在HOME: {current_position_dict}")
                except Exception as e:
                    print(f"重置过程中发生错误: {e}")
                    # 可以选择尝试获取位置，或者直接设为None/HOME
                    current_position_dict = HOME_POSITION_SDK.copy()
            elif command == 'move':
                if val1 is not None and val2 is not None and val3 is not None:
                    current_position_dict = move_to_target_and_report(swift, val1, val2, val3, DEFAULT_MOVE_SPEED_MMPM)
            elif command == 'incremental':
                if current_position_dict and val1 is not None and val2 is not None and val3 is not None:
                    target_x = current_position_dict.get('x', HOME_POSITION_SDK['x']) + val1
                    target_y = current_position_dict.get('y', HOME_POSITION_SDK['y']) + val2
                    target_z = current_position_dict.get('z', HOME_POSITION_SDK['z']) + val3
                    current_position_dict = move_to_target_and_report(swift, target_x, target_y, target_z, DEFAULT_MOVE_SPEED_MMPM)
                else:
                    print("错误: 无法进行增量移动，当前位置未知。请先移动到一个绝对位置。")
            elif command == 'show_limits':
                print_explored_limits()
            elif command == 'error':
                print("无效输入，请重试。")
            else:
                print("未知指令，请重试。")

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在尝试安全退出...")
    except Exception as e:
        print(f"主程序发生未处理的错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_explored_limits() # 退出前打印最终记录的范围
        if swift:
            disconnect_uarm(swift, reset_on_exit=True)
        print("测试脚本执行完毕。")

if __name__ == '__main__':
    main()