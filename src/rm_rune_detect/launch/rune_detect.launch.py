import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 获取包路径
    pkg_dir = get_package_share_directory('rm_rune_detect')
    
    # 声明启动参数
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'models', 'rune_detect_320.onnx'),
        description='符文检测模型路径'
    )
    
    exposure_factor_arg = DeclareLaunchArgument(
        'exposure_factor',
        default_value='0.6',
        description='曝光调整因子 (0.1-2.0)'
    )
    
    # 设置环境变量，确保OpenVINO库路径正确
    env_vars = [
        SetEnvironmentVariable('LD_LIBRARY_PATH', 
                              '/opt/intel/openvino_2025/runtime/lib:${LD_LIBRARY_PATH}'),
        # 设置日志级别为DEBUG，以显示所有日志信息
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        SetEnvironmentVariable('RCUTILS_LOGGING_USE_STDOUT', '1'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
    ]
    
    # 创建节点配置
    rune_detect_node = Node(
        package='rm_rune_detect',
        executable='rune_detect_node',
        name='rune_detect_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                'model_path': LaunchConfiguration('model_path'),
                'exposure_factor': LaunchConfiguration('exposure_factor'),
                'use_sensor_data_qos': False,  # 使用更快的QoS
                'enable_debug': False,  # 关闭调试输出以提高性能
            }
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # 降低日志级别
    )
    
    # 创建并返回启动描述
    return LaunchDescription(
        env_vars + 
        [
            model_path_arg,
            exposure_factor_arg,
            rune_detect_node
        ]
    ) 