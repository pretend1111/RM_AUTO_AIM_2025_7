import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 获取包路径
    pkg_dir = get_package_share_directory('rm_armor_detect')
    
    # 参数文件路径
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'armor_detect_params.yaml'),
        description='参数配置文件路径'
    )
    
    # 声明其他启动参数
    topic_name_arg = DeclareLaunchArgument(
        'topic_name',
        default_value='image_raw',
        description='订阅的图像话题名称'
    )
    
    display_image_arg = DeclareLaunchArgument(
        'display_image',
        default_value='true',
        description='是否显示图像窗口'
    )
    
    # 设置环境变量，确保OpenVINO库路径正确
    env_vars = [
        SetEnvironmentVariable('LD_LIBRARY_PATH', 
                              '/opt/intel/openvino_2025/runtime/lib:${LD_LIBRARY_PATH}'),
        # 设置日志级别，显示调试信息
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        SetEnvironmentVariable('RCUTILS_LOGGING_USE_STDOUT', '1'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
    ]
    
    # 创建装甲板检测节点
    armor_detect_node = Node(
        package='rm_armor_detect',
        executable='video_subscriber_node',
        name='video_subscriber_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'topic_name': LaunchConfiguration('topic_name'),
                'display_image': LaunchConfiguration('display_image'),
            }
        ],
        arguments=['--ros-args', '--log-level', 'info'],
    )
    
    # 创建并返回启动描述
    return LaunchDescription(
        env_vars + 
        [
            params_file_arg,
            topic_name_arg,
            display_image_arg,
            armor_detect_node
        ]
    ) 