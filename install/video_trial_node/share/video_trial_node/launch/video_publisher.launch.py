from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 声明参数
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('video_trial_node'),
            'videos',
            'rm_rune1.mp4'
        ]),
        description='Path to the video file'
    )
    
    frame_rate_arg = DeclareLaunchArgument(
        'frame_rate',
        default_value='150',
        description='Frame rate to publish video frames'
    )
    
    repeat_arg = DeclareLaunchArgument(
        'repeat',
        default_value='true',
        description='Whether to repeat the video when it ends'
    )
    
    # 创建节点
    video_publisher_node = Node(
        package='video_trial_node',
        executable='video_publisher_node',
        name='video_publisher_node',
        parameters=[{
            'video_path': LaunchConfiguration('video_path'),
            'frame_rate': LaunchConfiguration('frame_rate'),
            'repeat': LaunchConfiguration('repeat'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        video_path_arg,
        frame_rate_arg,
        repeat_arg,
        video_publisher_node,
    ]) 