#!/bin/bash

# 设置环境变量
export LD_LIBRARY_PATH=/opt/intel/openvino_2025/runtime/lib:$LD_LIBRARY_PATH

echo "启动符文识别节点..."
echo "设置ROS_DOMAIN_ID=1"
export ROS_DOMAIN_ID=1

# 使用launch文件启动节点
ros2 launch rm_rune_detect rune_detect.launch.py --ros-args --log-level debug 