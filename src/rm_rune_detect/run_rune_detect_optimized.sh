#!/bin/bash

echo "启动高性能符文识别节点..."

# 设置CPU性能模式（需要root权限）
if [ "$EUID" -eq 0 ]; then
    echo "设置CPU性能模式..."
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
else
    echo "注意：需要root权限才能设置CPU性能模式"
    echo "建议运行: sudo echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
fi

# 设置环境变量以优化性能
export OMP_NUM_THREADS=$(nproc)  # 使用所有CPU核心
export OPENVINO_LOG_LEVEL=1  # 减少OpenVINO日志输出
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_COLORIZED_OUTPUT=1

# 设置OpenVINO路径
export LD_LIBRARY_PATH="/opt/intel/openvino_2025/runtime/lib:${LD_LIBRARY_PATH}"

# 显示系统信息
echo "CPU核心数: $(nproc)"
echo "OpenMP线程数: $OMP_NUM_THREADS"
echo "OpenVINO日志级别: $OPENVINO_LOG_LEVEL"

# 构建项目（Release模式）
echo "构建项目（Release模式）..."
cd ~/code
colcon build --packages-select rm_rune_detect --cmake-args -DCMAKE_BUILD_TYPE=Release

# 启动节点
echo "启动节点..."
source install/setup.bash
ros2 launch rm_rune_detect rune_detect.launch.py 