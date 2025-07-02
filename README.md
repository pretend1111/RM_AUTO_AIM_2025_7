# RM_AUTO_AIM_2025_7

RM Auto Aim Detection System - 基于 ROS2 的自动瞄准检测系统，用于 RoboMaster 比赛。

## 项目结构

- `src/rm_armor_detect/` - 装甲板检测模块

  - 使用 OpenVINO 进行装甲板检测
  - 包含灯条检测和数字分类功能
  - 支持多种车型检测

- `src/rm_rune_detect/` - 符文检测模块

  - 基于深度学习的符文识别
  - 包含预处理和后处理功能

- `src/rm_armor_tracker/` - 装甲板跟踪模块

  - 目标跟踪算法

- `src/ros2-hik-camera-main/` - 海康相机驱动

  - 相机节点和配置文件
  - 支持海康工业相机

- `src/video_trial_node/` - 视频发布节点
  - 用于测试和调试的视频流发布

## 依赖要求

- ROS2 (Humble/Foxy)
- OpenVINO
- OpenCV
- 海康相机 SDK

## 编译和运行

```bash
# 编译
colcon build

# 运行装甲板检测
ros2 launch rm_armor_detect armor_detect.launch.py

# 运行符文检测
ros2 launch rm_rune_detect rune_detect.launch.py
```
