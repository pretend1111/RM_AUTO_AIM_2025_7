# 装甲板检测参数配置指南

## 概述

本文档详细说明了 `rm_armor_detect` 节点的参数配置系统。所有参数都可以通过配置文件 `armor_detect_params.yaml` 进行调整，无需重新编译代码。

## 配置文件结构

### 1. 基础订阅参数
```yaml
# 基础订阅参数
topic_name: "image_raw"                    # 订阅的图像话题名称
display_image: true                        # 是否显示图像窗口
window_name: "装甲板检测显示"              # 显示窗口名称
use_sensor_data_qos: true                  # 是否使用sensor_data QoS设置
```

### 2. 曝光增强参数
```yaml
# 曝光增强参数
enable_exposure_enhancement: true          # 是否启用曝光增强
exposure_gain: 0.7                         # 曝光增益系数 (0.1-2.0)
```

### 3. 车辆检测参数
```yaml
# 车辆检测参数
enable_car_detection: true                 # 是否启用车辆检测
model_path: "~/code/src/rm_armor_detect/models/car/car.onnx"  # 车辆检测模型路径
detection_confidence_threshold: 0.05       # 车辆检测置信度阈值 (0.0-1.0)
```

### 4. 数字分类参数
```yaml
# 数字分类参数
enable_number_classification: true         # 是否启用数字分类
number_model_path: "~/code/src/rm_armor_detect/models/number/mlp.onnx"     # 数字分类模型路径
number_label_path: "~/code/src/rm_armor_detect/models/number/label.txt"    # 数字分类标签文件路径
number_confidence_threshold: 0.7           # 数字分类置信度阈值 (0.0-1.0)
```

### 5. 基础检测参数
```yaml
# 基础检测参数
binary_threshold: 150                      # 二值化阈值 (0-255)
detect_color: 1                            # 检测颜色 (0:蓝色, 1:红色)
debug_contours: true                       # 是否显示轮廓调试信息
```

### 6. 灯条检测参数 ⭐重要
```yaml
# 灯条检测参数
light_params:
  min_ratio: 0.1                           # 灯条最小宽高比 (短边/长边)
  max_ratio: 0.6                           # 灯条最大宽高比
  max_angle: 40.0                          # 灯条最大倾斜角度(度)
```

### 7. 装甲板匹配参数 ⭐重要
```yaml
# 装甲板匹配参数
armor_params:
  min_light_ratio: 0.7                     # 灯条最小长度比 (短灯条/长灯条)
  min_small_center_distance: 0.8           # 小装甲板最小中心距 (单位:灯条长度)
  max_small_center_distance: 3.2           # 小装甲板最大中心距
  min_large_center_distance: 3.2           # 大装甲板最小中心距
  max_large_center_distance: 5.5           # 大装甲板最大中心距
  max_angle: 35.0                          # 装甲板最大角度(度)
```

## 使用方法

### 1. 使用默认配置启动
```bash
ros2 launch rm_armor_detect armor_detect.launch.py
```

### 2. 使用自定义配置文件
```bash
ros2 launch rm_armor_detect armor_detect.launch.py params_file:=/path/to/your/config.yaml
```

### 3. 启动时覆盖特定参数
```bash
ros2 launch rm_armor_detect armor_detect.launch.py topic_name:=/camera/image_raw display_image:=false
```

### 4. 运行时动态调整参数
```bash
ros2 param set /video_subscriber_node binary_threshold 200
ros2 param set /video_subscriber_node light_params.max_angle 45.0
```

## 调参指南

### 优先级调整顺序

1. **二值化阈值 (`binary_threshold`)**
   - 根据光照条件调整
   - 过暗环境：降低阈值 (100-130)
   - 过亮环境：提高阈值 (180-220)

2. **灯条宽高比 (`light_params.min_ratio`, `light_params.max_ratio`)**
   - 根据实际灯条形状调整
   - 细长灯条：降低 `max_ratio` (0.3-0.5)
   - 粗短灯条：提高 `max_ratio` (0.6-0.8)

3. **装甲板中心距 (`armor_params.*_center_distance`)**
   - 根据摄像头距离和装甲板实际尺寸调整
   - 远距离：减小数值
   - 近距离：增大数值

4. **角度参数 (`light_params.max_angle`, `armor_params.max_angle`)**
   - 根据机器人姿态和装甲板倾斜程度调整
   - 倾斜较大：增大角度阈值

### 调试技巧

1. **开启调试模式**
   ```yaml
   debug_contours: true
   ```
   显示紫色轮廓矩形，帮助观察轮廓检测效果

2. **查看参数生效情况**
   ```bash
   ros2 param list /video_subscriber_node
   ros2 param get /video_subscriber_node light_params.max_ratio
   ```

3. **实时调整参数**
   ```bash
   ros2 param set /video_subscriber_node binary_threshold 180
   ```

## 常见问题

### Q1: 灯条检测不到？
- 检查 `binary_threshold` 是否合适
- 调整 `light_params.min_ratio` 和 `light_params.max_ratio`
- 确认 `detect_color` 设置正确

### Q2: 装甲板匹配失败？
- 调整 `armor_params` 中的距离参数
- 检查 `armor_params.max_angle` 是否过小
- 确认 `min_light_ratio` 设置合理

### Q3: 参数修改后没有生效？
- 确认参数名称拼写正确
- 重启节点使配置文件生效
- 检查yaml文件格式是否正确

## 示例配置

### 室内环境配置
```yaml
binary_threshold: 120
light_params:
  max_ratio: 0.7
  max_angle: 45.0
armor_params:
  max_angle: 40.0
```

### 室外强光环境配置
```yaml
binary_threshold: 200
exposure_gain: 0.5
light_params:
  max_ratio: 0.5
  max_angle: 35.0
```

### 远距离检测配置
```yaml
armor_params:
  min_small_center_distance: 0.6
  max_small_center_distance: 2.8
  min_large_center_distance: 2.8
  max_large_center_distance: 4.5
``` 