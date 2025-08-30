# RM装甲板检测参数配置系统

## 📁 文件结构
```
rm_armor_detect/
├── config/
│   ├── armor_detect_params.yaml    # 默认配置文件
│   ├── presets/                    # 预设配置文件夹
│   │   ├── indoor_config.yaml      # 室内环境配置
│   │   ├── outdoor_config.yaml     # 室外环境配置
│   │   └── debug_config.yaml       # 调试模式配置
│   └── README.md                   # 详细配置指南
└── launch/
    └── armor_detect.launch.py      # 启动文件
```

## 🚀 快速开始

### 使用默认配置
```bash
ros2 launch rm_armor_detect armor_detect.launch.py
```

### 使用预设配置
```bash
# 室内环境
ros2 launch rm_armor_detect armor_detect.launch.py \
    params_file:=src/rm_armor_detect/config/presets/indoor_config.yaml

# 室外环境
ros2 launch rm_armor_detect armor_detect.launch.py \
    params_file:=src/rm_armor_detect/config/presets/outdoor_config.yaml

# 调试模式
ros2 launch rm_armor_detect armor_detect.launch.py \
    params_file:=src/rm_armor_detect/config/presets/debug_config.yaml
```

## ⚙️ 核心参数速查

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `binary_threshold` | 100-220 | 二值化阈值 |
| `light_params.max_ratio` | 0.3-0.8 | 灯条宽高比上限 |
| `light_params.max_angle` | 30-60° | 灯条角度上限 |
| `armor_params.max_small_center_distance` | 2.0-4.0 | 小装甲板距离上限 |

## 🔧 实时调参
```bash
# 查看参数
ros2 param get /video_subscriber_node binary_threshold

# 修改参数
ros2 param set /video_subscriber_node binary_threshold 180
ros2 param set /video_subscriber_node light_params.max_angle 45.0
```

详细说明请参考 README.md 文件。 