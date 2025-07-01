#!/bin/bash

# 设置ROS_DOMAIN_ID
export ROS_DOMAIN_ID=1

# 默认帧率
FRAME_RATE=150

# 默认视频
VIDEO_PATH="$HOME/code/src/video_trial_node/videos/rm_armor.mp4"

# 检查是否有自定义帧率参数
if [ "$1" != "" ]; then
  if [[ "$1" == *".mp4" ]]; then
    VIDEO_PATH="$1"
    echo "使用自定义视频: $VIDEO_PATH"
  else
    FRAME_RATE=$1
    echo "使用自定义帧率: $FRAME_RATE fps"
  fi
else
  echo "使用默认帧率: $FRAME_RATE fps"
  echo "使用默认视频: $VIDEO_PATH"
fi

# 检查是否有第二个参数（视频路径）
if [ "$2" != "" ]; then
  VIDEO_PATH="$2"
  echo "使用自定义视频: $VIDEO_PATH"
fi

# 启动视频发布节点
echo "启动视频发布节点..."
ros2 launch video_trial_node video_publisher.launch.py frame_rate:=$FRAME_RATE video_path:=$VIDEO_PATH

# 使用说明：
# ./run_video_publisher.sh          - 使用默认帧率(150fps)和默认视频(rm_armor.mp4)运行
# ./run_video_publisher.sh 200      - 使用200fps帧率和默认视频运行
# ./run_video_publisher.sh /path/to/video.mp4  - 使用默认帧率和指定视频运行
# ./run_video_publisher.sh 200 /path/to/video.mp4  - 使用200fps帧率和指定视频运行 