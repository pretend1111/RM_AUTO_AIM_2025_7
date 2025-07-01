#ifndef VIDEO_TRIAL_NODE__VIDEO_PUBLISHER_NODE_HPP_
#define VIDEO_TRIAL_NODE__VIDEO_PUBLISHER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace video_trial
{

class VideoPublisherNode : public rclcpp::Node
{
public:
  explicit VideoPublisherNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~VideoPublisherNode() override;

private:
  // 读取视频帧并发布
  void timer_callback();
  
  // 初始化视频捕获
  bool init_video_capture();
  
  // 节点参数
  std::string video_path_;
  std::string camera_name_;
  std::string camera_info_url_;
  int frame_rate_;
  bool repeat_;
  double scale_factor_;  // 图像缩放因子
  
  // OpenCV视频捕获
  cv::VideoCapture video_capture_;
  
  // 相机信息管理器
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;
  
  // 图像发布者
  image_transport::CameraPublisher camera_pub_;
  
  // 定时器
  rclcpp::TimerBase::SharedPtr timer_;
  
  // 处理循环播放
  int frame_count_{0};
  int total_frames_{0};
  
  // 缩放后的图像尺寸
  int scaled_width_{0};
  int scaled_height_{0};
  
  // 实际帧率计算
  rclcpp::Time last_fps_calc_time_;
  int fps_frame_count_{0};
  double actual_fps_;
};

}  // namespace video_trial

#endif  // VIDEO_TRIAL_NODE__VIDEO_PUBLISHER_NODE_HPP_ 