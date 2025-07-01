#pragma once

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include "rm_armor_detect/car_detector.hpp"
#include "rm_armor_detect/simple_lights_detector.hpp"
#include "rm_armor_detect/openvino_armor_detector.hpp"

namespace rm_armor_detect
{

// 检测模式枚举
enum class DetectionMode 
{
  TRADITIONAL_VISION = 0,  // 传统视觉模式（车辆检测+灯条检测+装甲板匹配）
  NEURAL_NETWORK = 1       // 神经网络识别模式（直接检测装甲板）
};

class VideoSubscriberNode : public rclcpp::Node
{
public:
  explicit VideoSubscriberNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~VideoSubscriberNode() override;

private:
  // 处理接收到的图像
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  
  // 显示线程函数
  void display_thread_func();
  
  // 传统视觉模式处理函数
  void process_traditional_vision(cv::Mat & frame_copy);
  
  // 神经网络模式处理函数
  void process_neural_network(cv::Mat & frame_copy);
  
  // 节点参数
  std::string topic_name_;
  bool display_image_;
  std::string window_name_;
  
  // 模式切换参数
  DetectionMode detection_mode_;
  std::string mode_name_;
  
  // 曝光增强参数
  double exposure_gain_;
  bool enable_exposure_enhancement_;
  
  // 传统视觉模式参数
  std::string model_path_;
  bool enable_car_detection_;
  double detection_confidence_threshold_;
  
  // 神经网络模式参数
  std::string armor_model_path_;
  double armor_confidence_threshold_;
  
  // 数字分类器参数
  std::string number_model_path_;
  std::string number_label_path_;
  bool enable_number_classification_;
  double number_confidence_threshold_;
  
  // 图像订阅者
  typename rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  
  // 传统视觉模式检测器
  std::unique_ptr<CarDetector> car_detector_;
  std::unique_ptr<SimpleLightsDetector> lights_detector_;
  
  // 神经网络模式检测器
  std::unique_ptr<OpenVinoArmorDetector> armor_detector_;
  
  // OpenCV图像处理
  cv::Mat current_frame_;
  std::mutex frame_mutex_;
  bool new_frame_available_;
  bool running_;
  
  // 显示线程
  std::thread display_thread_;
  
  // 帧率计算
  int fps_frame_count_;
  double current_fps_;
  rclcpp::Time last_fps_calc_time_;
};

}  // namespace rm_armor_detect