#ifndef RM_RUNE_DETECT__RUNE_DETECT_NODE_HPP_
#define RM_RUNE_DETECT__RUNE_DETECT_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>

#include "rm_rune_detect/openvino_detect.hpp"

namespace rm_rune_detect
{

class RuneDetectNode : public rclcpp::Node
{
public:
  explicit RuneDetectNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~RuneDetectNode() override;

private:
  // 图像回调处理函数
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  
  // 相机信息回调处理函数
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  
  // 初始化检测器
  void initialize_detector(bool enable_debug = false);
  
  // 异步处理线程函数
  void processing_thread_func();
  
  // 节点参数
  std::string model_path_;
  double exposure_factor_;
  
  // 检测器状态
  bool detector_initialized_{false};
  
  // 订阅器
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  
  // 发布器
  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::Publisher result_pub_;
  
  // 定时器（用于解决shared_from_this()在构造函数中不可用的问题）
  rclcpp::TimerBase::SharedPtr timer_;
  
  // 相机参数
  cv::Mat camera_matrix_;
  
  // 帧率计算
  rclcpp::Time last_time_;
  int frame_count_;
  
  // 异步处理相关
  std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::thread processing_thread_;
  bool stop_processing_ = false;
  
  // 性能统计
  std::chrono::steady_clock::time_point last_processing_time_;
  double processing_fps_ = 0.0;
  int processing_frame_count_ = 0;
  
  // 目标检测器实例
  std::unique_ptr<OpenvinoDetect> openvino_detector_;
};

}  // namespace rm_rune_detect

#endif  // RM_RUNE_DETECT__RUNE_DETECT_NODE_HPP_
