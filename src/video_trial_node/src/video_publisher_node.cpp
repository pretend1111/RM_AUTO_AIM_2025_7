#include "video_trial_node/video_publisher_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <chrono>

namespace video_trial
{

VideoPublisherNode::VideoPublisherNode(const rclcpp::NodeOptions & options)
: Node("video_publisher_node", options),
  last_fps_calc_time_(this->now()),
  actual_fps_(0.0)
{
  RCLCPP_INFO(this->get_logger(), "Starting VideoPublisherNode!");
  
  // 声明并获取参数
  video_path_ = this->declare_parameter<std::string>(
    "video_path",
    std::string(std::getenv("HOME")) + "/code/src/video_trial_node/videos/rm_armor.mp4");
  
  camera_name_ = this->declare_parameter<std::string>("camera_name", "video_camera");
  
  camera_info_url_ = this->declare_parameter<std::string>(
    "camera_info_url", "");
    
  frame_rate_ = this->declare_parameter<int>("frame_rate", 150);  // 默认提高到150帧
  
  repeat_ = this->declare_parameter<bool>("repeat", true);
  
  // 添加图像缩放参数以解决4K传输瓶颈
  // 移除写死的缩放因子，将在init_video_capture中根据视频尺寸智能计算
  scale_factor_ = this->declare_parameter<double>("scale_factor", 1.0);  // 默认不缩放，由程序智能判断
  
  // 检查视频文件是否存在
  if (!std::filesystem::exists(video_path_)) {
    RCLCPP_ERROR(this->get_logger(), "Video file does not exist: %s", video_path_.c_str());
    return;
  }
  
  RCLCPP_INFO(this->get_logger(), "Using video file: %s", video_path_.c_str());
  RCLCPP_INFO(this->get_logger(), "Target frame rate: %d fps", frame_rate_);
  RCLCPP_INFO(this->get_logger(), "Scale factor: %.2f", scale_factor_);
  
  // 创建相机信息管理器
  camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
  
  // 如果提供了相机信息URL，则加载相机信息
  if (!camera_info_url_.empty() && camera_info_manager_->validateURL(camera_info_url_)) {
    camera_info_manager_->loadCameraInfo(camera_info_url_);
    camera_info_msg_ = camera_info_manager_->getCameraInfo();
  } else {
    // 创建默认相机信息
    camera_info_msg_ = sensor_msgs::msg::CameraInfo();
    camera_info_msg_.width = 640;
    camera_info_msg_.height = 480;
    camera_info_msg_.k = {500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0};
    camera_info_msg_.distortion_model = "plumb_bob";
    camera_info_msg_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    RCLCPP_WARN(this->get_logger(), "No valid camera_info_url provided, using default camera parameters");
  }
  
  // 创建图像发布者
  bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", true);
  auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
  camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);
  
  // 初始化视频捕获
  if (!init_video_capture()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize video capture");
    return;
  }
  
  // 使用更快的定时器周期来实现高帧率
  // 将定时器周期设为1ms，在每个回调中控制实际发送频率
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(1),
    std::bind(&VideoPublisherNode::timer_callback, this));
  
  RCLCPP_INFO(this->get_logger(), "VideoPublisherNode started successfully");
}

VideoPublisherNode::~VideoPublisherNode()
{
  // 释放视频捕获资源
  if (video_capture_.isOpened()) {
    video_capture_.release();
  }
  RCLCPP_INFO(this->get_logger(), "VideoPublisherNode destroyed");
}

bool VideoPublisherNode::init_video_capture()
{
  // 打开视频文件
  video_capture_.open(video_path_);
  if (!video_capture_.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "Could not open video file: %s", video_path_.c_str());
    return false;
  }
  
  // 获取视频的总帧数
  total_frames_ = static_cast<int>(video_capture_.get(cv::CAP_PROP_FRAME_COUNT));
  
  // 获取视频的原始宽高
  int original_width = static_cast<int>(video_capture_.get(cv::CAP_PROP_FRAME_WIDTH));
  int original_height = static_cast<int>(video_capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
  
  // 智能缩放逻辑：如果横向分辨率大于1280，则压缩到1280宽度；否则保持原尺寸
  if (original_width > 1280) {
    // 计算缩放因子，使宽度变为1280
    scale_factor_ = 1280.0 / static_cast<double>(original_width);
    RCLCPP_INFO(this->get_logger(), "视频宽度(%d)大于1280，将压缩，缩放因子: %.3f", original_width, scale_factor_);
  } else {
    // 保持原尺寸
    scale_factor_ = 1.0;
    RCLCPP_INFO(this->get_logger(), "视频宽度(%d)不大于1280，保持原尺寸", original_width);
  }
  
  // 计算缩放后的尺寸
  scaled_width_ = static_cast<int>(original_width * scale_factor_);
  scaled_height_ = static_cast<int>(original_height * scale_factor_);
  
  // 更新相机信息的宽高（使用缩放后的尺寸）
  camera_info_msg_.width = scaled_width_;
  camera_info_msg_.height = scaled_height_;
  
  // 如果使用默认相机参数，则根据缩放后的视频尺寸调整
  if (camera_info_url_.empty()) {
    camera_info_msg_.k[0] = scaled_width_ * 0.8;  // fx
    camera_info_msg_.k[2] = scaled_width_ * 0.5;  // cx
    camera_info_msg_.k[4] = scaled_height_ * 0.8; // fy
    camera_info_msg_.k[5] = scaled_height_ * 0.5; // cy
  }
  
  RCLCPP_INFO(this->get_logger(), "Video properties: original=%dx%d, scaled=%dx%d, total_frames=%d",
    original_width, original_height, scaled_width_, scaled_height_, total_frames_);
    
  return true;
}

void VideoPublisherNode::timer_callback()
{
  static auto last_frame_time = this->now();
  auto current_time = this->now();
  
  // 控制帧率，计算每帧所需的时间间隔（纳秒）
  double frame_interval_ns = 1e9 / static_cast<double>(frame_rate_);
  
  // 检查是否应该发送下一帧
  double elapsed_ns = (current_time - last_frame_time).nanoseconds();
  if (elapsed_ns < frame_interval_ns) {
    return;  // 还没到发送下一帧的时间
  }
  
  // 检查视频捕获是否打开
  if (!video_capture_.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "Video capture is not opened");
    return;
  }
  
  cv::Mat frame;
  bool success = video_capture_.read(frame);
  
  // 如果读取失败或视频结束
  if (!success || frame.empty()) {
    // 如果设置了重复播放，则重置视频到开头
    if (repeat_) {
      RCLCPP_INFO(this->get_logger(), "End of video reached, restarting from the beginning");
      video_capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
      frame_count_ = 0;
      success = video_capture_.read(frame);
      
      if (!success || frame.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to restart video");
        return;
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "End of video reached");
      return;
    }
  }
  
  // 缩放图像以减少传输数据量
  cv::Mat scaled_frame;
  if (scale_factor_ != 1.0) {
    cv::resize(frame, scaled_frame, cv::Size(scaled_width_, scaled_height_), 0, 0, cv::INTER_LINEAR);
  } else {
    scaled_frame = frame;
  }
  
  // 修复红蓝通道颠倒问题：将BGR转换为RGB
  cv::Mat rgb_frame;
  cv::cvtColor(scaled_frame, rgb_frame, cv::COLOR_BGR2RGB);
  
  // 记录此帧发送时间
  last_frame_time = current_time;
  
  // 帧计数增加
  frame_count_++;
  
  // 计算实际帧率
  fps_frame_count_++;
  double elapsed_sec = (current_time - last_fps_calc_time_).seconds();
  if (elapsed_sec >= 1.0) {
    actual_fps_ = fps_frame_count_ / elapsed_sec;
    fps_frame_count_ = 0;
    last_fps_calc_time_ = current_time;
    RCLCPP_INFO(this->get_logger(), "Actual publishing rate: %.2f fps (data: %.1fMB/s)", 
                actual_fps_, actual_fps_ * scaled_width_ * scaled_height_ * 3 / 1024.0 / 1024.0);
  }
  
  // 创建ROS图像消息（使用RGB格式）
  auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", rgb_frame).toImageMsg();
  img_msg->header.stamp = current_time;
  img_msg->header.frame_id = "camera_optical_frame";
  
  // 更新相机信息时间戳
  camera_info_msg_.header = img_msg->header;
  
  // 发布图像和相机信息
  camera_pub_.publish(*img_msg, camera_info_msg_);
  
  // 每100帧打印一次状态信息
  if (frame_count_ % 100 == 0) {
    RCLCPP_INFO(this->get_logger(), "Published frame %d/%d", frame_count_, total_frames_);
  }
}

}  // namespace video_trial

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(video_trial::VideoPublisherNode) 