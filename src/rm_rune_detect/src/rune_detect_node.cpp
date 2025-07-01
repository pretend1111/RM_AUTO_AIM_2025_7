#include "rm_rune_detect/rune_detect_node.hpp"
#include "rm_rune_detect/preprocess.hpp"

#include <sensor_msgs/image_encodings.hpp>
#include <filesystem>
#include <openvino/openvino.hpp>

namespace rm_rune_detect
{

RuneDetectNode::RuneDetectNode(const rclcpp::NodeOptions & options)
: Node("rune_detect_node", options), 
  last_time_(this->now()),
  frame_count_(0)
{
  RCLCPP_DEBUG(this->get_logger(), "启动符文识别节点");
  
  // 初始化相机内参矩阵
  camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
  
  // 获取模型路径参数
  model_path_ = this->declare_parameter<std::string>("model_path", 
    std::string(std::getenv("HOME")) + "/code/src/rm_rune_detect/models/rune_detect_320.onnx");
    
  // 获取预处理参数
  exposure_factor_ = this->declare_parameter<double>("exposure_factor", 0.8);
  
  // 获取调试模式参数
  bool enable_debug = this->declare_parameter<bool>("enable_debug", true);
  RCLCPP_DEBUG(this->get_logger(), "调试模式: %s", enable_debug ? "开启" : "关闭");
  
  // 获取推理引擎参数（仅保留OpenVINO）
  std::string engine_type = this->declare_parameter<std::string>("inference_engine", "openvino");
  if (engine_type != "openvino") {
    RCLCPP_WARN(this->get_logger(), "仅支持OpenVINO推理引擎，将使用OpenVINO引擎");
  } else {
    RCLCPP_DEBUG(this->get_logger(), "使用OpenVINO推理引擎");
  }
    
  // 检查模型文件是否存在
  if (!std::filesystem::exists(model_path_)) {
    RCLCPP_ERROR(this->get_logger(), "模型文件不存在: %s", model_path_.c_str());
    RCLCPP_ERROR(this->get_logger(), "请检查模型路径参数是否正确");
  } else {
    RCLCPP_DEBUG(this->get_logger(), "找到模型文件: %s", model_path_.c_str());
    RCLCPP_DEBUG(this->get_logger(), "模型文件大小: %zu bytes", std::filesystem::file_size(model_path_));
    RCLCPP_DEBUG(this->get_logger(), "开始初始化检测器...");
    try {
      initialize_detector(enable_debug);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "初始化检测器时发生异常: %s", e.what());
    }
  }
  
  // 订阅相机内参
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_info", rclcpp::SensorDataQoS(),
    std::bind(&RuneDetectNode::camera_info_callback, this, std::placeholders::_1));
    
  // 订阅图像消息
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&RuneDetectNode::image_callback, this, std::placeholders::_1));
    
  // 创建一次性定时器，延迟初始化图像传输（解决shared_from_this问题）
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    [this]() {
      // 初始化图像传输
      it_ = std::make_shared<image_transport::ImageTransport>(this->shared_from_this());
      result_pub_ = it_->advertise("rune/result_image", 10);
      RCLCPP_DEBUG(this->get_logger(), "图像传输初始化完成");
      // 初始化完成后取消定时器
      timer_->cancel();
    }
  );
  
  RCLCPP_DEBUG(this->get_logger(), "曝光调整因子: %.2f", exposure_factor_);
  RCLCPP_DEBUG(this->get_logger(), "目标检测器初始化状态: %s", detector_initialized_ ? "成功" : "失败");
  
  // 初始化性能统计
  last_processing_time_ = std::chrono::steady_clock::now();
  
  // 启动异步处理线程
  processing_thread_ = std::thread(&RuneDetectNode::processing_thread_func, this);
  RCLCPP_INFO(this->get_logger(), "异步处理线程已启动");
}

RuneDetectNode::~RuneDetectNode()
{
  // 停止处理线程
  stop_processing_ = true;
  queue_cv_.notify_all();
  
  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }
  
  RCLCPP_DEBUG(this->get_logger(), "符文识别节点已关闭");
}

void RuneDetectNode::initialize_detector(bool enable_debug)
{
  try {
    // 检查模型文件是否存在
    if (!std::filesystem::exists(model_path_)) {
      RCLCPP_ERROR(this->get_logger(), "模型文件不存在，路径: %s", model_path_.c_str());
      detector_initialized_ = false;
      return;
    }
    
    RCLCPP_DEBUG(this->get_logger(), "模型文件大小: %zu bytes", 
               std::filesystem::file_size(model_path_));
               
    // 使用纯OpenVINO推理引擎
    openvino_detector_ = std::make_unique<OpenvinoDetect>(
      model_path_,       // 模型路径
      cv::Size(320, 320), // 输入尺寸
      0.5,               // 置信度阈值
      0.5,               // NMS阈值
      enable_debug       // 是否启用调试输出
    );
    detector_initialized_ = true;
    RCLCPP_DEBUG(this->get_logger(), "OpenVINO目标检测器初始化成功");
  } catch (const ov::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "OpenVINO异常: %s", e.what());
    detector_initialized_ = false;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "初始化检测器失败: %s", e.what());
    detector_initialized_ = false;
  }
}

void RuneDetectNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  RCLCPP_DEBUG(this->get_logger(), "接收到相机内参信息!");
  RCLCPP_DEBUG(this->get_logger(), "K矩阵: [%f, %f, %f, %f, %f, %f, %f, %f, %f]",
    msg->k[0], msg->k[1], msg->k[2],
    msg->k[3], msg->k[4], msg->k[5],
    msg->k[6], msg->k[7], msg->k[8]);
    
  camera_matrix_.at<double>(0, 0) = msg->k[0];  // fx
  camera_matrix_.at<double>(0, 2) = msg->k[2];  // cx
  camera_matrix_.at<double>(1, 1) = msg->k[4];  // fy
  camera_matrix_.at<double>(1, 2) = msg->k[5];  // cy
  camera_matrix_.at<double>(2, 2) = 1.0;
  
  // 收到相机内参后取消订阅
  camera_info_sub_.reset();
}

void RuneDetectNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // 快速帧率统计
  auto current_time = this->now();
  frame_count_++;
  double elapsed = (current_time - last_time_).seconds();
  
  if (elapsed >= 1.0) {
    double fps = frame_count_ / elapsed;
    RCLCPP_INFO(get_logger(), "[收帧FPS] 接收帧率: %.2f | [处理FPS] 处理帧率: %.2f", fps, processing_fps_);
    frame_count_ = 0;
    last_time_ = current_time;
  }
  
  // 将图像加入处理队列（非阻塞）
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // 限制队列大小，防止内存爆炸，保持最新的图像
    while (image_queue_.size() >= 3) {  // 保留最多3帧
      image_queue_.pop();
    }
    
    image_queue_.push(msg);
  }
  queue_cv_.notify_one();
}

// 异步处理线程函数
void RuneDetectNode::processing_thread_func()
{
  auto last_processing_time = std::chrono::steady_clock::now();
  int processing_frame_count = 0;
  
  while (!stop_processing_) {
    sensor_msgs::msg::Image::SharedPtr msg;
    
    // 等待新图像
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return !image_queue_.empty() || stop_processing_; });
      
      if (stop_processing_) break;
      
      msg = image_queue_.front();
      image_queue_.pop();
    }
    
    // 处理图像
    cv::Mat image;
    try {
      // BGR转换（更快的转换方式）
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      image = cv_ptr->image;
      
      if (image.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "接收到空图像");
        continue;
      }
      
      // 只在必要时执行曝光调整
      if (std::abs(exposure_factor_ - 1.0) > 0.05) {
        image = ImagePreprocessor::adjustExposure(image, exposure_factor_);
      }
      
      // 执行目标检测
      if (detector_initialized_) {
        auto detections = openvino_detector_->detect(image);
        
        if (!detections.empty()) {
          RCLCPP_DEBUG(get_logger(), "检测到 %zu 个目标", detections.size());
          
          // 打印检测详情（仅在调试模式）
          for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            if (det.class_id >= 0 && det.class_id <= 1) {
              RCLCPP_DEBUG(get_logger(), "目标 #%zu: 类别=%d, 置信度=%.4f, 位置=[%d,%d,%d,%d]",
                        i, det.class_id, det.confidence,
                        det.box.x, det.box.y, det.box.width, det.box.height);
            }
          }
          
          openvino_detector_->drawDetections(image, detections);
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "检测器未初始化");
      }
      
      // 只在调试模式下显示图像
      #ifdef DEBUG_DISPLAY
      cv::imshow("符文识别", image);
      cv::waitKey(1);
      #endif
      
      // 发布结果图像（只在有订阅者时）
      if (it_ && result_pub_.getNumSubscribers() > 0) {
        sensor_msgs::msg::Image::SharedPtr result_msg = 
          cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
        result_pub_.publish(result_msg);
      }
      
      // 统计处理帧率
      processing_frame_count++;
      auto current_time = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_processing_time).count();
      
      if (elapsed >= 1000) {  // 每秒统计一次
        processing_fps_ = processing_frame_count * 1000.0 / elapsed;
        processing_frame_count = 0;
        last_processing_time = current_time;
      }
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "处理异常: %s", e.what());
    }
  }
}

}  // namespace rm_rune_detect

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_rune_detect::RuneDetectNode)
