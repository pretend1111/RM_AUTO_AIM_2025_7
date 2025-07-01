#include "rm_armor_detect/video_subscriber_node.hpp"

namespace rm_armor_detect
{

VideoSubscriberNode::VideoSubscriberNode(const rclcpp::NodeOptions & options)
: Node("video_subscriber_node", options),
  new_frame_available_(false),
  running_(true),
  fps_frame_count_(0),
  current_fps_(0.0),
  last_fps_calc_time_(this->now())
{
  RCLCPP_INFO(this->get_logger(), "启动视频订阅节点！");
  
  // 声明并获取参数
  topic_name_ = this->declare_parameter<std::string>("topic_name", "image_raw");
  display_image_ = this->declare_parameter<bool>("display_image", true);
  window_name_ = this->declare_parameter<std::string>("window_name", "视频订阅显示");
  bool use_sensor_data_qos = this->declare_parameter<bool>("use_sensor_data_qos", true);
  
  // 添加模式切换参数
  int mode_value = this->declare_parameter<int>("detection_mode", 1);  // 0:传统视觉, 1:神经网络
  detection_mode_ = static_cast<DetectionMode>(mode_value);
  mode_name_ = detection_mode_ == DetectionMode::TRADITIONAL_VISION ? "传统视觉" : "神经网络识别";
  
  // 添加曝光增强参数
  exposure_gain_ = this->declare_parameter<double>("exposure_gain", 1.5);  // 默认不增强
  enable_exposure_enhancement_ = this->declare_parameter<bool>("enable_exposure_enhancement", true);  // 默认禁用
  
  // 传统视觉模式参数
  model_path_ = this->declare_parameter<std::string>("model_path", 
    std::string(std::getenv("HOME")) + "/code/src/rm_armor_detect/models/car/car.onnx");
  enable_car_detection_ = this->declare_parameter<bool>("enable_car_detection", true);
  detection_confidence_threshold_ = this->declare_parameter<double>("detection_confidence_threshold", 0.05);
  
  // 神经网络模式参数
  armor_model_path_ = this->declare_parameter<std::string>("armor_model_path",
    std::string(std::getenv("HOME")) + "/code/src/rm_armor_detect/models/armor/armor.onnx");
  armor_confidence_threshold_ = this->declare_parameter<double>("armor_confidence_threshold", 0.4);
  
  // 神经网络专用的曝光调整参数
  bool enable_nn_exposure_adjustment = this->declare_parameter<bool>("enable_nn_exposure_adjustment", false);
  double nn_exposure_adjustment_factor = this->declare_parameter<double>("nn_exposure_adjustment_factor", 1.0);
  
  // 添加车辆检测专用的降曝光参数
  bool enable_car_exposure_reduction = this->declare_parameter<bool>("enable_car_exposure_reduction", true);
  double car_exposure_reduction_factor = this->declare_parameter<double>("car_exposure_reduction_factor", 0.6);
  
  // 添加数字分类参数
  number_model_path_ = this->declare_parameter<std::string>("number_model_path",
    std::string(std::getenv("HOME")) + "/code/src/rm_armor_detect/models/number/mlp.onnx");
  number_label_path_ = this->declare_parameter<std::string>("number_label_path",
    std::string(std::getenv("HOME")) + "/code/src/rm_armor_detect/models/number/label.txt");
  enable_number_classification_ = this->declare_parameter<bool>("enable_number_classification", true);
  number_confidence_threshold_ = this->declare_parameter<double>("number_confidence_threshold", 0.7);
  
  // 灯条检测参数
  int binary_threshold = this->declare_parameter<int>("binary_threshold", 210);
  int detect_color = this->declare_parameter<int>("detect_color", 1);  // 0:蓝色, 1:红色
  bool debug_contours = this->declare_parameter<bool>("debug_contours", true);  // 是否显示轮廓调试信息
  
  // 设置QoS - 优化帧率接收性能
  auto qos = use_sensor_data_qos ?
    rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data) :
    rclcpp::QoS(rclcpp::KeepLast(10)).reliability(rclcpp::ReliabilityPolicy::Reliable);
  
  // 创建订阅者
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    topic_name_, qos, 
    std::bind(&VideoSubscriberNode::image_callback, this, std::placeholders::_1));
  
  RCLCPP_INFO(this->get_logger(), "订阅主题: %s (使用%s QoS设置, 队列深度: 100)", 
              topic_name_.c_str(), 
              use_sensor_data_qos ? "sensor_data" : "default");
  
  RCLCPP_INFO(this->get_logger(), "检测模式: %s", mode_name_.c_str());
  
  if (enable_exposure_enhancement_) {
    RCLCPP_INFO(this->get_logger(), "曝光增强已启用，增益: %.2f", exposure_gain_);
  } else {
    RCLCPP_INFO(this->get_logger(), "曝光增强已禁用");
  }
  
  // 根据模式初始化相应的检测器
  if (detection_mode_ == DetectionMode::TRADITIONAL_VISION) {
    // 初始化传统视觉模式检测器
    if (enable_car_detection_) {
    car_detector_ = std::make_unique<CarDetector>(model_path_);
    if (car_detector_->initialize()) {
      // 设置车辆检测专用的降曝光参数
      car_detector_->set_exposure_reduction(enable_car_exposure_reduction, car_exposure_reduction_factor);
      
      RCLCPP_INFO(this->get_logger(), "车辆检测器初始化成功，模型路径: %s", model_path_.c_str());
      RCLCPP_INFO(this->get_logger(), "检测置信度阈值: %.2f", detection_confidence_threshold_);
      RCLCPP_INFO(this->get_logger(), "车辆检测降曝光: %s, 因子: %.2f", 
                  enable_car_exposure_reduction ? "启用" : "禁用", car_exposure_reduction_factor);
      
      // 初始化灯条检测器
      lights_detector_ = std::make_unique<SimpleLightsDetector>(binary_threshold, detect_color);
      lights_detector_->set_debug_mode(debug_contours);
      RCLCPP_INFO(this->get_logger(), "灯条检测器初始化成功，二值化阈值: %d, 检测颜色: %s, 调试模式: %s", 
                  binary_threshold, detect_color == 0 ? "蓝色" : "红色", debug_contours ? "开启" : "关闭");
      
      // 初始化数字分类器
      if (enable_number_classification_) {
        try {
          lights_detector_->initNumberClassifier(number_model_path_, number_label_path_, number_confidence_threshold_);
          RCLCPP_INFO(this->get_logger(), "数字分类器初始化成功，模型路径: %s", number_model_path_.c_str());
          RCLCPP_INFO(this->get_logger(), "数字分类置信度阈值: %.2f", number_confidence_threshold_);
        } catch (const std::exception & e) {
          RCLCPP_ERROR(this->get_logger(), "数字分类器初始化失败: %s", e.what());
          enable_number_classification_ = false;
        }
      } else {
        RCLCPP_INFO(this->get_logger(), "数字分类已禁用");
      }
      } else {
        RCLCPP_ERROR(this->get_logger(), "车辆检测器初始化失败，将禁用检测功能");
        enable_car_detection_ = false;
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "传统视觉模式下车辆检测已禁用");
    }
  } else if (detection_mode_ == DetectionMode::NEURAL_NETWORK) {
    // 初始化神经网络模式检测器
    armor_detector_ = std::make_unique<OpenVinoArmorDetector>(armor_model_path_);
    if (armor_detector_->initialize()) {
      armor_detector_->set_confidence_threshold(static_cast<float>(armor_confidence_threshold_));
      
      // 设置神经网络专用的曝光调整参数
      armor_detector_->set_exposure_adjustment(enable_nn_exposure_adjustment, nn_exposure_adjustment_factor);
      
      RCLCPP_INFO(this->get_logger(), "OpenVINO装甲板检测器初始化成功，模型路径: %s", armor_model_path_.c_str());
      RCLCPP_INFO(this->get_logger(), "装甲板检测置信度阈值: %.2f", armor_confidence_threshold_);
      RCLCPP_INFO(this->get_logger(), "神经网络曝光调整: %s, 调整因子: %.2f", 
                  enable_nn_exposure_adjustment ? "启用" : "禁用", nn_exposure_adjustment_factor);
    } else {
      RCLCPP_ERROR(this->get_logger(), "OpenVINO装甲板检测器初始化失败");
      // 如果神经网络模式初始化失败，可以考虑回退到传统视觉模式
    }
  }
  
  // 如果需要显示图像，则创建显示窗口
  if (display_image_) {
    // 启动显示线程
    display_thread_ = std::thread(&VideoSubscriberNode::display_thread_func, this);
    RCLCPP_INFO(this->get_logger(), "启动图像显示线程");
  } else {
    RCLCPP_INFO(this->get_logger(), "图像显示已禁用，仅统计接收帧率");
  }
  
  RCLCPP_INFO(this->get_logger(), "视频订阅节点启动成功");
}

VideoSubscriberNode::~VideoSubscriberNode()
{
  // 停止运行并等待显示线程结束
  running_ = false;
  if (display_thread_.joinable()) {
    display_thread_.join();
  }
  
  // 关闭所有OpenCV窗口
  if (display_image_) {
    cv::destroyAllWindows();
  }
  
  RCLCPP_INFO(this->get_logger(), "视频订阅节点已销毁");
}

void VideoSubscriberNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    // 统计接收帧率（不管是否显示）
    fps_frame_count_++;
    auto current_time = this->now();
    double elapsed_sec = (current_time - last_fps_calc_time_).seconds();
    if (elapsed_sec >= 1.0) {
      current_fps_ = static_cast<double>(fps_frame_count_) / elapsed_sec;
      fps_frame_count_ = 0;
      last_fps_calc_time_ = current_time;
      RCLCPP_INFO(this->get_logger(), "接收帧率: %.2f fps (图像尺寸: %dx%d)", 
                  current_fps_, msg->width, msg->height);
    }
    
    // 将ROS图像消息转换为OpenCV格式并进行处理
    auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    cv::Mat frame = cv_ptr->image;
    
    // 在回调中完成所有图像处理，避免在display线程中阻塞
    cv::Mat processed_frame = frame.clone();
    
    // 曝光增强处理
    if (enable_exposure_enhancement_) {
      cv::Mat enhanced_frame;
      double beta = (exposure_gain_ - 1.0) * 50;
      processed_frame.convertTo(enhanced_frame, -1, exposure_gain_, beta);
      enhanced_frame.convertTo(processed_frame, CV_8UC3);
    }
    
    // 根据检测模式进行相应的处理
    if (detection_mode_ == DetectionMode::TRADITIONAL_VISION) {
      process_traditional_vision(processed_frame);
    } else if (detection_mode_ == DetectionMode::NEURAL_NETWORK) {
      process_neural_network(processed_frame);
    }
    
    // 如果需要显示，则更新显示缓冲区
    if (display_image_) {
      // 快速检查：如果显示缓冲区还有未处理的帧，跳过显示更新但继续处理
      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!new_frame_available_) {  // 只有当缓冲区空闲时才更新显示
          // 添加帧率显示文本
          std::string fps_text = "RX: " + std::to_string(static_cast<int>(std::round(current_fps_)));
          cv::putText(processed_frame, fps_text, cv::Point(20, 50), 
                      cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
          
          current_frame_ = std::move(processed_frame);  // 使用移动语义避免拷贝
          new_frame_available_ = true;
        }
      }
    }
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge异常: %s", e.what());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "图像处理异常: %s", e.what());
  }
}

void VideoSubscriberNode::display_thread_func()
{
  cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
  
  cv::Mat frame_copy;
  int display_fps_count = 0;
  auto last_display_fps_time = this->now();
  
  while (running_) {
    bool has_new_frame = false;
    
    // 尝试获取新帧（已经在image_callback中完成了所有处理）
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (new_frame_available_ && !current_frame_.empty()) {
        // 使用移动语义避免拷贝
        frame_copy = std::move(current_frame_);
        new_frame_available_ = false;
        has_new_frame = true;
      }
    }
    
    // 如果有新帧，则显示
    if (has_new_frame) {
      // 显示帧率计数
      display_fps_count++;
      
      // 计算显示帧率（每秒更新一次）
      auto current_time = this->now();
      double elapsed_sec = (current_time - last_display_fps_time).seconds();
      if (elapsed_sec >= 1.0) {
        double display_fps = static_cast<double>(display_fps_count) / elapsed_sec;
        display_fps_count = 0;
        last_display_fps_time = current_time;
        RCLCPP_INFO(this->get_logger(), "显示帧率: %.2f fps", display_fps);
      }
      
      // 显示图像（图像已经在image_callback中处理完毕）
      cv::imshow(window_name_, frame_copy);
      
      // 减少等待时间以支持更高帧率 - 从30ms改为1ms
      int key = cv::waitKey(1);
      if (key == 27) {  // ESC键
        RCLCPP_INFO(this->get_logger(), "用户按下ESC键，关闭显示");
        break;
      }
    } else {
      // 减少睡眠时间以提高响应速度 - 从10ms改为1ms
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  
  cv::destroyWindow(window_name_);
}

void VideoSubscriberNode::process_traditional_vision(cv::Mat & frame_copy)
{
  // 传统视觉模式：车辆检测 + 灯条检测 + 装甲板匹配
  std::vector<Detection> detections;  // 声明在外部以便灯光检测使用
  if (enable_car_detection_ && car_detector_) {
    try {
      // 执行车辆检测
      detections = car_detector_->detect(frame_copy, static_cast<float>(detection_confidence_threshold_));
      
      // 在图像上绘制检测结果
      car_detector_->draw_detections(frame_copy, detections);
      
      // 可选：输出检测结果到日志（避免过于频繁）
      static int detection_log_counter = 0;
      if (++detection_log_counter % 30 == 0) {  // 每30帧输出一次
        RCLCPP_DEBUG(this->get_logger(), "传统视觉模式检测到 %zu 个车辆", detections.size());
      }
      
      // 执行灯条和装甲板检测
      if (lights_detector_ && !detections.empty()) {
        auto light_results = lights_detector_->process_detections(frame_copy, detections);
        
        // 绘制灯条和装甲板结果
        for (const auto & result : light_results) {
          if (!result.lights.empty() || !result.armors.empty() || !result.debug_contour_rects.empty()) {
            // 绘制裁剪区域的边框
            cv::rectangle(frame_copy, result.car_box, cv::Scalar(0, 0, 255), 2);
            
            // 获取原始坐标偏移
            cv::Point offset(result.car_box.x, result.car_box.y);
            
            // 创建结果图像副本用于绘制灯条和装甲板
            cv::Mat result_viz = result.cropped_image.clone();
            
            // 绘制灯条和装甲板（不包括紫色轮廓）
            lights_detector_->drawResults(result_viz, result.lights, result.armors, {});
            
            // 先将结果复制回主图像
            result_viz.copyTo(frame_copy(result.car_box));
            
            // 然后在主图像上绘制紫色轮廓矩形（确保不被覆盖）
            if (lights_detector_->get_debug_mode() && !result.debug_contour_rects.empty()) {
              std::cout << "[调试] 在主图像上绘制" << result.debug_contour_rects.size() << "个紫色轮廓矩形" << std::endl;
              for (const auto & rect : result.debug_contour_rects) {
                cv::Point2f vertices[4];
                rect.points(vertices);
                
                // 将坐标从裁剪图像坐标系转换到主图像坐标系
                cv::Scalar purple_color(255, 0, 255);  // BGR格式的紫色
                for (int i = 0; i < 4; i++) {
                  cv::Point2f pt1 = vertices[i] + cv::Point2f(offset.x, offset.y);
                  cv::Point2f pt2 = vertices[(i + 1) % 4] + cv::Point2f(offset.x, offset.y);
                  cv::line(frame_copy, pt1, pt2, purple_color, 3);  // 增加线宽到3确保可见
                }
              }
            }
          }
        }
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                            "传统视觉模式检测失败: %s", e.what());
    }
  }
}

void VideoSubscriberNode::process_neural_network(cv::Mat & frame_copy)
{
  // 神经网络识别模式：直接检测装甲板
  if (armor_detector_) {
    try {
      // 执行装甲板检测
      auto armor_detections = armor_detector_->detect(frame_copy, static_cast<float>(armor_confidence_threshold_));
      
      // 在图像上绘制检测结果
      armor_detector_->draw_detections(frame_copy, armor_detections);
      
      // 可选：输出检测结果到日志（避免过于频繁）
      static int detection_log_counter = 0;
      if (++detection_log_counter % 30 == 0) {  // 每30帧输出一次
        RCLCPP_DEBUG(this->get_logger(), "神经网络模式检测到 %zu 个装甲板", armor_detections.size());
      }
      
      // 如果需要进行数字识别，可以在这里添加
      if (enable_number_classification_ && !armor_detections.empty()) {
        // TODO: 可以在这里添加数字识别功能，对检测到的装甲板进行数字分类
        // 需要将ArmorDetection转换为Armor结构，然后调用数字分类器
      }
      
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                            "神经网络模式检测失败: %s", e.what());
    }
  }
}

}  // namespace rm_armor_detect

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_armor_detect::VideoSubscriberNode) 