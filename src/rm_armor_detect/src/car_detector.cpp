#include "rm_armor_detect/car_detector.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace rm_armor_detect
{

CarDetector::CarDetector(const std::string & model_path)
: model_path_(model_path), is_initialized_(false),
  enable_exposure_reduction_(true), exposure_reduction_factor_(0.7),  // 默认启用降曝光，因子0.7
  no_detection_count_(0)  // 初始化无检测计数器
{
}

bool CarDetector::initialize()
{
  try {
    // 读取模型
    auto model = core_.read_model(model_path_);
    
    // 直接编译模型，不预先重塑
    compiled_model_ = core_.compile_model(model, "CPU");
    
    // 创建推理请求
    infer_request_ = compiled_model_.create_infer_request();
    
    // 手动设置固定的输入尺寸 (YOLOv8n通常使用640x640)
    input_height_ = 480;
    input_width_ = 480;
    input_shape_ = {1, 3, input_height_, input_width_};  // NCHW格式
    
    is_initialized_ = true;
    std::cout << "车辆检测器初始化成功，输入尺寸: " << input_width_ << "x" << input_height_ << std::endl;
    std::cout << "车辆检测降曝光: " << (enable_exposure_reduction_ ? "启用" : "禁用") 
              << ", 降曝光因子: " << exposure_reduction_factor_ << std::endl;
    return true;
    
  } catch (const std::exception & e) {
    std::cerr << "初始化车辆检测器失败: " << e.what() << std::endl;
    return false;
  }
}

void CarDetector::set_exposure_reduction(bool enable, double factor)
{
  enable_exposure_reduction_ = enable;
  exposure_reduction_factor_ = std::clamp(factor, 0.1, 1.0);  // 限制在0.1-1.0范围内
  std::cout << "车辆检测降曝光设置更新: " << (enable_exposure_reduction_ ? "启用" : "禁用") 
            << ", 降曝光因子: " << exposure_reduction_factor_ << std::endl;
}

cv::Mat CarDetector::apply_exposure_reduction(const cv::Mat & image)
{
  if (!enable_exposure_reduction_) {
    return image.clone();
  }
  
  cv::Mat reduced_image;
  // 使用简单的乘法降低曝光，这比convertTo更直接
  image.convertTo(reduced_image, -1, exposure_reduction_factor_, 0);
  
  return reduced_image;
}

cv::Mat CarDetector::preprocess(const cv::Mat & image)
{
  // 首先应用车辆检测专用的降曝光处理
  cv::Mat exposure_reduced = apply_exposure_reduction(image);
  
  cv::Mat resized_image;
  cv::resize(exposure_reduced, resized_image, cv::Size(input_width_, input_height_));
  
  // 计算缩放比例
  scale_x_ = static_cast<float>(image.cols) / static_cast<float>(input_width_);
  scale_y_ = static_cast<float>(image.rows) / static_cast<float>(input_height_);
  
  // 检查输入图像通道数，判断是BGR还是RGB
  cv::Mat rgb_image;
  if (image.channels() == 3) {
    // 假设输入已经是RGB格式（来自video_trial_node），直接使用
    // 不再进行BGR到RGB的转换，因为video_trial_node现在发送RGB格式
    rgb_image = resized_image.clone();
  } else {
    std::cerr << "警告：不支持的图像通道数: " << image.channels() << std::endl;
    rgb_image = resized_image.clone();
  }
  
  // 归一化到[0, 1]
  rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0);
  
  return rgb_image;
}

std::vector<Detection> CarDetector::detect(const cv::Mat & image, float conf_threshold)
{
  if (!is_initialized_) {
    std::cerr << "检测器未初始化" << std::endl;
    return {};
  }
  
  // 预处理
  cv::Mat preprocessed = preprocess(image);
  
  // 创建输入张量
  auto input_port = compiled_model_.input();
  ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), input_shape_);
  
  // 将图像数据复制到张量中 (NCHW格式)
  float* input_data = input_tensor.data<float>();
  std::vector<cv::Mat> channels(3);
  cv::split(preprocessed, channels);
  
  for (int c = 0; c < 3; ++c) {
    std::memcpy(input_data + c * input_height_ * input_width_, 
                channels[c].data, 
                input_height_ * input_width_ * sizeof(float));
  }
  
  // 设置输入张量
  infer_request_.set_input_tensor(input_tensor);
  
  // 执行推理
  infer_request_.infer();
  
  // 获取输出
  auto output_tensor = infer_request_.get_output_tensor();
  
  // 后处理（使用优化的参数：更严格的NMS）
  std::vector<Detection> current_detections = postprocess(output_tensor, conf_threshold, 0.2f);  // 进一步降低NMS阈值
  
  // 处理无检测情况的逻辑
  if (current_detections.empty()) {
    // 当前帧没有检测到车辆
    no_detection_count_++;
    
    // 如果连续无检测帧数未超过限制且有上一帧的检测结果，使用扩大的上一帧检测框
    if (no_detection_count_ <= MAX_NO_DETECTION_FRAMES && !last_detections_.empty()) {
      std::cout << "车辆检测失败，使用上一帧检测框(扩大30%)，连续无检测帧数: " 
                << no_detection_count_ << "/" << MAX_NO_DETECTION_FRAMES << std::endl;
      
      // 扩大上一帧的检测框
      std::vector<Detection> expanded_detections;
      for (const auto& last_det : last_detections_) {
        Detection expanded_det = last_det;
        expanded_det.box = expand_box(last_det.box, 1.3f, cv::Size(image.cols, image.rows));  // 扩大30%
        expanded_det.confidence *= 0.8f;  // 降低置信度以标识这是基于历史的检测
        expanded_detections.push_back(expanded_det);
      }
      return expanded_detections;
    } else if (no_detection_count_ > MAX_NO_DETECTION_FRAMES) {
      std::cout << "连续" << MAX_NO_DETECTION_FRAMES << "帧无车辆检测，停止使用历史检测框" << std::endl;
    }
  } else {
    // 当前帧检测到车辆，重置计数器并保存检测结果
    no_detection_count_ = 0;
    last_detections_ = current_detections;
  }
  
  return current_detections;
}

std::vector<Detection> CarDetector::postprocess(const ov::Tensor & output_tensor, 
                                               float conf_threshold, 
                                               float nms_threshold)
{
  const float* output_data = output_tensor.data<const float>();
  auto output_shape = output_tensor.get_shape();
  
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;
  
  // YOLOv8单类别模型输出格式: [1, 5, 8400]
  // 5个值为: [x_center, y_center, width, height, confidence]
  if (output_shape.size() != 3 || output_shape[1] != 5) {
    std::cerr << "不支持的输出格式，期望 [1, 5, N]" << std::endl;
    return {};
  }
  
  size_t num_detections = output_shape[2];
  
  // 统计检测结果
  int valid_detections = 0;
  int confidence_filtered = 0;
  float max_confidence = 0.0f;
  
  // 处理每个检测结果
  for (size_t i = 0; i < num_detections; ++i) {
    // YOLOv8输出是转置的：[x, y, w, h, conf] 按列存储
    float x_center = output_data[i];                    // 第1行
    float y_center = output_data[num_detections + i];   // 第2行
    float width = output_data[2 * num_detections + i];  // 第3行
    float height = output_data[3 * num_detections + i]; // 第4行
    float confidence = output_data[4 * num_detections + i]; // 第5行
    
    // 统计最大置信度
    max_confidence = std::max(max_confidence, confidence);
    
    // 记录所有置信度大于0.01的检测
    if (confidence > 0.01f) {
      valid_detections++;
    }
    
    // 置信度阈值过滤
    if (confidence < conf_threshold) {
      if (confidence > 0.01f) confidence_filtered++;
      continue;
    }
    
    // 转换为真实图像坐标
    int x = static_cast<int>((x_center - width / 2) * scale_x_);
    int y = static_cast<int>((y_center - height / 2) * scale_y_);
    int w = static_cast<int>(width * scale_x_);
    int h = static_cast<int>(height * scale_y_);
    
    // 更严格的边界检查和尺寸合理性检查
    if (x >= 0 && y >= 0 && w >= 20 && h >= 20 &&      // 最小尺寸20x20
        w <= 600 && h <= 400 &&                        // 最大尺寸600x400
        x + w <= static_cast<int>(scale_x_ * input_width_) &&  // 不超出图像边界
        y + h <= static_cast<int>(scale_y_ * input_height_)) {
      boxes.emplace_back(x, y, w, h);
      scores.emplace_back(confidence);
      class_ids.emplace_back(0);  // 单类别模型，类别ID为0
    }
  }
  
  // 打印检测统计（每120次调用打印一次）
  static int debug_counter = 0;
  if (++debug_counter % 120 == 0) {
    std::cout << "检测统计: 有效=" << valid_detections 
              << ", 过滤=" << confidence_filtered 
              << ", 候选=" << boxes.size() 
              << ", 最高置信度=" << std::fixed << std::setprecision(3) << max_confidence << std::endl;
  }
  
  // 多级NMS策略：先按置信度排序，再进行更严格的NMS
  std::vector<std::pair<float, int>> score_indices;
  for (size_t i = 0; i < scores.size(); ++i) {
    score_indices.emplace_back(scores[i], static_cast<int>(i));
  }
  std::sort(score_indices.rbegin(), score_indices.rend());  // 按置信度降序排列
  
  // 更严格的NMS：使用更低的阈值
  std::vector<int> indices = nms(boxes, scores, nms_threshold);
  
  // 进一步过滤：移除置信度过低的结果（动态阈值）
  std::vector<Detection> detections;
  float adaptive_threshold = std::max(conf_threshold, max_confidence * 0.3f);  // 自适应阈值
  
  for (int idx : indices) {
    if (scores[idx] >= adaptive_threshold) {  // 使用自适应阈值
      Detection det;
      det.box = boxes[idx];
      det.confidence = scores[idx];
      det.class_name = "car";  // 车辆检测
      detections.emplace_back(det);
    }
  }
  
  return detections;
}

std::vector<int> CarDetector::nms(const std::vector<cv::Rect> & boxes, 
                                 const std::vector<float> & scores, 
                                 float nms_threshold)
{
  std::vector<int> indices;
  
  // 如果没有检测框，直接返回
  if (boxes.empty()) {
    return indices;
  }
  
  // 使用OpenCV的NMS，但先做预处理
  std::vector<cv::Rect> filtered_boxes;
  std::vector<float> filtered_scores;
  std::vector<int> original_indices;
  
  // 预过滤：移除面积过小或过大的框
  for (size_t i = 0; i < boxes.size(); ++i) {
    int area = boxes[i].width * boxes[i].height;
    if (area >= 400 && area <= 240000) {  // 面积在400-240000像素之间
      filtered_boxes.push_back(boxes[i]);
      filtered_scores.push_back(scores[i]);
      original_indices.push_back(static_cast<int>(i));
    }
  }
  
  if (filtered_boxes.empty()) {
    return indices;
  }
  
  // 执行NMS
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(filtered_boxes, filtered_scores, 0.0f, nms_threshold, nms_indices);
  
  // 映射回原始索引
  for (int nms_idx : nms_indices) {
    if (nms_idx < static_cast<int>(original_indices.size())) {
      indices.push_back(original_indices[nms_idx]);
    }
  }
  
  // 进一步过滤：移除置信度差异太大的重叠框
  if (indices.size() > 1) {
    std::vector<int> final_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      bool keep = true;
      for (size_t j = i + 1; j < indices.size(); ++j) {
        // 计算IoU
        cv::Rect& box1 = const_cast<cv::Rect&>(boxes[indices[i]]);
        cv::Rect& box2 = const_cast<cv::Rect&>(boxes[indices[j]]);
        
        cv::Rect intersection = box1 & box2;
        if (intersection.area() > 0) {
          float iou = static_cast<float>(intersection.area()) / 
                      static_cast<float>((box1 | box2).area());
          
          // 如果IoU > 0.1且置信度差异较大，保留置信度更高的
          if (iou > 0.1f && std::abs(scores[indices[i]] - scores[indices[j]]) > 0.02f) {
            if (scores[indices[i]] < scores[indices[j]]) {
              keep = false;
              break;
            }
          }
        }
      }
      if (keep) {
        final_indices.push_back(indices[i]);
      }
    }
    return final_indices;
  }
  
  return indices;
}

cv::Rect CarDetector::expand_box(const cv::Rect& box, float scale, const cv::Size& image_size)
{
  // 计算扩大后的尺寸
  int expanded_width = static_cast<int>(box.width * scale);
  int expanded_height = static_cast<int>(box.height * scale);
  
  // 计算扩大后的左上角坐标（保持中心点不变）
  int expanded_x = box.x - (expanded_width - box.width) / 2;
  int expanded_y = box.y - (expanded_height - box.height) / 2;
  
  // 确保不超出图像边界
  expanded_x = std::max(0, expanded_x);
  expanded_y = std::max(0, expanded_y);
  expanded_width = std::min(expanded_width, image_size.width - expanded_x);
  expanded_height = std::min(expanded_height, image_size.height - expanded_y);
  
  return cv::Rect(expanded_x, expanded_y, expanded_width, expanded_height);
}

void CarDetector::draw_detections(cv::Mat & image, const std::vector<Detection> & detections)
{
  for (const auto & detection : detections) {
    // 绘制边界框
    cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);
    
    // 准备标签文本
    std::string label = detection.class_name + ": " + 
                       std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    
    // 计算文本尺寸
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    
    // 绘制文本背景
    cv::Point text_origin(detection.box.x, detection.box.y - 5);
    cv::rectangle(image, 
                  cv::Point(text_origin.x, text_origin.y - text_size.height - baseline),
                  cv::Point(text_origin.x + text_size.width, text_origin.y + baseline),
                  cv::Scalar(0, 255, 0), cv::FILLED);
    
    // 绘制标签文本
    cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(0, 0, 0), 1);
  }
}

}  // namespace rm_armor_detect 