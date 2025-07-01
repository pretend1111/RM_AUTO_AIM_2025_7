#include "rm_armor_detect/openvino_armor_detector.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace rm_armor_detect
{

OpenVinoArmorDetector::OpenVinoArmorDetector(const std::string & model_path)
: model_path_(model_path), is_initialized_(false), confidence_threshold_(0.4f),
  enable_exposure_adjustment_(false), exposure_adjustment_factor_(1.0)
{
}

bool OpenVinoArmorDetector::initialize()
{
  try {
    // 读取模型
    auto model = core_.read_model(model_path_);
    
    // 编译模型
    compiled_model_ = core_.compile_model(model, "CPU");
    
    // 创建推理请求
    infer_request_ = compiled_model_.create_infer_request();
    
    // 设置固定的输入尺寸 (根据模型规格: 480x480)
    input_height_ = 480;
    input_width_ = 480;
    input_shape_ = {1, 3, input_height_, input_width_};  // NCHW格式
    
    is_initialized_ = true;
    std::cout << "OpenVINO装甲板检测器初始化成功，输入尺寸: " << input_width_ << "x" << input_height_ << std::endl;
    std::cout << "模型路径: " << model_path_ << std::endl;
    std::cout << "默认置信度阈值: " << confidence_threshold_ << std::endl;
    std::cout << "神经网络曝光调整: " << (enable_exposure_adjustment_ ? "启用" : "禁用") 
              << ", 调整因子: " << exposure_adjustment_factor_ << std::endl;
    return true;
    
  } catch (const std::exception & e) {
    std::cerr << "初始化OpenVINO装甲板检测器失败: " << e.what() << std::endl;
    return false;
  }
}

void OpenVinoArmorDetector::set_exposure_adjustment(bool enable, double factor)
{
  enable_exposure_adjustment_ = enable;
  exposure_adjustment_factor_ = std::clamp(factor, 0.1, 3.0);  // 限制在0.1-3.0范围内
  std::cout << "神经网络曝光调整设置更新: " << (enable_exposure_adjustment_ ? "启用" : "禁用") 
            << ", 调整因子: " << exposure_adjustment_factor_ << std::endl;
}

cv::Mat OpenVinoArmorDetector::apply_exposure_adjustment(const cv::Mat & image)
{
  if (!enable_exposure_adjustment_) {
    return image.clone();
  }
  
  cv::Mat adjusted_image;
  // 使用convertTo进行曝光调整
  // alpha参数控制增益，beta参数控制偏移
  double alpha = exposure_adjustment_factor_;
  double beta = 0;  // 不添加偏移
  
  image.convertTo(adjusted_image, -1, alpha, beta);
  
  return adjusted_image;
}

cv::Mat OpenVinoArmorDetector::preprocess(const cv::Mat & image)
{
  // 首先应用神经网络专用的曝光调整处理
  cv::Mat exposure_adjusted = apply_exposure_adjustment(image);
  
  cv::Mat resized_image;
  cv::resize(exposure_adjusted, resized_image, cv::Size(input_width_, input_height_));
  
  // 计算缩放比例
  scale_x_ = static_cast<float>(image.cols) / static_cast<float>(input_width_);
  scale_y_ = static_cast<float>(image.rows) / static_cast<float>(input_height_);
  
  // 转换为RGB格式并归一化到[0, 1]
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
  rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0);
  
  return rgb_image;
}

std::vector<ArmorDetection> OpenVinoArmorDetector::detect(const cv::Mat & image, float conf_threshold)
{
  if (!is_initialized_) {
    std::cerr << "OpenVINO装甲板检测器未初始化" << std::endl;
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
  
  // 后处理
  std::vector<ArmorDetection> detections = postprocess(output_tensor, image, conf_threshold, 0.5f);
  
  return detections;
}

std::vector<ArmorDetection> OpenVinoArmorDetector::postprocess(
  const ov::Tensor & output_tensor, 
  const cv::Mat & original_image,
  float conf_threshold, 
  float nms_threshold)
{
  const float* output_data = output_tensor.data<const float>();
  auto output_shape = output_tensor.get_shape();
  
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<Color> colors;
  std::vector<std::vector<cv::Point2f>> keypoints_list;
  
  // 模型输出格式: [1, 14175, 15]
  // 15个值: [0-4]位置信息, [5-12]关键点坐标, [13-14]类别概率
  if (output_shape.size() != 3 || output_shape[2] != 15) {
    std::cerr << "不支持的输出格式，期望 [1, N, 15]，实际: [" 
              << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
    return {};
  }
  
  size_t num_detections = output_shape[1];  // 14175
  
  // 处理每个检测结果
  for (size_t i = 0; i < num_detections; ++i) {
    // 获取位置信息 (相对于480x480图像)
    float cx = output_data[i * 15 + 0];
    float cy = output_data[i * 15 + 1];
    float w = output_data[i * 15 + 2];
    float h = output_data[i * 15 + 3];
    float confidence = output_data[i * 15 + 4];
    
    // 获取关键点坐标 (相对于480x480图像)
    std::vector<cv::Point2f> keypoints(4);
    for (int j = 0; j < 4; ++j) {
      keypoints[j].x = output_data[i * 15 + 5 + j * 2];
      keypoints[j].y = output_data[i * 15 + 6 + j * 2];
    }
    
    // 获取类别概率
    float prob_blue = output_data[i * 15 + 13];
    float prob_red = output_data[i * 15 + 14];
    
    // 置信度阈值过滤
    if (confidence < conf_threshold) {
      continue;
    }
    
    // 确定颜色类别
    Color color = prob_red > prob_blue ? Color::RED : Color::BLUE;
    
    // 转换为真实图像坐标
    int x = static_cast<int>((cx - w / 2) * scale_x_);
    int y = static_cast<int>((cy - h / 2) * scale_y_);
    int box_w = static_cast<int>(w * scale_x_);
    int box_h = static_cast<int>(h * scale_y_);
    
    // 边界检查
    if (x >= 0 && y >= 0 && box_w > 0 && box_h > 0 &&
        x + box_w <= original_image.cols && y + box_h <= original_image.rows) {
      
      boxes.emplace_back(x, y, box_w, box_h);
      scores.emplace_back(confidence);
      colors.emplace_back(color);
      
      // 转换关键点坐标到真实图像坐标
      std::vector<cv::Point2f> real_keypoints(4);
      for (int j = 0; j < 4; ++j) {
        real_keypoints[j].x = keypoints[j].x * scale_x_;
        real_keypoints[j].y = keypoints[j].y * scale_y_;
      }
      keypoints_list.emplace_back(real_keypoints);
    }
  }
  
  // 执行NMS
  std::vector<int> indices = nms(boxes, scores, nms_threshold);
  
  // 构建最终检测结果
  std::vector<ArmorDetection> detections;
  for (int idx : indices) {
    ArmorDetection detection;
    detection.box = boxes[idx];
    detection.confidence = scores[idx];
    detection.color = colors[idx];
    detection.keypoints = keypoints_list[idx];
    
    // 将关键点转换为装甲板结构
    detection.armor = keypoints_to_armor(detection.keypoints, detection.color);
    
    detections.push_back(detection);
  }
  
  std::cout << "神经网络检测完成: 候选=" << boxes.size() 
            << ", NMS后=" << detections.size() << std::endl;
  
  return detections;
}

std::vector<int> OpenVinoArmorDetector::nms(
  const std::vector<cv::Rect> & boxes, 
  const std::vector<float> & scores, 
  float nms_threshold)
{
  std::vector<int> indices;
  
  if (boxes.empty()) {
    return indices;
  }
  
  // 使用OpenCV的NMS
  cv::dnn::NMSBoxes(boxes, scores, 0.0f, nms_threshold, indices);
  
  return indices;
}

Armor OpenVinoArmorDetector::keypoints_to_armor(const std::vector<cv::Point2f> & keypoints, Color color)
{
  if (keypoints.size() != 4) {
    return Armor();  // 返回空装甲板
  }
  
  Armor armor;
  
  // 假设关键点顺序为：左上、右上、右下、左下
  // 构造虚拟的左右灯条
  Light left_light, right_light;
  
  // 左灯条：使用左上和左下点
  left_light.top = keypoints[0];
  left_light.bottom = keypoints[3];
  left_light.center = (left_light.top + left_light.bottom) / 2;
  left_light.length = cv::norm(left_light.top - left_light.bottom);
  left_light.width = 5.0f;  // 假设宽度
  left_light.color = color;
  
  // 右灯条：使用右上和右下点
  right_light.top = keypoints[1];
  right_light.bottom = keypoints[2];
  right_light.center = (right_light.top + right_light.bottom) / 2;
  right_light.length = cv::norm(right_light.top - right_light.bottom);
  right_light.width = 5.0f;  // 假设宽度
  right_light.color = color;
  
  // 构造装甲板
  armor.left_light = left_light;
  armor.right_light = right_light;
  armor.center = (left_light.center + right_light.center) / 2;
  armor.color = color;
  
  // 设置四个关键点
  armor.four_points = keypoints;
  
  // 判断装甲板类型（基于宽度）
  float width = cv::norm(left_light.center - right_light.center);
  armor.type = width > 100 ? ArmorType::LARGE : ArmorType::SMALL;
  
  return armor;
}

void OpenVinoArmorDetector::draw_detections(cv::Mat & image, const std::vector<ArmorDetection> & detections)
{
  for (const auto & detection : detections) {
    // 绘制边界框
    cv::Scalar box_color = detection.color == Color::RED ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
    cv::rectangle(image, detection.box, box_color, 2);
    
    // 绘制关键点
    cv::Scalar point_color = cv::Scalar(0, 255, 0);
    for (size_t i = 0; i < detection.keypoints.size(); ++i) {
      cv::circle(image, detection.keypoints[i], 4, point_color, -1);
      cv::putText(image, std::to_string(i), detection.keypoints[i] + cv::Point2f(5, 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1);
    }
    
    // 连接关键点形成装甲板轮廓
    if (detection.keypoints.size() == 4) {
      cv::Scalar contour_color = cv::Scalar(0, 255, 255);
      for (int i = 0; i < 4; ++i) {
        cv::line(image, detection.keypoints[i], detection.keypoints[(i + 1) % 4], contour_color, 2);
      }
    }
    
    // 准备标签文本
    std::string color_text = detection.color == Color::RED ? "RED" : "BLUE";
    std::string label = color_text + ": " + 
                       std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    
    // 计算文本尺寸
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    
    // 绘制文本背景
    cv::Point text_origin(detection.box.x, detection.box.y - 5);
    cv::rectangle(image, 
                  cv::Point(text_origin.x, text_origin.y - text_size.height - baseline),
                  cv::Point(text_origin.x + text_size.width, text_origin.y + baseline),
                  box_color, cv::FILLED);
    
    // 绘制标签文本
    cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(255, 255, 255), 1);
  }
}

}  // namespace rm_armor_detect
