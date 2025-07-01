#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include "rm_armor_detect/armor_types.hpp"

namespace rm_armor_detect
{

// 神经网络检测结果（带关键点的装甲板）
struct ArmorDetection
{
  cv::Rect box;                       // 装甲板边界框
  float confidence;                   // 置信度
  Color color;                        // 装甲板颜色 (BLUE/RED)
  std::vector<cv::Point2f> keypoints; // 4个关键点坐标
  Armor armor;                        // 转换后的装甲板结构
};

// OpenVINO装甲板检测器（神经网络识别模式）
class OpenVinoArmorDetector
{
public:
  explicit OpenVinoArmorDetector(const std::string & model_path);
  ~OpenVinoArmorDetector() = default;

  // 初始化检测器
  bool initialize();
  
  // 执行装甲板检测（神经网络模式）
  std::vector<ArmorDetection> detect(const cv::Mat & image, float conf_threshold = 0.4f);
  
  // 在图像上绘制检测结果
  void draw_detections(cv::Mat & image, const std::vector<ArmorDetection> & detections);
  
  // 设置置信度阈值
  void set_confidence_threshold(float threshold) { confidence_threshold_ = threshold; }
  
  // 获取当前置信度阈值
  float get_confidence_threshold() const { return confidence_threshold_; }
  
  // 设置神经网络专用的曝光调整参数
  void set_exposure_adjustment(bool enable, double factor = 1.0);
  
  // 获取当前曝光调整设置
  bool get_exposure_adjustment_enabled() const { return enable_exposure_adjustment_; }
  double get_exposure_adjustment_factor() const { return exposure_adjustment_factor_; }

private:
  // 预处理输入图像
  cv::Mat preprocess(const cv::Mat & image);
  
  // 神经网络专用的曝光调整处理
  cv::Mat apply_exposure_adjustment(const cv::Mat & image);
  
  // 后处理检测结果
  std::vector<ArmorDetection> postprocess(const ov::Tensor & output_tensor, 
                                         const cv::Mat & original_image,
                                         float conf_threshold = 0.4f, 
                                         float nms_threshold = 0.5f);
  
  // 执行非极大值抑制
  std::vector<int> nms(const std::vector<cv::Rect> & boxes, 
                       const std::vector<float> & scores, 
                       float nms_threshold);
  
  // 将关键点转换为装甲板结构
  Armor keypoints_to_armor(const std::vector<cv::Point2f> & keypoints, Color color);

private:
  std::string model_path_;
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  
  // 模型输入输出信息
  ov::Shape input_shape_;
  size_t input_height_;
  size_t input_width_;
  
  // 图像缩放比例
  float scale_x_;
  float scale_y_;
  
  bool is_initialized_;
  float confidence_threshold_;
  
  // 神经网络专用的曝光调整参数
  bool enable_exposure_adjustment_;
  double exposure_adjustment_factor_;
};

}  // namespace rm_armor_detect 