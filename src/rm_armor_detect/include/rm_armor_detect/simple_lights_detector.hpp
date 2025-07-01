#pragma once

#include "opencv2/opencv.hpp"
#include "rm_armor_detect/armor_types.hpp"
#include "rm_armor_detect/number_classifier.hpp"
#include "rm_armor_detect/car_detector.hpp"

namespace rm_armor_detect
{

// 灯条检测结果
struct SimpleLightResult {
  cv::Rect car_box;     // 车辆检测框
  float confidence;     // 车辆置信度
  cv::Mat cropped_image; // 裁剪出的原始图像
  cv::Mat gray_image;    // 灰度图像
  cv::Mat binary_image;  // 二值化图像
  std::vector<Light> lights; // 检测到的灯条
  std::vector<Armor> armors; // 匹配到的装甲板
  std::vector<cv::RotatedRect> debug_contour_rects; // 调试模式：所有轮廓的最小外接矩形
};

// 灯条参数
struct LightParams {
  float min_ratio;       // 最小宽高比
  float max_ratio;       // 最大宽高比
  float max_angle;       // 最大角度
};

// 装甲板参数
struct ArmorParams {
  float min_light_ratio;            // 最小灯条长度比
  float min_small_center_distance;  // 小装甲板最小中心距
  float max_small_center_distance;  // 小装甲板最大中心距
  float min_large_center_distance;  // 大装甲板最小中心距
  float max_large_center_distance;  // 大装甲板最大中心距
  float max_angle;                  // 最大角度
};

class SimpleLightsDetector
{
public:
  SimpleLightsDetector(int binary_threshold, int detect_color = 0);
  ~SimpleLightsDetector() = default;

  // 初始化数字分类器
  void initNumberClassifier(
    const std::string & model_path, 
    const std::string & label_path,
    double threshold = 0.7,
    const std::vector<std::string> & ignore_classes = {});

  // 处理多个检测框
  std::vector<SimpleLightResult> process_detections(
    const cv::Mat & original_image,
    const std::vector<Detection> & car_detections);

  // 处理单个检测框
  SimpleLightResult process_single_detection(
    const cv::Mat & original_image,
    const Detection & car_detection);

  // 灯条检测
  std::vector<Light> findLights(const cv::Mat & rgb_img, const cv::Mat & binary_img);

  // 查找所有轮廓的最小外接矩形（调试用）
  std::vector<cv::RotatedRect> findAllContourRects(const cv::Mat & binary_img);

  // 判断是否为灯条
  bool isLight(const Light & light);

  // 装甲板匹配
  std::vector<Armor> matchLights(const std::vector<Light> & lights);

  // 检查是否包含其他灯条
  bool containLight(const Light & light_1, const Light & light_2, const std::vector<Light> & lights);

  // 判断是否为装甲板
  ArmorType isArmor(const Light & light_1, const Light & light_2);

  // 处理装甲板结果
  void processArmorResults(std::vector<Armor> & armors);

  // 绘制结果
  void drawResults(cv::Mat & img, const std::vector<Light> & lights, const std::vector<Armor> & armors, 
                   const std::vector<cv::RotatedRect> & debug_rects = {});

  // 裁剪矩形到图像边界
  cv::Rect clamp_rect_to_image(const cv::Rect & rect, const cv::Size & image_size);

  // 设置二值化阈值
  void set_binary_threshold(int threshold) { binary_threshold_ = threshold; }

  // 获取当前二值化阈值
  int get_binary_threshold() const { return binary_threshold_; }

  // 设置调试模式
  void set_debug_mode(bool debug) { debug_mode_ = debug; }

  // 获取调试模式状态
  bool get_debug_mode() const { return debug_mode_; }

private:
  int binary_threshold_;  // 二值化阈值
  int detect_color_;      // 检测颜色 (0:BLUE, 1:RED)
  bool debug_mode_ = true; // 调试模式标志，默认开启

  LightParams light_params_;  // 灯条参数
  ArmorParams armor_params_;  // 装甲板参数

  // 数字分类器
  std::unique_ptr<NumberClassifier> classifier_;
  bool has_classifier_ = false;
};

}  // namespace rm_armor_detect 