#include "rm_armor_detect/simple_lights_detector.hpp"
#include <algorithm>
#include <future>
#include <thread>
#include <iostream>

namespace rm_armor_detect
{

SimpleLightsDetector::SimpleLightsDetector(int binary_threshold, int detect_color)
: binary_threshold_(binary_threshold), detect_color_(detect_color)
{
  // 初始化默认的灯条和装甲板参数 - 根据提供的新值更新
  light_params_.min_ratio = 0.1f;  // 灯条最小宽高比
  light_params_.max_ratio = 0.5f;  // 灯条最大宽高比（更新为0.4，原为0.5）
  light_params_.max_angle = 40.0f; // 灯条最大倾斜角度（更新为40.0，原为45.0）
  
  armor_params_.min_light_ratio = 0.7f;       // 灯条最小长度比
  armor_params_.min_small_center_distance = 0.8f;  // 小装甲板最小中心距（更新为0.8，原为1.0）
  armor_params_.max_small_center_distance = 3.2f;  // 小装甲板最大中心距（更新为3.2，原为2.5）
  armor_params_.min_large_center_distance = 3.2f;  // 大装甲板最小中心距（更新为3.2，原为3.0）
  armor_params_.max_large_center_distance = 5.5f;  // 大装甲板最大中心距（更新为5.5，原为5.0）
  armor_params_.max_angle = 35.0f;            // 装甲板最大角度（更新为35.0，原为20.0）
  
  std::cout << "[调试] SimpleLightsDetector 初始化完成:" << std::endl;
  std::cout << "  - 二值化阈值: " << binary_threshold_ << std::endl;
  std::cout << "  - 检测颜色: " << (detect_color_ == 0 ? "蓝色" : "红色") << std::endl;
  std::cout << "  - 灯条参数: ratio[" << light_params_.min_ratio << "-" << light_params_.max_ratio << "], angle<" << light_params_.max_angle << "°" << std::endl;
  std::cout << "  - 装甲板参数: 小装甲距离[" << armor_params_.min_small_center_distance << "-" << armor_params_.max_small_center_distance << "], 大装甲距离[" << armor_params_.min_large_center_distance << "-" << armor_params_.max_large_center_distance << "]" << std::endl;
}

// 初始化数字分类器
void SimpleLightsDetector::initNumberClassifier(
  const std::string & model_path, 
  const std::string & label_path,
  double threshold,
  const std::vector<std::string> & ignore_classes)
{
  try {
    classifier_ = std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);
    has_classifier_ = true;
    std::cout << "[调试] 数字分类器初始化成功：" << model_path << std::endl;
  } catch (const cv::Exception & e) {
    std::cerr << "[错误] 数字分类器初始化失败：" << e.what() << std::endl;
    has_classifier_ = false;
  }
}

std::vector<SimpleLightResult> SimpleLightsDetector::process_detections(
  const cv::Mat & original_image,
  const std::vector<Detection> & car_detections)
{
  std::vector<SimpleLightResult> results;
  
  if (car_detections.empty() || original_image.empty()) {
    return results;
  }

  std::cout << "[调试] ========== 新帧处理开始 ==========" << std::endl;
  std::cout << "[调试] 输入图像尺寸: " << original_image.cols << "x" << original_image.rows << std::endl;
  std::cout << "[调试] 车辆检测框数量: " << car_detections.size() << std::endl;

  // 使用并行处理多个检测框
  std::vector<std::future<SimpleLightResult>> futures;
  
  // 为每个检测框创建异步任务
  for (size_t i = 0; i < car_detections.size(); ++i) {
    const auto & detection = car_detections[i];
    std::cout << "[调试] 车辆检测框[" << i << "]: (" << detection.box.x << "," << detection.box.y 
              << "," << detection.box.width << "," << detection.box.height << ") 置信度=" 
              << detection.confidence << std::endl;
    
    futures.push_back(std::async(std::launch::async, 
      &SimpleLightsDetector::process_single_detection, 
      this, 
      std::ref(original_image), 
      std::ref(detection)));
  }
  
  // 收集所有异步任务的结果
  results.reserve(futures.size());
  for (size_t i = 0; i < futures.size(); ++i) {
    auto result = futures[i].get();
    std::cout << "[调试] 检测框[" << i << "]处理结果: 灯条=" << result.lights.size() 
              << "个, 装甲板=" << result.armors.size() << "个" << std::endl;
    results.push_back(result);
  }
  
  // 统计总结果
  size_t total_lights = 0, total_armors = 0;
  for (const auto & result : results) {
    total_lights += result.lights.size();
    total_armors += result.armors.size();
  }
  std::cout << "[调试] 本帧总结果: 灯条=" << total_lights << "个, 装甲板=" << total_armors << "个" << std::endl;
  std::cout << "[调试] ========== 帧处理结束 ==========" << std::endl;
  
  return results;
}

SimpleLightResult SimpleLightsDetector::process_single_detection(
  const cv::Mat & original_image,
  const Detection & car_detection)
{
  SimpleLightResult result;
  result.car_box = car_detection.box;
  result.confidence = car_detection.confidence;
  
  // 确保检测框在图像边界内
  cv::Rect clamped_box = clamp_rect_to_image(car_detection.box, original_image.size());
  
  std::cout << "[调试] 单框处理: 原始框(" << car_detection.box.x << "," << car_detection.box.y 
            << "," << car_detection.box.width << "," << car_detection.box.height << ") -> 裁剪框(" 
            << clamped_box.x << "," << clamped_box.y << "," << clamped_box.width << "," 
            << clamped_box.height << ")" << std::endl;
  
  // 截取检测框内的图像
  if (clamped_box.width > 0 && clamped_box.height > 0) {
    result.cropped_image = original_image(clamped_box).clone();
    
    // 灰度化处理
    if (result.cropped_image.channels() == 3) {
      cv::cvtColor(result.cropped_image, result.gray_image, cv::COLOR_BGR2GRAY);
    } else if (result.cropped_image.channels() == 1) {
      result.gray_image = result.cropped_image.clone();
    }
    
    // 二值化处理
    if (!result.gray_image.empty()) {
      cv::threshold(result.gray_image, result.binary_image, 
                   binary_threshold_, 255, cv::THRESH_BINARY);
      
      std::cout << "[调试] 图像预处理完成: 裁剪图像" << result.cropped_image.cols << "x" 
                << result.cropped_image.rows << ", 二值化阈值=" << binary_threshold_ << std::endl;
    }
    
    // 添加灯条检测功能
    if (!result.binary_image.empty()) {
      result.lights = findLights(result.cropped_image, result.binary_image);
      std::cout << "[调试] 灯条检测完成: 发现" << result.lights.size() << "个有效灯条" << std::endl;
      
      // 在调试模式下，收集所有轮廓的最小外接矩形
      if (debug_mode_) {
        result.debug_contour_rects = findAllContourRects(result.binary_image);
        std::cout << "[调试] 收集到" << result.debug_contour_rects.size() << "个轮廓矩形用于调试显示" << std::endl;
      }
      
      // 添加装甲板匹配功能
      if (!result.lights.empty()) {
        result.armors = matchLights(result.lights);
        std::cout << "[调试] 装甲板匹配完成: 匹配到" << result.armors.size() << "个装甲板" << std::endl;
        
        // 如果有数字分类器且有装甲板，执行数字识别
        if (has_classifier_ && !result.armors.empty()) {
          std::cout << "[调试] 开始数字识别..." << std::endl;
          classifier_->extractNumbers(result.cropped_image, result.armors);
          classifier_->classify(result.armors);
          std::cout << "[调试] 数字识别完成" << std::endl;
        }
        
        // 处理装甲板结果
        processArmorResults(result.armors);
      }
    }
  } else {
    std::cout << "[警告] 裁剪框无效，跳过处理" << std::endl;
  }
  
  return result;
}

cv::Rect SimpleLightsDetector::clamp_rect_to_image(
  const cv::Rect & rect, 
  const cv::Size & image_size)
{
  int x = std::max(0, rect.x);
  int y = std::max(0, rect.y);
  int width = std::min(rect.width, image_size.width - x);
  int height = std::min(rect.height, image_size.height - y);
  
  // 确保宽度和高度为正数
  width = std::max(0, width);
  height = std::max(0, height);
  
  return cv::Rect(x, y, width, height);
}

std::vector<Light> SimpleLightsDetector::findLights(const cv::Mat & rgb_img, const cv::Mat & binary_img)
{
  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::cout << "[调试] findLights: 发现" << contours.size() << "个轮廓" << std::endl;

  vector<Light> lights;
  int contour_size_filtered = 0;
  int ratio_filtered = 0;
  int angle_filtered = 0;
  int boundary_filtered = 0;
  int valid_lights = 0;

  for (size_t i = 0; i < contours.size(); ++i) {
    const auto & contour = contours[i];
    
    if (contour.size() < 5) {
      contour_size_filtered++;
      continue;
    }

    auto r_rect = cv::minAreaRect(contour);
    auto light = Light(r_rect);

    // 调试单个灯条信息
    std::cout << "[调试] 轮廓[" << i << "]: 长度=" << light.length << ", 宽度=" << light.width 
              << ", 比例=" << (light.width/light.length) << ", 角度=" << light.tilt_angle << "°" << std::endl;

    if (!isLight(light)) {
      // 详细检查失败原因
      float ratio = light.width / light.length;
      bool ratio_ok = light_params_.min_ratio < ratio && ratio < light_params_.max_ratio;
      bool angle_ok = light.tilt_angle < light_params_.max_angle;
      
      if (!ratio_ok) {
        ratio_filtered++;
        std::cout << "[调试] 轮廓[" << i << "] 比例筛选失败: " << ratio 
                  << " 不在[" << light_params_.min_ratio << "," << light_params_.max_ratio << "]范围内" << std::endl;
      }
      if (!angle_ok) {
        angle_filtered++;
        std::cout << "[调试] 轮廓[" << i << "] 角度筛选失败: " << light.tilt_angle 
                  << "° > " << light_params_.max_angle << "°" << std::endl;
      }
      continue;
    }

    auto rect = light.boundingRect();
    if (  // 避免越界
      0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rgb_img.cols && 0 <= rect.y &&
      0 <= rect.height && rect.y + rect.height <= rgb_img.rows) {
      int sum_r = 0, sum_b = 0;
      auto roi = rgb_img(rect);
      // 遍历ROI区域
      for (int row = 0; row < roi.rows; row++) {
        for (int col = 0; col < roi.cols; col++) {
          if (cv::pointPolygonTest(contour, cv::Point2f(col + rect.x, row + rect.y), false) >= 0) {
            // 如果点在轮廓内
            sum_r += roi.at<cv::Vec3b>(row, col)[0];
            sum_b += roi.at<cv::Vec3b>(row, col)[2];
          }
        }
      }
      // 红色像素和 > 蓝色像素和?
      light.color = sum_r > sum_b ? Color::RED : Color::BLUE;
      
      std::cout << "[调试] 轮廓[" << i << "] 颜色分析: R=" << sum_r << ", B=" << sum_b 
                << " -> " << (light.color == Color::RED ? "红色" : "蓝色") << std::endl;
      
      lights.emplace_back(light);
      valid_lights++;
    } else {
      boundary_filtered++;
      std::cout << "[调试] 轮廓[" << i << "] 边界检查失败: 矩形(" << rect.x << "," << rect.y 
                << "," << rect.width << "," << rect.height << ") 超出图像边界(" 
                << rgb_img.cols << "x" << rgb_img.rows << ")" << std::endl;
    }
  }

  std::cout << "[调试] 灯条筛选统计:" << std::endl;
  std::cout << "  - 总轮廓数: " << contours.size() << std::endl;
  std::cout << "  - 轮廓尺寸过滤: " << contour_size_filtered << std::endl;
  std::cout << "  - 比例过滤: " << ratio_filtered << std::endl;
  std::cout << "  - 角度过滤: " << angle_filtered << std::endl;
  std::cout << "  - 边界过滤: " << boundary_filtered << std::endl;
  std::cout << "  - 有效灯条: " << valid_lights << std::endl;

  return lights;
}

std::vector<cv::RotatedRect> SimpleLightsDetector::findAllContourRects(const cv::Mat & binary_img)
{
  std::cout << "[调试] findAllContourRects: 输入图像尺寸=" << binary_img.cols << "x" << binary_img.rows << std::endl;
  
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::cout << "[调试] findAllContourRects: 发现" << contours.size() << "个轮廓" << std::endl;

  std::vector<cv::RotatedRect> debug_rects;
  debug_rects.reserve(contours.size());

  for (size_t i = 0; i < contours.size(); ++i) {
    const auto & contour = contours[i];
    
    if (contour.size() >= 5) {  // 需要至少5个点才能拟合旋转矩形
      auto r_rect = cv::minAreaRect(contour);
      debug_rects.push_back(r_rect);
      std::cout << "[调试] 轮廓[" << i << "]: " << contour.size() << "个点 -> 矩形中心(" 
                << r_rect.center.x << "," << r_rect.center.y << "), 尺寸(" 
                << r_rect.size.width << "x" << r_rect.size.height << ")" << std::endl;
    } else {
      std::cout << "[调试] 轮廓[" << i << "]: " << contour.size() << "个点 -> 跳过（点数不足）" << std::endl;
    }
  }

  std::cout << "[调试] findAllContourRects: 收集了" << debug_rects.size() << "个有效矩形" << std::endl;
  return debug_rects;
}

bool SimpleLightsDetector::isLight(const Light & light)
{
  // 灯条的比例 (短边 / 长边)
  float ratio = light.width / light.length;
  bool ratio_ok = light_params_.min_ratio < ratio && ratio < light_params_.max_ratio;

  bool angle_ok = light.tilt_angle < light_params_.max_angle;

  bool is_light = ratio_ok && angle_ok;

  return is_light;
}

std::vector<Armor> SimpleLightsDetector::matchLights(const std::vector<Light> & lights)
{
  std::vector<Armor> armors;

  // 将int类型的detect_color_转换为Color枚举类型
  Color target_color = static_cast<Color>(detect_color_);
  
  std::cout << "[调试] 装甲板匹配开始: " << lights.size() << "个灯条, 目标颜色=" 
            << (target_color == Color::RED ? "红色" : "蓝色") << std::endl;

  int color_filtered = 0;
  int contain_filtered = 0;
  int geometry_filtered = 0;
  int valid_armors = 0;

  // 遍历所有灯条对
  for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
      
      std::cout << "[调试] 检测灯条对: 灯条1颜色=" << (light_1->color == Color::RED ? "红" : "蓝") 
                << ", 灯条2颜色=" << (light_2->color == Color::RED ? "红" : "蓝") << std::endl;
      
      if (light_1->color != target_color || light_2->color != target_color) {
        color_filtered++;
        std::cout << "[调试] 颜色筛选失败: 灯条颜色不匹配目标颜色" << std::endl;
        continue;
      }

      if (containLight(*light_1, *light_2, lights)) {
        contain_filtered++;
        std::cout << "[调试] 包含检测失败: 两灯条之间存在其他灯条" << std::endl;
        continue;
      }

      auto type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID) {
        auto armor = Armor(*light_1, *light_2);
        armor.type = type;
        armors.emplace_back(armor);
        valid_armors++;
        std::cout << "[调试] 发现有效装甲板: 类型=" << (type == ArmorType::SMALL ? "小装甲板" : "大装甲板") << std::endl;
      } else {
        geometry_filtered++;
        std::cout << "[调试] 几何验证失败: 不符合装甲板几何约束" << std::endl;
      }
    }
  }

  std::cout << "[调试] 装甲板匹配统计:" << std::endl;
  std::cout << "  - 可能的灯条对数: " << (lights.size() * (lights.size() - 1) / 2) << std::endl;
  std::cout << "  - 颜色过滤: " << color_filtered << std::endl;
  std::cout << "  - 包含过滤: " << contain_filtered << std::endl;
  std::cout << "  - 几何过滤: " << geometry_filtered << std::endl;
  std::cout << "  - 有效装甲板: " << valid_armors << std::endl;

  return armors;
}

bool SimpleLightsDetector::containLight(
  const Light & light_1, const Light & light_2, const std::vector<Light> & lights)
{
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);

  for (const auto & test_light : lights) {
    if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

    if (
      bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
      bounding_rect.contains(test_light.center)) {
      return true;
    }
  }

  return false;
}

ArmorType SimpleLightsDetector::isArmor(const Light & light_1, const Light & light_2)
{
  std::cout << "[调试] isArmor检查开始:" << std::endl;
  std::cout << "  - 灯条1: 长度=" << light_1.length << ", 中心(" << light_1.center.x << "," << light_1.center.y << ")" << std::endl;
  std::cout << "  - 灯条2: 长度=" << light_2.length << ", 中心(" << light_2.center.x << "," << light_2.center.y << ")" << std::endl;

  // 两个灯条的长度比例 (短边 / 长边)
  float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                           : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > armor_params_.min_light_ratio;
  
  std::cout << "  - 长度比例检查: " << light_length_ratio << " > " << armor_params_.min_light_ratio 
            << " ? " << (light_ratio_ok ? "通过" : "失败") << std::endl;

  // 两个灯条中心的距离 (单位: 灯条长度)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  bool center_distance_ok = (armor_params_.min_small_center_distance <= center_distance &&
                           center_distance < armor_params_.max_small_center_distance) ||
                          (armor_params_.min_large_center_distance <= center_distance &&
                           center_distance < armor_params_.max_large_center_distance);

  std::cout << "  - 平均长度: " << avg_light_length << std::endl;
  std::cout << "  - 中心距离: " << cv::norm(light_1.center - light_2.center) << " 像素" << std::endl;
  std::cout << "  - 标准化距离: " << center_distance << std::endl;
  std::cout << "  - 小装甲板范围: [" << armor_params_.min_small_center_distance << ", " << armor_params_.max_small_center_distance << ")" << std::endl;
  std::cout << "  - 大装甲板范围: [" << armor_params_.min_large_center_distance << ", " << armor_params_.max_large_center_distance << ")" << std::endl;
  
  bool is_small = (armor_params_.min_small_center_distance <= center_distance &&
                   center_distance < armor_params_.max_small_center_distance);
  bool is_large = (armor_params_.min_large_center_distance <= center_distance &&
                   center_distance < armor_params_.max_large_center_distance);
  
  std::cout << "  - 距离检查: 小装甲板=" << (is_small ? "符合" : "不符合") 
            << ", 大装甲板=" << (is_large ? "符合" : "不符合") 
            << " -> " << (center_distance_ok ? "通过" : "失败") << std::endl;

  // 灯条中心连线的角度
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  bool angle_ok = angle < armor_params_.max_angle;

  std::cout << "  - 中心差值: (" << diff.x << ", " << diff.y << ")" << std::endl;
  std::cout << "  - 连线角度: " << angle << "° < " << armor_params_.max_angle << "° ? " 
            << (angle_ok ? "通过" : "失败") << std::endl;

  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  std::cout << "  - 综合判断: " << (is_armor ? "是装甲板" : "不是装甲板") << std::endl;

  // 判断装甲板类型
  ArmorType type;
  if (is_armor) {
    type = center_distance > armor_params_.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    std::cout << "  - 装甲板类型: " << (type == ArmorType::LARGE ? "大装甲板" : "小装甲板") 
              << " (距离=" << center_distance << " vs 阈值=" << armor_params_.min_large_center_distance << ")" << std::endl;
  } else {
    type = ArmorType::INVALID;
    std::cout << "  - 验证失败原因: ";
    if (!light_ratio_ok) std::cout << "长度比例不符 ";
    if (!center_distance_ok) std::cout << "中心距离不符 ";
    if (!angle_ok) std::cout << "角度不符 ";
    std::cout << std::endl;
  }

  return type;
}

void SimpleLightsDetector::processArmorResults(std::vector<Armor> & armors)
{
  for (size_t i = 0; i < armors.size(); i++) {
    // 设置装甲板的基本信息
    armors[i].color = armors[i].left_light.color;
    armors[i].four_points = {
      armors[i].left_light.top, 
      armors[i].right_light.top, 
      armors[i].right_light.bottom, 
      armors[i].left_light.bottom
    };
    
    // 如果有数字识别结果，则处理ID转换
    if (!armors[i].number.empty()) {
      if (armors[i].number == "1" || armors[i].number == "2" || armors[i].number == "3" || 
          armors[i].number == "4" || armors[i].number == "5") {
        armors[i].id = std::stoi(armors[i].number);
      }
      else if (armors[i].number == "outpost") {
        armors[i].id = 0;
      }
      else if (armors[i].number == "guard") {
        armors[i].id = 6;
      }
      else if (armors[i].number == "base") {
        armors[i].id = 7;
      }
      
      armors[i].number_score = armors[i].confidence;
      armors[i].color_score = 1.0;
      
      // 创建名称字符串
      if (armors[i].color == Color::BLUE) {
        armors[i].name = std::to_string(armors[i].id) + " blue--" + 
                         std::to_string(armors[i].number_score) + "--" + 
                         std::to_string(armors[i].color_score);
      } else {
        armors[i].name = std::to_string(armors[i].id) + " red--" + 
                         std::to_string(armors[i].number_score) + "--" + 
                         std::to_string(armors[i].color_score);
      }
      
      // 设置矩形框
      armors[i].rect = cv::Rect(armors[i].left_light.top, armors[i].right_light.bottom);
    }
  }
}

void SimpleLightsDetector::drawResults(cv::Mat & img, const std::vector<Light> & lights, const std::vector<Armor> & armors, 
                                       const std::vector<cv::RotatedRect> & debug_rects)
{
  std::cout << "[调试] drawResults调用: debug_mode_=" << debug_mode_ 
            << ", debug_rects.size()=" << debug_rects.size() 
            << ", img.size()=" << img.cols << "x" << img.rows << std::endl;
  
  // 调试模式下绘制所有轮廓的最小外接矩形（紫色）
  if (debug_mode_ && !debug_rects.empty()) {
    std::cout << "[调试] 开始绘制紫色轮廓矩形..." << std::endl;
    for (size_t i = 0; i < debug_rects.size(); i++) {
      const auto & rect = debug_rects[i];
      cv::Point2f vertices[4];
      rect.points(vertices);
      
      std::cout << "[调试] 矩形[" << i << "]: 中心(" << rect.center.x << "," << rect.center.y 
                << "), 尺寸(" << rect.size.width << "x" << rect.size.height 
                << "), 角度=" << rect.angle << "°" << std::endl;
      
      // 用紫色线条绘制旋转矩形
      cv::Scalar purple_color(255, 0, 255);  // BGR格式的紫色
      for (int j = 0; j < 4; j++) {
        cv::line(img, vertices[j], vertices[(j + 1) % 4], purple_color, 2);  // 增加线宽到2
      }
    }
    std::cout << "[调试] 绘制了" << debug_rects.size() << "个紫色轮廓矩形" << std::endl;
  } else {
    std::cout << "[调试] 跳过紫色轮廓绘制: debug_mode_=" << debug_mode_ 
              << ", debug_rects为空=" << debug_rects.empty() << std::endl;
  }

  // 绘制灯条
  for (const auto & light : lights) {
    cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color = light.color == Color::RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
    cv::line(img, light.top, light.bottom, line_color, 1);
  }

  // 绘制装甲板
  for (const auto & armor : armors) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    
    // 显示数字和置信度
    if (!armor.classfication_result.empty()) {
      cv::putText(img, armor.classfication_result, armor.left_light.top, 
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
  }
}

}  // namespace rm_armor_detect 