#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace rm_armor_detect
{

enum class Color { BLUE = 0, RED = 1 };

enum class ArmorType { SMALL, LARGE, INVALID };

// 灯条结构
struct Light
{
  Light() = default;
  Light(const cv::RotatedRect & r_rect)
  {
    cv::Point2f vertices[4];
    r_rect.points(vertices);
    std::sort(vertices, vertices + 4, [](const cv::Point2f & a, const cv::Point2f & b) {
      return a.y < b.y;
    });
    std::sort(vertices, vertices + 2, [](const cv::Point2f & a, const cv::Point2f & b) {
      return a.x < b.x;
    });
    std::sort(vertices + 2, vertices + 4, [](const cv::Point2f & a, const cv::Point2f & b) {
      return a.x > b.x;
    });
    top = (vertices[0] + vertices[1]) / 2;
    bottom = (vertices[2] + vertices[3]) / 2;
    center = (top + bottom) / 2;
    
    // 修复：正确计算灯条长度和宽度
    // 在OpenCV中，RotatedRect的size.width总是较大的值，size.height总是较小的值
    // 无论矩形的实际方向如何，width >= height
    length = std::max(r_rect.size.width, r_rect.size.height);   // 长边（较大值）
    width = std::min(r_rect.size.width, r_rect.size.height);    // 短边（较小值）
    
    // 修复：计算长边与竖直方向的夹角的绝对值
    // OpenCV的RotatedRect.angle范围是[-90, 0)，表示从x轴（水平）到长边的角度
    // 要得到长边与竖直方向(y轴)的夹角，公式是：|90 + angle|
    // 但如果结果大于90度，则用180度减去它来确保角度在0-90度范围内
    tilt_angle = std::abs(90 + r_rect.angle);
    if (tilt_angle > 90) {
      tilt_angle = 180 - tilt_angle;
    }
  }

  cv::Point2f top;       // 灯条顶部中点
  cv::Point2f bottom;    // 灯条底部中点
  cv::Point2f center;    // 灯条中心点
  float length;          // 灯条长度（长边）
  float width;           // 灯条宽度（短边）
  float tilt_angle;      // 灯条长边与竖直方向的夹角（绝对值，0-90度）
  Color color;           // 灯条颜色

  // 获取灯条的包围矩形
  cv::Rect boundingRect() const
  {
    cv::Point2f vertices[4] = {top, bottom, top, bottom};
    float half_w = width / 2;
    vertices[0] = cv::Point2f(top.x - half_w, top.y);
    vertices[1] = cv::Point2f(top.x + half_w, top.y);
    vertices[2] = cv::Point2f(bottom.x - half_w, bottom.y);
    vertices[3] = cv::Point2f(bottom.x + half_w, bottom.y);
    return cv::boundingRect(std::vector<cv::Point2f>(vertices, vertices + 4));
  }
};

// 装甲板结构
struct Armor
{
  Armor() = default;
  
  Armor(const Light & left, const Light & right)
  : left_light(left), right_light(right)
  {
    center = (left.center + right.center) / 2;
  }

  Light left_light;      // 左灯条
  Light right_light;     // 右灯条
  cv::Point2f center;    // 装甲板中心点
  ArmorType type;        // 装甲板类型
  Color color;           // 装甲板颜色
  
  // 数字识别相关
  cv::Mat number_img;                // 数字ROI图像
  std::string number;                // 识别出的数字
  double confidence;                 // 置信度
  std::string classfication_result;  // 分类结果字符串
  
  // 输出结果相关
  int id;                            // 装甲板ID
  double number_score;               // 数字分数
  double color_score;                // 颜色分数
  std::string name;                  // 装甲板名称
  cv::Rect rect;                     // 装甲板矩形框
  std::vector<cv::Point2f> four_points;  // 装甲板四个点
};

} // namespace rm_armor_detect 