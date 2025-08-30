#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rm_armor_detect/armor_types.hpp"

namespace rm_armor_detect
{

// PnP解算结果结构体
struct PnPResult
{
  cv::Mat rvec;                    // 旋转向量
  cv::Mat tvec;                    // 平移向量
  cv::Point3f position;            // 装甲板中心在相机坐标系下的位置 (x, y, z)
  double distance;                 // 装甲板到相机的距离
  double yaw_angle;                // 偏航角 (弧度)
  double pitch_angle;              // 俯仰角 (弧度)
  double distance_to_image_center; // 装甲板中心到图像中心的像素距离
  bool valid;                      // 解算是否有效
};

class PnpSolver
{
public:
  PnpSolver();
  ~PnpSolver() = default;
  
  // 初始化PnP求解器
  void init();
  
  // 设置相机参数
  void set_camera_params(int image_width, int image_height, 
                        const cv::Mat & camera_matrix, 
                        const cv::Mat & dist_coeffs);
  
  // PnP解算主函数
  bool solve_pnp(const std::vector<cv::Point2f> & image_points, 
                 ArmorType armor_type,
                 PnPResult & result);
  
  // 计算装甲板中心到图像中心的距离
  double get_distance_armor_center_to_image_center(const cv::Point2f & armor_center);
  
  // 从旋转向量和平移向量计算角度信息
  void calculate_angles(const cv::Mat & rvec, const cv::Mat & tvec, 
                       double & yaw_angle, double & pitch_angle);
  
  // 降自由度优化：使用固定pitch和roll，只求解yaw角
  double solve_yaw_angle_optimized(const std::vector<cv::Point2f> & image_points, 
                                  ArmorType armor_type, 
                                  const cv::Mat & tvec);
  
  // 使用黄金分割法搜索最优yaw角
  double golden_section_search_yaw(const std::vector<cv::Point2f> & image_points,
                                  ArmorType armor_type,
                                  const cv::Mat & tvec,
                                  double left_bound, double right_bound, 
                                  double tolerance = 1e-3);
  
  // 计算给定yaw角下的重投影误差
  double calculate_yaw_reprojection_error(const std::vector<cv::Point2f> & observed_points,
                                         const std::vector<cv::Point3f> & object_points,
                                         const cv::Mat & tvec,
                                         double yaw, double pitch, double roll);
  
  // 检查相机参数是否已设置
  bool is_initialized() const { return initialized_; }
  
  // 获取相机参数
  cv::Mat get_camera_matrix() const { return camera_matrix_; }
  cv::Mat get_dist_coeffs() const { return dist_coeffs_; }
  
private:
  // 装甲板物理尺寸常量 (单位: 米)
  static constexpr double SMALL_ARMOR_WIDTH = 0.135;        // 小装甲板宽度
  static constexpr double SMALL_ARMOR_LIGHT_HEIGHT = 0.055; // 小装甲板灯条高度
  static constexpr double BIG_ARMOR_WIDTH = 0.225;          // 大装甲板宽度
  static constexpr double BIG_ARMOR_LIGHT_HEIGHT = 0.055;   // 大装甲板灯条高度
  
  // 相机参数
  int image_width_;
  int image_height_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Point2f image_center_;
  bool initialized_;
  
  // 3D模型点
  std::vector<cv::Point3f> small_armor_points_3d_;
  std::vector<cv::Point3f> big_armor_points_3d_;
  
  // 初始化3D模型点
  void init_armor_3d_points();
};

}  // namespace rm_armor_detect 