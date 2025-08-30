#ifndef RM_RUNE_DETECT__PREPROCESS_HPP_
#define RM_RUNE_DETECT__PREPROCESS_HPP_

#include <opencv2/opencv.hpp>

namespace rm_rune_detect
{

/**
 * @brief 图像预处理类，提供图像曝光度调整功能
 */
class ImagePreprocessor
{
public:
  /**
   * @brief 调整图像曝光度
   * @param input 输入图像
   * @param exposure_factor 曝光调整因子，范围[0.1-2.0]，小于1降低曝光，大于1增加曝光
   * @return 处理后的图像
   */
  static cv::Mat adjustExposure(const cv::Mat &input, double exposure_factor = 0.5);
};

}  // namespace rm_rune_detect

#endif  // RM_RUNE_DETECT__PREPROCESS_HPP_
