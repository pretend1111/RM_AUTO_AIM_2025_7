#include "rm_rune_detect/preprocess.hpp"

namespace rm_rune_detect
{

cv::Mat ImagePreprocessor::adjustExposure(const cv::Mat &input, double exposure_factor)
{
  if (input.empty()) {
    return input;
  }
  
  // 限制曝光因子范围
  exposure_factor = std::max(0.1, std::min(exposure_factor, 2.0));
  
  cv::Mat result;
  cv::Mat hsv;
  
  // 转换为HSV色彩空间，便于调整亮度
  cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
  
  // 分离通道
  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);
  
  // 调整V通道(亮度)
  channels[2] = channels[2] * exposure_factor;
  
  // 合并通道
  cv::merge(channels, hsv);
  
  // 转回BGR色彩空间
  cv::cvtColor(hsv, result, cv::COLOR_HSV2BGR);
  
  return result;
}

}  // namespace rm_rune_detect 