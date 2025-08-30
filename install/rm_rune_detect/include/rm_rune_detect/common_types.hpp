#ifndef RM_RUNE_DETECT__COMMON_TYPES_HPP_
#define RM_RUNE_DETECT__COMMON_TYPES_HPP_

#include <opencv2/opencv.hpp>

namespace rm_rune_detect
{

// 检测结果结构体
struct Detection {
    cv::Rect box;            // 边界框
    int class_id;            // 类别ID
    float confidence;        // 置信度
};

}  // namespace rm_rune_detect

#endif  // RM_RUNE_DETECT__COMMON_TYPES_HPP_ 