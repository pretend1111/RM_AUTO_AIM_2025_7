#ifndef RM_RUNE_DETECT__OPENVINO_DETECT_HPP_
#define RM_RUNE_DETECT__OPENVINO_DETECT_HPP_

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace rm_rune_detect
{

// 检测结果结构体
struct Detection {
    cv::Rect box;        // 边界框
    float confidence;    // 置信度
    int class_id;        // 类别ID
};

class OpenvinoDetect
{
public:
    // 构造函数：使用默认或指定输入尺寸
    OpenvinoDetect(
        const std::string &model_path,
        const cv::Size model_input_shape = cv::Size(320, 320),
        const float confidence_threshold = 0.5,
        const float nms_threshold = 0.45,
        const bool enable_debug = false);
    
    // 析构函数
    ~OpenvinoDetect() = default;
    
    // 执行目标检测
    std::vector<Detection> detect(cv::Mat &frame);
    
    // 绘制检测结果
    void drawDetections(cv::Mat &frame, const std::vector<Detection> &detections);
    
    // 设置类别名称
    void setClassNames(const std::vector<std::string> &class_names) {
        classes_ = class_names;
    }

private:
    // 初始化模型
    void initializeModel(const std::string &model_path);
    
    // 预处理
    void preprocess(const cv::Mat &frame);
    
    // 后处理
    std::vector<Detection> postprocess(const cv::Mat &frame);
    
    // 从YOLO输出张量中提取检测结果
    std::vector<Detection> extractDetections(const float* data, 
                                            const ov::Shape& output_shape,
                                            const cv::Mat& frame);
    
    // 基于颜色特征扩展检测区域（特别适用于能量机关）
    cv::Rect enlargeDetectionArea(const cv::Mat& frame, const cv::Rect& box);
                                            
    // OpenVINO模型与推理相关
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest inference_request_;
    
    // 模型配置
    cv::Size model_input_shape_;
    float confidence_threshold_;
    float nms_threshold_;
    
    // 原始帧与输入尺寸的比例
    cv::Point2f scale_factor_;
    
    // 类别名称
    std::vector<std::string> classes_ = {"0", "1"};
    
    // 调试开关
    bool enable_debug_ = false;
};

}  // namespace rm_rune_detect

#endif  // RM_RUNE_DETECT__OPENVINO_DETECT_HPP_
