#ifndef RM_ARMOR_DETECT__CAR_DETECTOR_HPP_
#define RM_ARMOR_DETECT__CAR_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace rm_armor_detect
{

struct Detection
{
  cv::Rect box;
  float confidence;
  std::string class_name;
};

class CarDetector
{
public:
  explicit CarDetector(const std::string & model_path);
  ~CarDetector() = default;

  // 初始化检测器
  bool initialize();
  
  // 执行检测
  std::vector<Detection> detect(const cv::Mat & image, float conf_threshold = 0.25f);
  
  // 在图像上绘制检测结果
  void draw_detections(cv::Mat & image, const std::vector<Detection> & detections);
  
  // 设置车辆检测专用的降曝光参数
  void set_exposure_reduction(bool enable, double factor = 0.7);

private:
  // 预处理输入图像
  cv::Mat preprocess(const cv::Mat & image);
  
  // 车辆检测专用的降曝光处理
  cv::Mat apply_exposure_reduction(const cv::Mat & image);
  
  // 后处理检测结果
  std::vector<Detection> postprocess(const ov::Tensor & output_tensor, 
                                    float conf_threshold = 0.5f, 
                                    float nms_threshold = 0.4f);
  
  // 执行非极大值抑制
  std::vector<int> nms(const std::vector<cv::Rect> & boxes, 
                       const std::vector<float> & scores, 
                       float nms_threshold);
  
  // 扩大检测框并确保不超出图像边界
  cv::Rect expand_box(const cv::Rect& box, float scale, const cv::Size& image_size);

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
  
  // 车辆检测专用的降曝光参数
  bool enable_exposure_reduction_;
  double exposure_reduction_factor_;
  
  // 用于处理检测失败的逻辑
  std::vector<Detection> last_detections_;  // 上一帧的检测结果
  int no_detection_count_;  // 连续无检测的帧数
  static const int MAX_NO_DETECTION_FRAMES = 10;  // 最大无检测帧数
};

}  // namespace rm_armor_detect

#endif  // RM_ARMOR_DETECT__CAR_DETECTOR_HPP_ 