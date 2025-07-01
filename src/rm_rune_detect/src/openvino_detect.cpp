#include "rm_rune_detect/openvino_detect.hpp"
#include <random>
#include <opencv2/dnn.hpp>

namespace rm_rune_detect
{

// 构造函数
OpenvinoDetect::OpenvinoDetect(
    const std::string &model_path,
    const cv::Size model_input_shape,
    const float confidence_threshold,
    const float nms_threshold,
    const bool enable_debug)
    : model_input_shape_(model_input_shape),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold),
      enable_debug_(enable_debug)
{
    if (enable_debug_) {
        // std::cout << "创建OpenVINO检测器，输入尺寸: " << model_input_shape_.width 
        //           << "x" << model_input_shape_.height << std::endl;
        // std::cout << "置信度阈值: " << confidence_threshold_ 
        //           << ", NMS阈值: " << nms_threshold_ << std::endl;
    }
    
    initializeModel(model_path);
}

// 初始化模型
void OpenvinoDetect::initializeModel(const std::string &model_path)
{
    try {
        if (enable_debug_) {
            // std::cout << "加载模型: " << model_path << std::endl;
        }
        
        // 创建OpenVINO Core对象
        core_ = ov::Core();
        
        // 读取模型
        std::shared_ptr<ov::Model> model = core_.read_model(model_path);
        
        // 如果是动态形状模型，设置为静态形状
        if (model->is_dynamic()) {
            if (enable_debug_) {
                // std::cout << "检测到动态形状模型，设置为静态形状: [1, 3, " 
                //           << model_input_shape_.height << ", " 
                //           << model_input_shape_.width << "]" << std::endl;
            }
            
            model->reshape({1, 3, 
                           static_cast<long int>(model_input_shape_.height),
                           static_cast<long int>(model_input_shape_.width)});
        }
        
        // 设置预处理
        ov::preprocess::PrePostProcessor ppp(model);
        
        // 设置输入格式：U8类型，NHWC布局，BGR格式
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        
        // 设置预处理步骤：转为FP32，RGB格式，归一化
        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale({255.0f, 255.0f, 255.0f});
        
        // 设置模型输入格式：NCHW布局
        ppp.input().model().set_layout("NCHW");
        
        // 设置输出为FP32格式
        ppp.output().tensor().set_element_type(ov::element::f32);
        
        // 构建预处理管道
        model = ppp.build();
        
        // 编译模型，使用CPU设备
        compiled_model_ = core_.compile_model(model, "CPU");
        
        // 创建推理请求
        inference_request_ = compiled_model_.create_infer_request();
        
        if (enable_debug_) {
            // 获取模型信息
            // std::cout << "模型输入数量: " << model->inputs().size() << std::endl;
            // std::cout << "模型输出数量: " << model->outputs().size() << std::endl;
            
            // 打印输入形状
            for (const auto& input : model->inputs()) {
                const auto& shape = input.get_shape();
                // std::cout << "输入形状: [";
                // for (size_t i = 0; i < shape.size(); ++i) {
                //     std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
                // }
                // std::cout << "]" << std::endl;
            }
            
            // 打印输出形状
            for (const auto& output : model->outputs()) {
                const auto& shape = output.get_shape();
                // std::cout << "输出形状: [";
                // for (size_t i = 0; i < shape.size(); ++i) {
                //     std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
                // }
                // std::cout << "]" << std::endl;
            }
        }
        
        if (enable_debug_) {
            // std::cout << "模型初始化成功" << std::endl;
        }
    }
    catch (const ov::Exception& e) {
        std::cerr << "OpenVINO初始化错误: " << e.what() << std::endl;
        throw;
    }
    catch (const std::exception& e) {
        std::cerr << "初始化模型时发生异常: " << e.what() << std::endl;
        throw;
    }
}

// 执行目标检测
std::vector<Detection> OpenvinoDetect::detect(cv::Mat &frame)
{
    try {
        // 预处理
        preprocess(frame);
        
        // 执行推理
        inference_request_.infer();
        
        // 后处理并返回结果
        return postprocess(frame);
    }
    catch (const std::exception& e) {
        std::cerr << "检测过程中发生异常: " << e.what() << std::endl;
        return {};
    }
}

// 预处理图像以增强目标特征
void OpenvinoDetect::preprocess(const cv::Mat &frame)
{
    // 创建预处理后的图像
    cv::Mat processed;
    
    // 增强发光目标特征（针对能量机关蓝色发光部分）
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // 分离通道，获取亮度
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    // 增强亮度
    cv::Mat enhanced_v;
    cv::normalize(channels[2], enhanced_v, 0, 255, cv::NORM_MINMAX);
    
    // 如果启用调试，保存增强前后对比
    // if (enable_debug_) {
    //     cv::imshow("原始亮度", channels[2]);
    //     cv::imshow("增强亮度", enhanced_v);
    //     cv::waitKey(1);
    // }
    
    // 使用增强后的亮度
    channels[2] = enhanced_v;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, processed, cv::COLOR_HSV2BGR);
    
    // 调整图像大小
    cv::Mat resized;
    cv::resize(processed, resized, model_input_shape_, 0, 0, cv::INTER_LINEAR);
    
    // 保存原图和缩放后的图片大小比例，用于后处理坐标还原
    scale_factor_.x = static_cast<float>(frame.cols) / model_input_shape_.width;
    scale_factor_.y = static_cast<float>(frame.rows) / model_input_shape_.height;
    
    if (enable_debug_) {
        // std::cout << "缩放比例: " << scale_factor_.x << "x" << scale_factor_.y << std::endl;
    }
    
    // 设置输入张量
    ov::Tensor input_tensor = inference_request_.get_input_tensor();
    
    // 复制数据到输入张量
    std::memcpy(input_tensor.data(), resized.data, resized.total() * resized.elemSize());
}

// 后处理推理结果
std::vector<Detection> OpenvinoDetect::postprocess(const cv::Mat &frame)
{
    // 获取输出张量
    const ov::Tensor& output_tensor = inference_request_.get_output_tensor();
    const float* detection_data = output_tensor.data<const float>();
    const ov::Shape& output_shape = output_tensor.get_shape();
    
    if (enable_debug_) {
        // std::cout << "推理结果输出形状: [";
        // for (size_t i = 0; i < output_shape.size(); ++i) {
        //     std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
        // }
        // std::cout << "]" << std::endl;
    }
    
    // 提取检测结果
    return extractDetections(detection_data, output_shape, frame);
}

// 从YOLOv8格式输出中提取检测结果
std::vector<Detection> OpenvinoDetect::extractDetections(const float* data, 
                                                       const ov::Shape& output_shape, 
                                                       const cv::Mat& frame)
{
    std::vector<Detection> detections;
    
    // 获取图像尺寸
    int img_width = frame.cols;
    int img_height = frame.rows;
    
    if (enable_debug_) {
        // std::cout << "原始图像尺寸: " << img_width << "x" << img_height << std::endl;
        // std::cout << "缩放因子: " << scale_factor_.x << "x" << scale_factor_.y << std::endl;
    }
    
    // 处理YOLOv8输出
    // 典型输出格式为[1, 84, 8400]，其中84 = 4(边界框) + 80(类别数)，8400是预测框数量
    // 注意：实际使用的模型可能有不同输出格式
    
    if (output_shape.size() < 2) {
        std::cerr << "输出维度不足，无法解析" << std::endl;
        return detections;
    }
    
    // 确定输出布局
    size_t rows, cols;
    
    // 存储候选检测结果
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // 标记是否已经处理过检测结果
    bool processed_detections = false;
    
    // 预先声明变量，避免goto跨越初始化问题
    size_t num_boxes = 0;
    
    // 检查输出形状，确定行列
    if (output_shape.size() == 3 && output_shape[0] == 1) {
        // 格式为[1, rows, cols]
        rows = output_shape[1];
        cols = output_shape[2];
        
        if (enable_debug_) {
            // std::cout << "检测到输出格式: [1, " << rows << ", " << cols << "]" << std::endl;
        }
    }
    else if (output_shape.size() == 2) {
        // 格式为[rows, cols]
        rows = output_shape[0];
        cols = output_shape[1];
        
        if (enable_debug_) {
            // std::cout << "检测到输出格式: [" << rows << ", " << cols << "]" << std::endl;
        }
    }
    else {
        std::cerr << "不支持的输出形状" << std::endl;
        return detections;
    }
    
    // 判断输出格式，确定类别个数和边界框表示
    const size_t box_dims = 4;  // x, y, w, h
    int num_classes = 0;
    
    if (rows > box_dims) {
        // 输出格式为[1, 84, 8400]，其中84 = 4(边界框) + 80(类别数)
        num_classes = rows - box_dims;
    }
    else if (cols > box_dims) {
        // 输出格式为[8400, 84]，其中84 = 4(边界框) + 80(类别数)
        num_classes = cols - box_dims;
    }
    
    if (enable_debug_) {
        // std::cout << "检测到类别数: " << num_classes << std::endl;
    }
    
    // 限制类别数到实际定义的类别
    num_classes = std::min(num_classes, static_cast<int>(classes_.size()));
    
    if (num_classes <= 0) {
        std::cerr << "无有效类别信息" << std::endl;
        return detections;
    }
    
    // 解析输出布局
    bool transpose = false;
    
    // 定义转置标志
    // false: [1, 84, 8400] - 边界框+置信度在行，每个预测框在列
    // true: [1, 8400, 84] 或 [8400, 84] - 每个预测框在行，边界框+置信度在列
    if (rows <= cols && rows >= box_dims + 1) {
        // 输出格式可能是 [1, 84, 8400]
        transpose = false;
        
        if (enable_debug_) {
            // std::cout << "使用非转置布局解析: 类别在行，框在列" << std::endl;
        }
    }
    else if (cols >= box_dims + 1 && cols <= 100) {
        // 输出格式可能是 [8400, 84]
        transpose = true;
        
        if (enable_debug_) {
            // std::cout << "使用转置布局解析: 框在行，类别在列" << std::endl;
        }
    }
    else {
        // 尝试特殊格式，如[1, 6, 2100]
        if (output_shape.size() == 3 && output_shape[0] == 1 && output_shape[1] == 6) {
            if (enable_debug_) {
                // std::cout << "检测到特殊输出格式 [1, 6, 2100]" << std::endl;
            }
            
            // 特殊格式处理，参考旧代码
            const int num_detections = output_shape[2];
            
            // 遍历所有检测
            for (int i = 0; i < num_detections; i++) {
                // 获取置信度
                float raw_confidence = data[5 * num_detections + i];
                float confidence = 1.0f / (1.0f + std::exp(-raw_confidence));
                
                // 降低置信度阈值以提高召回率
                float adjusted_threshold = 0.4f;  // 降低置信度阈值
                
                // 如果置信度低于阈值，跳过
                if (confidence < adjusted_threshold) {
                    continue;
                }
                
                // 获取边界框坐标
                float x1 = data[0 * num_detections + i];
                float y1 = data[1 * num_detections + i];
                float x2 = data[2 * num_detections + i];
                float y2 = data[3 * num_detections + i];
                
                // 获取类别得分
                float class_score = data[4 * num_detections + i];
                int class_id = class_score > 0.5f ? 1 : 0;  // 降低类别阈值，提高召回率
                
                // 检查是否需要反转坐标
                if (x1 > x2) std::swap(x1, x2);
                if (y1 > y2) std::swap(y1, y2);
                
                // 检查是否是归一化坐标(0-1范围)
                // 对于能量机关目标，大多数情况下应该是归一化坐标
                bool normalized = true;
                
                if (std::abs(x1) > 1.0f || std::abs(y1) > 1.0f || 
                    std::abs(x2) > 1.0f || std::abs(y2) > 1.0f) {
                    normalized = false;
                }
                
                // 转换坐标，确保正确应用缩放因子
                if (normalized) {
                    // 如果是归一化坐标(0-1)，直接应用到原始图像尺寸
                    // 这是关键修复：使用原始图像尺寸而不是模型输入尺寸*缩放因子
                    x1 = x1 * img_width;
                    y1 = y1 * img_height;
                    x2 = x2 * img_width;
                    y2 = y2 * img_height;
                } else {
                    // 如果已经是像素坐标，但基于模型输入尺寸，需要应用缩放因子
                    x1 = x1 * scale_factor_.x;
                    y1 = y1 * scale_factor_.y;
                    x2 = x2 * scale_factor_.x;
                    y2 = y2 * scale_factor_.y;
                }
                
                // 计算中心点坐标和宽高
                float center_x = (x1 + x2) / 2.0f;
                float center_y = (y1 + y2) / 2.0f;
                float width = x2 - x1;
                float height = y2 - y1;
                
                // 检查尺寸有效性（能量机关可能有特殊尺寸，放宽限制）
                if (width <= 0 || height <= 0) {
                    continue;
                }
                
                // 能量机关特性：放宽过滤条件
                bool valid_size = (width > 10 && height > 10);  // 降低最小尺寸要求
                bool not_too_large = (width < img_width && height < img_height);  // 不要求太严格
                bool valid_ratio = (width / height < 10 && height / width < 10);  // 放宽宽高比限制
                
                if (valid_size && not_too_large && valid_ratio) {
                    // 使用锚框坐标创建检测框
                    cv::Rect box;
                    box.x = static_cast<int>(x1);
                    box.y = static_cast<int>(y1);
                    box.width = static_cast<int>(width);
                    box.height = static_cast<int>(height);
                    
                    // 确保不超出图像边界
                    box.x = std::max(0, box.x);
                    box.y = std::max(0, box.y);
                    if (box.x + box.width > img_width) 
                        box.width = img_width - box.x;
                    if (box.y + box.height > img_height) 
                        box.height = img_height - box.y;
                    
                    // 过滤小尺寸
                    if (box.width > 5 && box.height > 5) {
                        // 添加到候选列表
                        boxes.push_back(box);
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id);
                        
                        if (enable_debug_) {
                            // std::cout << "候选框: 类别=" << class_id 
                            //           << ", 置信度=" << confidence 
                            //           << ", 坐标=[" << box.x << "," << box.y << "," 
                            //           << box.width << "," << box.height << "]" << std::endl;
                        }
                    }
                }
            }
            
            // 标记已处理检测结果
            processed_detections = true;
        }
        else {
            std::cerr << "无法确定输出布局，尝试默认转置模式" << std::endl;
            transpose = true;
        }
    }
    
    // 如果还没处理过检测结果
    if (!processed_detections) {
        // 遍历所有预测框
        num_boxes = transpose ? rows : cols;
        
        for (size_t i = 0; i < num_boxes; i++) {
            float max_confidence = 0.0f;
            int best_class_id = -1;
            
            // 查找最高置信度的类别
            for (int class_idx = 0; class_idx < num_classes; class_idx++) {
                float class_score = 0.0f;
                
                if (transpose) {
                    // 输出格式为[8400, 84]
                    class_score = data[i * cols + (box_dims + class_idx)];
                } else {
                    // 输出格式为[1, 84, 8400]
                    class_score = data[(box_dims + class_idx) * cols + i];
                }
                
                if (class_score > max_confidence) {
                    max_confidence = class_score;
                    best_class_id = class_idx;
                }
            }
            
            // 降低置信度阈值以适应能量机关
            float adjusted_threshold = 0.4f;
            
            // 如果置信度低于阈值，跳过
            if (max_confidence < adjusted_threshold) {
                continue;
            }
            
            // 读取边界框坐标
            // YOLOv8输出是中心点坐标(x,y)和宽高(w,h)的归一化值或像素值
            float x, y, w, h;
            
            if (transpose) {
                // 输出格式为[8400, 84]
                x = data[i * cols + 0];
                y = data[i * cols + 1];
                w = data[i * cols + 2];
                h = data[i * cols + 3];
            } else {
                // 输出格式为[1, 84, 8400]
                x = data[0 * cols + i];
                y = data[1 * cols + i];
                w = data[2 * cols + i];
                h = data[3 * cols + i];
            }
            
            // 检查是否是归一化坐标(0-1范围)
            // 对于能量机关目标，大多数情况下应该是归一化坐标
            bool normalized = true;
            
            if (std::abs(x) > 1.0f || std::abs(y) > 1.0f || 
                std::abs(w) > 1.0f || std::abs(h) > 1.0f) {
                normalized = false;
            }
            
            // 转换坐标，确保正确应用缩放因子
            if (normalized) {
                // 如果是归一化坐标(0-1)，直接应用到原始图像尺寸
                // 这是关键修复：使用原始图像尺寸而不是模型输入尺寸*缩放因子
                x = x * img_width;
                y = y * img_height;
                w = w * img_width;
                h = h * img_height;
            } else {
                // 如果已经是像素坐标，但基于模型输入尺寸，需要应用缩放因子
                x = x * scale_factor_.x;
                y = y * scale_factor_.y;
                w = w * scale_factor_.x;
                h = h * scale_factor_.y;
            }
            
            // 计算左上角坐标(YOLO格式是中心点坐标和宽高)
            float x1 = x - w / 2;
            float y1 = y - h / 2;
            float x2 = x + w / 2;
            float y2 = y + h / 2;
            
            // 计算最终宽高
            float width = x2 - x1;
            float height = y2 - y1;
            
            // 过滤无效检测框
            if (width <= 0 || height <= 0) {
                continue;
            }
            
            // 能量机关特性：放宽过滤条件
            bool valid_size = (width > 10 && height > 10);  // 降低最小尺寸要求
            bool not_too_large = (width < img_width && height < img_height);  // 不要求太严格
            bool valid_ratio = (width / height < 10 && height / width < 10);  // 放宽宽高比限制
            
            if (valid_size && not_too_large && valid_ratio) {
                // 创建边界框
                cv::Rect box;
                box.x = static_cast<int>(x1);
                box.y = static_cast<int>(y1);
                box.width = static_cast<int>(width);
                box.height = static_cast<int>(height);
                
                // 确保在图像范围内
                box.x = std::max(0, box.x);
                box.y = std::max(0, box.y);
                if (box.x + box.width > img_width)
                    box.width = img_width - box.x;
                if (box.y + box.height > img_height)
                    box.height = img_height - box.y;
                
                if (box.width > 5 && box.height > 5) {
                    // 添加到候选列表
                    boxes.push_back(box);
                    confidences.push_back(max_confidence);
                    class_ids.push_back(best_class_id);
                    
                    if (enable_debug_) {
                        // std::cout << "候选框: 类别=" << best_class_id 
                        //          << ", 置信度=" << max_confidence 
                        //          << ", 坐标=[" << box.x << "," << box.y << "," 
                        //          << box.width << "," << box.height << "]" << std::endl;
                    }
                }
            }
        }
    }
    
    // 如果没有检测到物体，直接返回
    if (boxes.empty()) {
        if (enable_debug_) {
            // std::cout << "未检测到物体" << std::endl;
        }
        return detections;
    }
    
    // 调整NMS阈值，能量机关可能需要更严格的NMS进行去重
    float adjusted_nms_threshold = 0.2f;
    
    // 应用非极大值抑制(NMS)
    std::vector<int> indices;
    try {
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, adjusted_nms_threshold, indices);
    }
    catch (const cv::Exception& e) {
        std::cerr << "NMS过程发生异常: " << e.what() << std::endl;
        return detections;
    }
    
    if (enable_debug_) {
        // std::cout << "NMS后保留 " << indices.size() << " 个检测框" << std::endl;
    }
    
    // 限制最大返回检测数量
    const int max_detections = 2;  // 能量机关只有少量目标，限制结果数量
    int keep_count = std::min(static_cast<int>(indices.size()), max_detections);
    
    // 构建最终检测结果
    for (int i = 0; i < keep_count; i++) {
        int idx = indices[i];
        
        Detection detection;
        detection.box = boxes[idx];
        detection.confidence = confidences[idx];
        detection.class_id = class_ids[idx];
        
        if (enable_debug_) {
            // std::cout << "最终检测 #" << i << ": 类别=" << detection.class_id 
            //          << " (" << (detection.class_id < static_cast<int>(classes_.size()) ? classes_[detection.class_id] : "未知")
            //          << "), 置信度=" << detection.confidence 
            //          << ", 坐标=[" << detection.box.x << "," << detection.box.y 
            //          << "," << detection.box.width << "," << detection.box.height << "]" << std::endl;
        }
        
        // 如果是小框，尝试基于颜色扩展检测区域（针对能量机关）
        if (detection.box.width < 100 && detection.box.height < 100) {
            cv::Rect enlarged_box = enlargeDetectionArea(frame, detection.box);
            detection.box = enlarged_box;
            
            if (enable_debug_) {
                // std::cout << "扩展后的框: [" << enlarged_box.x << "," << enlarged_box.y 
                //           << "," << enlarged_box.width << "," << enlarged_box.height << "]" << std::endl;
            }
        }
        
        detections.push_back(detection);
    }
    
    return detections;
}

// 基于颜色特征扩展检测区域（特别适用于能量机关）
cv::Rect OpenvinoDetect::enlargeDetectionArea(const cv::Mat& frame, const cv::Rect& box) {
    // 确保框在图像范围内
    cv::Rect safe_box = box;
    safe_box.x = std::max(0, safe_box.x);
    safe_box.y = std::max(0, safe_box.y);
    safe_box.width = std::min(frame.cols - safe_box.x, safe_box.width);
    safe_box.height = std::min(frame.rows - safe_box.y, safe_box.height);
    
    // 获取框内区域
    cv::Mat roi = frame(safe_box);
    
    // 转换为HSV色彩空间
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    
    // 提取蓝色范围（能量机关通常为蓝色）
    cv::Mat blue_mask;
    cv::inRange(hsv, cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255), blue_mask);
    
    // 查找蓝色区域的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blue_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 如果没有找到轮廓，返回原始框
    if (contours.empty()) {
        return box;
    }
    
    // 找到最大的轮廓
    size_t largest_idx = 0;
    double largest_area = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            largest_idx = i;
        }
    }
    
    // 计算轮廓的边界矩形
    cv::Rect contour_box = cv::boundingRect(contours[largest_idx]);
    
    // 将轮廓的坐标转换回原图坐标系
    contour_box.x += safe_box.x;
    contour_box.y += safe_box.y;
    
    // 扩展原始框以包含蓝色区域
    cv::Rect enlarged_box;
    enlarged_box.x = std::min(box.x, contour_box.x);
    enlarged_box.y = std::min(box.y, contour_box.y);
    enlarged_box.width = std::max(box.x + box.width, contour_box.x + contour_box.width) - enlarged_box.x;
    enlarged_box.height = std::max(box.y + box.height, contour_box.y + contour_box.height) - enlarged_box.y;
    
    // 确保不超出图像边界
    enlarged_box.x = std::max(0, enlarged_box.x);
    enlarged_box.y = std::max(0, enlarged_box.y);
    enlarged_box.width = std::min(frame.cols - enlarged_box.x, enlarged_box.width);
    enlarged_box.height = std::min(frame.rows - enlarged_box.y, enlarged_box.height);
    
    return enlarged_box;
}

// 绘制检测结果
void OpenvinoDetect::drawDetections(cv::Mat &frame, const std::vector<Detection> &detections)
{
    for (const auto &det : detections) {
        try {
            // 确保类别ID在合法范围内
            if (det.class_id < 0 || det.class_id >= static_cast<int>(classes_.size())) {
                if (enable_debug_) {
                    // std::cout << "跳过绘制: 类别ID " << det.class_id << " 超出有效范围" << std::endl;
                }
                continue;
            }
            
            const cv::Rect &box = det.box;
            
            // 确保边界框在图像范围内
            if (box.x < 0 || box.y < 0 || box.width <= 0 || box.height <= 0 || 
                box.x + box.width > frame.cols || box.y + box.height > frame.rows) {
                if (enable_debug_) {
                    // std::cout << "跳过绘制: 边界框超出图像范围" << std::endl;
                }
                continue;
            }
            
            // 根据类别选择颜色
            cv::Scalar color;
            if (det.class_id == 0) {
                color = cv::Scalar(0, 255, 0);  // 类别0用绿色
            } else {
                color = cv::Scalar(0, 255, 255);  // 类别1用黄色
            }
            
            // 绘制边界框，使用更粗的线条
            cv::rectangle(frame, box, color, 3);
            
            // 创建标签文本
            std::string label = classes_[det.class_id] + " " + 
                               std::to_string(det.confidence).substr(0, 4);
            
            // 获取文本大小
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                               0.7, 2, &baseline);
            
            // 绘制文本背景
            cv::rectangle(frame, 
                         cv::Point(box.x, box.y - textSize.height - 10),
                         cv::Point(box.x + textSize.width, box.y),
                         color, -1);
            
            // 绘制文本
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        }
        catch (const cv::Exception& e) {
            if (enable_debug_) {
                // std::cout << "绘制过程发生异常: " << e.what() << std::endl;
            }
            continue;
        }
    }
}

}  // namespace rm_rune_detect
