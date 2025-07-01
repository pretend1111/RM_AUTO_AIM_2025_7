#pragma once
// OpenCV
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

#include "rm_armor_detect/armor_types.hpp"

namespace rm_armor_detect
{

class NumberClassifier
{
public:
  NumberClassifier(
    const std::string & model_path, const std::string & label_path, const double threshold,
    const std::vector<std::string> & ignore_classes = {});

  void extractNumbers(const cv::Mat & src, std::vector<Armor> & armors);

  void classify(std::vector<Armor> & armors);

  cv::Mat numberlcassfy_helper(cv::Mat & number_image);

  double get_weights_parameter(cv::Mat & number_image);

  double threshold;

private:
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
};

} // namespace rm_armor_detect 