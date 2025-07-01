#include "rm_rune_detect/rune_detect_node.hpp"

#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<rm_rune_detect::RuneDetectNode>();
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
} 