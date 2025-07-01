#include "rm_armor_detect/video_subscriber_node.hpp"

#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<rm_armor_detect::VideoSubscriberNode>();
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
} 