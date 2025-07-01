#include "video_trial_node/video_publisher_node.hpp"

#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<video_trial::VideoPublisherNode>();
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
} 