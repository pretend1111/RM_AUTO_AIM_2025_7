cmake_minimum_required(VERSION 3.8)
project(video_trial_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 包含头文件目录
include_directories(include)

# 添加可执行文件
add_executable(video_publisher_node
  src/video_publisher_main.cpp
  src/video_publisher_node.cpp
)

# 添加库（用于组件加载）
add_library(${PROJECT_NAME} SHARED
  src/video_publisher_node.cpp
)

# 链接OpenCV库
target_link_libraries(video_publisher_node
  ${OpenCV_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

# 添加依赖
ament_target_dependencies(video_publisher_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  camera_info_manager
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  camera_info_manager
)

# 注册ROS组件
rclcpp_components_register_nodes(${PROJECT_NAME} 
  "video_trial::VideoPublisherNode"
)

# 安装
install(TARGETS
  video_publisher_node
  ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
