cmake_minimum_required(VERSION 3.8)
project(rm_armor_detect)

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
find_package(OpenCV REQUIRED)

# 设置OpenVINO路径
set(OpenVINO_DIR "$ENV{HOME}/openvino/runtime/cmake")
find_package(OpenVINO REQUIRED)

include_directories(include)

# 添加库文件，包含车辆检测器、灯条检测器、数字分类器和OpenVINO装甲板检测器
add_library(video_subscriber_component SHARED
  src/video_subscriber_node.cpp
  src/car_detector.cpp
  src/lights_detector.cpp
  src/number_classifier.cpp
  src/openvino_armor_detector.cpp
)
target_compile_definitions(video_subscriber_component
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
ament_target_dependencies(video_subscriber_component
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
)

# 链接OpenVINO库
target_link_libraries(video_subscriber_component
  openvino::runtime
)

rclcpp_components_register_node(video_subscriber_component
  PLUGIN "rm_armor_detect::VideoSubscriberNode"
  EXECUTABLE video_subscriber_node
)

# 添加独立可执行文件
add_executable(video_subscriber_main
  src/video_subscriber_main.cpp
)
ament_target_dependencies(video_subscriber_main
  rclcpp
  OpenCV
)
target_link_libraries(video_subscriber_main
  video_subscriber_component
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include/
)

# 安装库文件
install(TARGETS
  video_subscriber_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装可执行文件
install(TARGETS
  video_subscriber_main
  DESTINATION lib/${PROJECT_NAME}
)

# 安装模型文件
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models/
)

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

ament_package()
