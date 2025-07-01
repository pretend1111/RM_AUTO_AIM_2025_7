cmake_minimum_required(VERSION 3.8)
project(rm_rune_detect)

# 设置Release模式编译优化
set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -mtune=native)
endif()

# 设置OpenVINO的路径
set(OpenVINO_DIR "/opt/intel/openvino_2025/runtime/cmake")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenMP)

# 包含头文件目录
include_directories(include)
include_directories("/opt/intel/openvino_2025/runtime/include")

# 添加 ROS2 节点
add_executable(rune_detect_node 
  src/rune_detect_main.cpp
  src/rune_detect_node.cpp
  src/openvino_detect.cpp
  src/preprocess.cpp
)

# 添加库
add_library(${PROJECT_NAME} SHARED
  src/rune_detect_node.cpp
  src/openvino_detect.cpp
  src/preprocess.cpp
)

# 链接依赖库
target_link_libraries(rune_detect_node
  ${OpenCV_LIBRARIES}
  openvino::runtime
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  openvino::runtime
)

# 添加OpenMP支持（如果可用）
if(OpenMP_CXX_FOUND)
    target_link_libraries(rune_detect_node OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# 为可执行文件指定依赖项
ament_target_dependencies(rune_detect_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

# 为库指定依赖项
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

# 注册组件
rclcpp_components_register_nodes(${PROJECT_NAME} "rm_rune_detect::RuneDetectNode")

# 安装目标
install(TARGETS
  rune_detect_node
  ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include/
)

# 安装模型文件
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

# 安装启动文件（如果有）
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
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
