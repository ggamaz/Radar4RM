cmake_minimum_required(VERSION 3.8)
project(radar_livox_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
add_definitions(-msse)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(PCL REQUIRED)
find_package(livox_sdk REQUIRED)


# livox ros2 driver target
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/lvx_file.cpp
  src/ldq.cpp
  src/lds.cpp
  src/lds_lvx.cpp
  src/lds_lidar.cpp
  src/lds_hub.cpp
  src/lddc.cpp
  src/radar_livox_driver.cpp
  timesync/timesync.cpp
  timesync/user_uart/user_uart.cpp
  common/comm/comm_protocol.cpp
  common/comm/sdk_protocol.cpp
  common/comm/gps_protocol.cpp
)

target_compile_options(${PROJECT_NAME}
  PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-field-initializers -fPIC>
)
# set(livox_sdk_INCLUDE_DIRS
#   livox_sdk_vendor
#   # livox_sdk_vendor/include/livox_def
#   # livox_sdk_vendor/include/livox_sdk
# )
target_include_directories(${PROJECT_NAME} PRIVATE ${livox_sdk_INCLUDE_DIRS})

# include file direcotry
target_include_directories(${PROJECT_NAME} PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${APR_INCLUDE_DIRS}
  common
  common/rapidjson
  comon/rapdidxml
  common/comm
  timesync
  timesync/user_uart
  radar_livox_driver
  # livox_sdk_vendor
)

# link libraries
target_link_libraries(${PROJECT_NAME}
	${livox_sdk_LIBRARIES}
	${Boost_LIBRARY}
	${PCL_LIBRARIES}
	${APR_LIBRARIES}
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "livox_ros::RadarLivoxDriver"
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
