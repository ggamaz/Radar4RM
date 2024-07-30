#ifndef RADAR_LOCALIZATION_HPP_
#define RADAR_LOCALIZATION_HPP_
// STD
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <queue>
// MSG
#include "radar_interfaces/msg/detection_array.hpp"
#include "radar_interfaces/msg/target_info_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <radar_interfaces/msg/client_map_receive_data.hpp>
#include <radar_interfaces/msg/game_robot_status.hpp>
#include <radar_interfaces/msg/radar_mark_data.hpp>
#include <radar_interfaces/srv/point_transform.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// ROS2
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>

namespace fyt::radar
{
struct CameraTransfrom
{
  Eigen::Isometry3d T_cam_lidar;
  Eigen::Matrix<double, 3, 3> K;
  Eigen::Matrix<double, 1, 5> C;
};

class RadarLocalization : public rclcpp::Node
{
public:
  explicit RadarLocalization(const rclcpp::NodeOptions& options);

private:
  const int BlueNum = 0;

  // subscribe radar point cloud. accumulate the last 5 pointcloud
  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr accumulate_pc_;
  std::queue<pcl::PointCloud<pcl::PointXYZ>> pc_queue_;
  const size_t accumulate_lenth = 5;  // accumulate the last 5 pointcloud

  // subscribe image detections
  rclcpp::Subscription<radar_interfaces::msg::DetectionArray>::SharedPtr detections_sub_;
  void detections_callback(const radar_interfaces::msg::DetectionArray::SharedPtr msg);

  // publish target info
  rclcpp::Publisher<radar_interfaces::msg::TargetInfoArray>::SharedPtr targets_pub_;

  // lidar to map transform
  tf2_ros::Buffer::SharedPtr tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
  void trans_callback();
  Eigen::Isometry3d T_map_lidar_;

  // subscribe camera info. get the camera intrinsic matrix and distortion coefficients
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  CameraTransfrom camera_transform_;

};

}  // namespace fyt::radar

#endif  // RADAR_LOCALIZATION_HPP_