#include "radar_localization/radar_localization.hpp"

#include <cv_bridge/cv_bridge.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/highgui.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace fyt::radar
{

RadarLocalization::RadarLocalization(const rclcpp::NodeOptions& options) : Node("radar_localization_node", options)
{
  //get the transform between map and lidar
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&RadarLocalization::trans_callback, this));

  // read the camera to lidar transform
  camera_transform_.T_cam_lidar = Eigen::Isometry3d::Identity();
  auto t = this->declare_parameter("T_camera_lidar", std::vector<double>{ 0, 0, 0, 0, 0, 0, 1 });
  camera_transform_.T_cam_lidar.translation() = Eigen::Vector3d(t[0], t[1], t[2]);
  camera_transform_.T_cam_lidar.linear() = Eigen::Quaterniond(t[6], t[3], t[4], t[5]).toRotationMatrix();

  // camrea info
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_info", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::CameraInfo::SharedPtr camera_info) {
        camera_transform_.K << camera_info->k[0], camera_info->k[1], camera_info->k[2], camera_info->k[3],
            camera_info->k[4], camera_info->k[5], camera_info->k[6], camera_info->k[7], camera_info->k[8];
        camera_transform_.C << camera_info->d[0], camera_info->d[1], camera_info->d[2], camera_info->d[3],
            camera_info->d[4];
        cam_info_sub_.reset();
      });

  // point cloud
  accumulate_pc_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  accumulate_pc_->points.resize(10000);
  auto qos = rclcpp::SensorDataQoS();
  radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "livox/lidar", qos, std::bind(&RadarLocalization::lidar_callback, this, std::placeholders::_1));

  // detect part
  detections_sub_ = this->create_subscription<radar_interfaces::msg::DetectionArray>(
      "detections", qos, std::bind(&RadarLocalization::detections_callback, this, std::placeholders::_1));

  // target info publisher
  targets_pub_ = this->create_publisher<radar_interfaces::msg::TargetInfoArray>("targets", qos);
  RCLCPP_INFO(this->get_logger(), "RadarLocalization has been started.");
}

// radar cloud accumulate
void RadarLocalization::lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  pcl::PointCloud<pcl::PointXYZ> pc;
  pcl::fromROSMsg(*msg, pc);
  pc_queue_.push(pc);
  if (pc_queue_.size() > accumulate_lenth)
  {
    auto last_pc = pc_queue_.front().makeShared();
    pc_queue_.pop();
    if (!last_pc->empty() && !accumulate_pc_->empty())
    {
      pcl::SegmentDifferences<pcl::PointXYZ> seg;
      seg.setInputCloud(accumulate_pc_->makeShared());
      seg.setTargetCloud(last_pc);
      seg.segment(*accumulate_pc_);
    }
  }
  *accumulate_pc_ += pc;
}

void RadarLocalization::detections_callback(const radar_interfaces::msg::DetectionArray::SharedPtr msg)
{
  radar_interfaces::msg::TargetInfoArray targets;
  targets.targets.resize(msg->detections.size());
  targets.header.stamp = msg->header.stamp;

  // for every detection process function
  auto target_process = [this](const radar_interfaces::msg::Detection& detection,
                               radar_interfaces::msg::TargetInfo& target) {
    if (detection.color != BlueNum)
      return;
    auto translation = camera_transform_.T_cam_lidar.translation();
    double img_x_tl = detection.bbox.top_left.x;  // unit: pixel
    double img_x_br = detection.bbox.bottom_right.x;
    double img_y_tl = detection.bbox.top_left.y;
    double img_y_br = detection.bbox.bottom_right.y;

    Eigen::Matrix<double, 3, 2> target_mat;
    target_mat << img_x_tl, img_x_br, img_y_tl, img_y_br, 1, 1;
    Eigen::Matrix<double, 3, 2> position_arrange =
        camera_transform_.T_cam_lidar.rotation().inverse() * camera_transform_.K.inverse() * target_mat;

    int count = 0;
    for (auto& radar_point : accumulate_pc_->points)
    {
      double z_cam = radar_point.x - translation.x();
      double radar_tl_y = z_cam * position_arrange(1, 0) + translation.y();
      double radar_tl_z = z_cam * position_arrange(2, 0) + translation.z();
      double radar_br_y = z_cam * position_arrange(1, 1) + translation.y();
      double radar_br_z = z_cam * position_arrange(2, 1) + translation.z();
      if (radar_point.y < radar_tl_y && radar_point.y > radar_br_y && radar_point.z < radar_tl_z &&
          radar_point.z > radar_br_z)
      {
        count++;
        target.x += (radar_point.x - target.x) / count;
        target.y += (radar_point.y - target.y) / count;
        target.z += (radar_point.z - target.z) / count;
      }
    }
    Eigen::Vector4d p_lidar(target.x, target.y, target.z, 1);
    Eigen::Vector4d p_map = T_map_lidar_ * p_lidar;

    Eigen::Vector3d p_map3d = p_map.head<3>() / p_map[3];
    target.x = p_map3d[0];
    target.y = p_map3d[1];
    target.z = p_map3d[2];
    target.class_name = detection.class_name;
    target.class_id = detection.class_id;
    target.color = detection.color;
  };

  // create threads
  std::vector<std::thread> threads;
  for (size_t i = 0; i < msg->detections.size(); i++)
  {
    threads.push_back(std::thread(target_process, msg->detections[i], std::ref(targets.targets[i])));
  }
  // 等待所有线程完成
  for (auto& thread : threads)
  {
    thread.join();
  }
  // 发布目标信息
  targets_pub_->publish(targets);
}

void RadarLocalization::trans_callback()
{
  geometry_msgs::msg::TransformStamped transform;
  try
  {
    transform = tf_buffer_->lookupTransform("map", "livox", tf2::TimePointZero);
    T_map_lidar_ = tf2::transformToEigen(transform);
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    RCLCPP_ERROR(this->get_logger(), "Can not find the transform between map and livox");
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    return;
  }
}

};  // namespace fyt::radar

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(fyt::radar::RadarLocalization)
