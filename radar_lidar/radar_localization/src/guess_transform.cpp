
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/hovered_drawings.hpp>
#include <guik/model_control.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <common/estimate_pose.hpp>
#include <common/extract_points.hpp>
#include <common/raw_points.hpp>
#include <common/visual_lidar_data.hpp>
#include <common/visual_lidar_visualizer.hpp>

namespace vlcal
{
const std::vector<Eigen::Vector4d> RedMapPoints = {
  { 8917.72, 9286.8, 420.0, 0.0 },       // 红方R0左上
  { 8917.72, 8886.8, 420.0, 0.0 },       // 红方R0右上
  { 17209.22, 12578.66, 1159.35, 0.0 },  // 蓝方前哨站顶端
  // { 0.0, 1.0, 1.0, 0.0 },  //
};
const std::vector<Eigen::Vector4d> BlueMapPoints = {
  { 19582.68, 5713.2, 420.0, 0.0 },     // 蓝方B0左上
  { 19582.68, 6113.2, 420.0, 0.0 },     // 蓝方B0右上
  { 11294.08, 2421.34, 1159.35, 0.0 },  // 红方前哨站顶端
  // { 0.0, 1.0, 1.0, 0.0 },  //蓝方B0右下
};

class PickingPointCloud
{
public:
  PickingPointCloud()
  {
    //manual mode
    correspondences.first = RedMapPoints;
    // correspondences.first = BlueMapPoints; 
    correspondences.second.resize(RedMapPoints.size());
  }
  void add_point(const Eigen::Vector4d& pt)
  {
    guik::LightViewer::instance()->append_text(
        (boost::format("picked_3d: %.1f %.1f %.1f") % pt.x() % pt.y() % pt.z()).str());
    correspondences.second[index] = pt;
    index++;
  }
  void delete_point()
  {
    if (index > 0)
    {
      guik::LightViewer::instance()->append_text("3D point deleted");
      correspondences.second[index] = Eigen::Vector4d::Zero();
      index--;
    }
    else
    {
      guik::LightViewer::instance()->append_text("No 3D point to delete");
    }
  }
  std::optional<Eigen::Isometry3d> estimate()
  {
    if (index < correspondences.first.size() - 1)
    {
      guik::LightViewer::instance()->append_text("At least 3 correspondences are necessary!!");
      return std::nullopt;
    }

    PoseEstimation3D est;
    return est.estimate(correspondences);
  }
  size_t index = 0;
  std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> correspondences;
};

class InitialGuessManual
{
public:
  InitialGuessManual()
  {
    vis.reset(new VisualLiDARVisualizer(true, true));
    picking_point_cloud.reset(new PickingPointCloud());

    auto viewer = guik::LightViewer::instance();
    viewer->invoke([] {
      ImGui::SetNextWindowPos({ 55, 300 }, ImGuiCond_Once);
      ImGui::Begin("texts");
      ImGui::End();
      ImGui::SetNextWindowPos({ 55, 60 }, ImGuiCond_Once);
      ImGui::Begin("visualizer");
      ImGui::End();
      ImGui::SetNextWindowPos({ 55, 150 }, ImGuiCond_Once);
      ImGui::Begin("control");
      ImGui::End();
    });
    // ROS
    node = rclcpp::Node::make_shared("initial_guess_manual");
    broadcater_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    pc_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", qos, std::bind(&InitialGuessManual::pc_callback, this, std::placeholders::_1));
  }

  void spin()
  {
    Eigen::Isometry3d init_T_lidar_camera = Eigen::Isometry3d::Identity();
    init_T_lidar_camera.linear() =
        (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();

    auto viewer = guik::LightViewer::instance();
    guik::ModelControl T_lidar_camera_gizmo("T_lidar_camera", init_T_lidar_camera.matrix().cast<float>());
    viewer->register_ui_callback("gizmo", [&] {
      auto& io = ImGui::GetIO();
      if (!io.WantCaptureMouse && io.MouseClicked[1])
      {
        const float depth = viewer->pick_depth({ io.MousePos[0], io.MousePos[1] });
        if (depth > -1.0f && depth < 1.0f)
        {
          const Eigen::Vector3f pt_3d = viewer->unproject({ io.MousePos[0], io.MousePos[1] }, depth);
          picking_point_cloud->add_point(Eigen::Vector4d(pt_3d.x(), pt_3d.y(), pt_3d.z(), 1.0));

          guik::HoveredDrawings hovered;
          hovered.add_cross(pt_3d, IM_COL32(64, 64, 64, 255), 15.0f, 4.0f);
          hovered.add_cross(pt_3d, IM_COL32(0, 255, 0, 255), 15.0f, 3.0f);
          viewer->register_ui_callback("hovered", hovered.create_callback());
        }
      }

      ImGui::Begin("control", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

      if (ImGui::Button("Delete picked points"))
      {
        picking_point_cloud->delete_point();
      }

      if (ImGui::Button("Estimate"))
      {
        // const auto T_camera_lidar = picking->estimate();
        const auto T_map_lidar = picking_point_cloud->estimate();
        if (T_map_lidar.has_value())
        {
          geometry_msgs::msg::TransformStamped transform;
          transform.header.stamp = node->now();
          transform.header.frame_id = "map";
          transform.child_frame_id = "livox";
          // 提取平移部分
          auto eigen_transform = T_map_lidar.value();
          transform.transform.translation.x = eigen_transform.translation().x();
          transform.transform.translation.y = eigen_transform.translation().y();
          transform.transform.translation.z = eigen_transform.translation().z();
          // 提取旋转部分（四元数）
          Eigen::Quaterniond quat(eigen_transform.rotation());
          transform.transform.rotation.x = quat.x();
          transform.transform.rotation.y = quat.y();
          transform.transform.rotation.z = quat.z();
          transform.transform.rotation.w = quat.w();

          broadcater_->sendTransform(transform);
        }
        else
        {
          std::cout << "Estimation failed" << std::endl;
        }
      }

      ImGui::SameLine();
      ImGui::End();
    });

    while (vis->spin_once())
    {
      rclcpp::spin_some(node);
      cv::waitKey(1);
    }
  }

private:
  rclcpp::Node::SharedPtr node;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> broadcater_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    vlcal::RawPoints::Ptr raw_points = vlcal::extract_raw_points(*msg);
    auto points = std::make_shared<vlcal::FrameCPU>(raw_points->points);
    points->add_intensities(raw_points->intensities);
    if (vis)
    {
      vis->update_data(points);
    }
  };
  std::unique_ptr<VisualLiDARVisualizer> vis;
  std::unique_ptr<PickingPointCloud> picking_point_cloud;
};

}  // namespace vlcal

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  vlcal::InitialGuessManual init_guess;
  init_guess.spin();

  return 0;
}