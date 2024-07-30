
#include "radar_localization/guess_transform.hpp"

namespace vlcal
{

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