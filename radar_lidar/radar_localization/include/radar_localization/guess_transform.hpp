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

const std::vector<Eigen::Vector4d> MapPoints = {
  { 8917.72, 9286.8, 420.0, 0.0 },  //红方R0左上
  { 8917.72, 8886.8, 420.0, 0.0 },  //红方R0右上
  { 17209.22, 12578.66, 1159.35, 0.0 },  //蓝方前哨站顶端
  { 0.0, 1.0, 1.0, 0.0 },  //
  { 1.0, 0.0, 0.0, 0.0 },  //
  { 1.0, 0.0, 1.0, 0.0 },  //
  { 1.0, 1.0, 0.0, 0.0 },  //
  { 1.0, 1.0, 1.0, 0.0 },  //
};

class PickingPointCloud
{
public:
  PickingPointCloud()
  {
    correspondences.first = MapPoints;
    correspondences.second.resize(MapPoints.size());
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

}  // namespace vlcal
