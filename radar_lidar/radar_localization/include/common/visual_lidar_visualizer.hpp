#pragma once

#include "glk/pointcloud_buffer.hpp"
#include <common/visual_lidar_data.hpp>
namespace vlcal {

/**
 * @brief A class to visualize LiDAR-camera dataset while painting LiDAR points with camera images
 * @note  What a poor class name!
 */
class VisualLiDARVisualizer {
public:
  VisualLiDARVisualizer(
    const bool draw_sphere,
    const bool show_image_cv = false);

  void set_T_camera_lidar(const Eigen::Isometry3d& T_camera_lidar);

  double get_image_display_scale() const { return image_display_scale; }

  bool spin_once();

  void update_data(const FrameCPU::Ptr& points){
    // updated_data.reset(new VisualLiDARData(image, points));
    points_buffer = std::make_shared<glk::PointCloudBuffer>(points->points, points->size());
  }
  VisualLiDARData::ConstPtr updated_data;
private:
  void ui_callback();
  void color_update_task();

private:
  const bool draw_sphere;
  const bool show_image_cv;

  double image_display_scale;
  Eigen::Isometry3d T_camera_lidar;

  glk::PointCloudBuffer::Ptr points_buffer;

};

}  // namespace vlcal
