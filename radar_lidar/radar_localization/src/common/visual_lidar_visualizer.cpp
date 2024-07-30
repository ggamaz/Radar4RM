#include <common/visual_lidar_visualizer.hpp>

#include <glk/primitives/primitives.hpp>
#include <glk/texture_opencv.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <opencv2/highgui.hpp>
namespace vlcal {

VisualLiDARVisualizer::VisualLiDARVisualizer(const bool draw_sphere,
                                             const bool show_image_cv)
    : draw_sphere(draw_sphere), show_image_cv(show_image_cv),
      T_camera_lidar(Eigen::Isometry3d::Identity()) {
  auto viewer = guik::LightViewer::instance();
  viewer->set_draw_xy_grid(false);
  viewer->use_arcball_camera_control();

  image_display_scale = 1.0;

  viewer->register_ui_callback("ui", [this]() { ui_callback(); });
}

void VisualLiDARVisualizer::ui_callback() {
  auto viewer = guik::LightViewer::instance();

  ImGui::Begin("visualizer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  if (ImGui::Button("Update") && points_buffer) {
    viewer->update_drawable("points", points_buffer, guik::Rainbow());

    if (draw_sphere) {
      viewer->update_drawable("sphere", glk::Primitives::sphere(),
                              guik::VertexColor());
    }
  }
  ImGui::End();
}

void VisualLiDARVisualizer::set_T_camera_lidar(
    const Eigen::Isometry3d &T_camera_lidar) {
  this->T_camera_lidar = T_camera_lidar;
}

bool VisualLiDARVisualizer::spin_once() {
  auto viewer = guik::LightViewer::instance();
  return viewer->spin_once();
}

} // namespace vlcal
