#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <vector>

struct PoseEstimationParams {
  PoseEstimationParams() {
    ransac_iterations = 8192;
    ransac_error_thresh = 5.0;
    robust_kernel_width = 10.0;
  }

  int ransac_iterations;       ///< RANSAC iterations
  double ransac_error_thresh;  ///< RANSAC inlier threshold [pix]
  double robust_kernel_width;  ///< Robust kernel width for reprojection error
                               ///< minimization
};
/**
 * @brief Cost function for 3D-3D correspondences
 */
class TranformCost {
public:
  TranformCost(const Eigen::Vector3d &map_point3d, const Eigen::Vector3d &lidar_point3d)
  : map_point3d(map_point3d), lidar_point3d(lidar_point3d) {}
  template <typename T>
  bool operator()(const T *const T_map_lidar_params, T *residual) const {
    const Eigen::Map<Sophus::SE3<T> const> T_map_lidar(T_map_lidar_params);
    const Eigen::Matrix<T, 3, 1> pt_map = T_map_lidar * lidar_point3d;

    residual[0] = pt_map[0] - map_point3d[0];
    residual[1] = pt_map[1] - map_point3d[1];
    residual[2] = pt_map[2] - map_point3d[2];
    return true;
  }

private:
  const Eigen::Vector3d map_point3d;
  const Eigen::Vector3d lidar_point3d;
};

/**
 * @brief Pose estimation based on 3D-3D correspondences
 */
class PoseEstimation3D {
public:
  // explicit PoseEstimation3D(PoseEstimationParams const& params = PoseEstimationParams());

  Eigen::Isometry3d estimate(
    const std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> &correspondences,
    std::vector<bool> *inliers = nullptr);

private:
  Eigen::Matrix3d estimate_rotation_ransac(
    const std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> &correspondences,
    std::vector<bool> *inliers);
  Eigen::Isometry3d estimate_pose_lsq(
    const std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> &correspondences,
    const Eigen::Isometry3d &T_camera_lidar);

  const PoseEstimationParams params;
};
