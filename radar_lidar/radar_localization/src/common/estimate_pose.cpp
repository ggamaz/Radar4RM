#include <random>
#include <common/estimate_pose.hpp>
#include <sophus/ceres_manifold.hpp>


#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <omp.h>



Eigen::Isometry3d PoseEstimation3D::estimate(
  const std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> &correspondences,
  std::vector<bool> *inliers) {
  // RANSAC
  Eigen::Isometry3d T_camera_lidar = Eigen::Isometry3d::Identity();
  T_camera_lidar.linear() = estimate_rotation_ransac(correspondences, inliers);

  std::cout << "--- T_camera_lidar (RANSAC) ---" << std::endl;
  std::cout << T_camera_lidar.matrix() << std::endl;
  // Reprojection error minimization
  T_camera_lidar = estimate_pose_lsq(correspondences, T_camera_lidar);

  std::cout << "--- T_camera_lidar (LSQ) ---" << std::endl;
  std::cout << T_camera_lidar.matrix() << std::endl;

  return T_camera_lidar;
}

Eigen::Matrix3d PoseEstimation3D::estimate_rotation_ransac(
  const std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> &correspondences,
  std::vector<bool> *inliers) {
  std::cout << "estimating bearing vectors" << std::endl;

  // Compute bearing vectors
  std::vector<Eigen::Vector4d> directions_map = correspondences.first;
  std::vector<Eigen::Vector4d> directions_lidar = correspondences.second;

  // LSQ rotation estimation
  // https://web.stanford.edu/class/cs273/refs/umeyama.pdf
  const auto find_rotation = [&](const std::vector<int> &indices) {
    Eigen::Matrix<double, 3, -1> A(3, indices.size());
    Eigen::Matrix<double, 3, -1> B(3, indices.size());

    for (size_t i = 0; i < indices.size(); i++) {
      const int index = indices[i];
      const auto &d_m = directions_map[index];
      const auto &d_l = directions_lidar[index];

      A.col(i) = d_m.head<3>();
      B.col(i) = d_l.head<3>();
    }

    const Eigen::Matrix3d AB = A * B.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(AB, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();
    // const Eigen::Matrix3d D = svd.singularValues().asDiagonal();
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();

    double det = U.determinant() * V.determinant();
    if (det < 0.0) {
      S(2, 2) = -1.0;
    }

    const Eigen::Matrix3d R_map_lidar = U * S * V.transpose();
    return R_map_lidar;
  };

  const double error_thresh_sq = std::pow(params.ransac_error_thresh, 2);

  int best_num_inliers = 0;
  Eigen::Matrix4d best_R_map_lidar;

  std::mt19937 mt;
  std::vector<std::mt19937> mts(omp_get_max_threads());
  for (size_t i = 0; i < mts.size(); i++) {
    mts[i] = std::mt19937(mt() + 8192 * i);
  }

  const int num_samples = 2;

  std::cout << "estimating rotation using RANSAC" << std::endl;
#pragma omp parallel for
  for (int i = 0; i < params.ransac_iterations; i++) {
    const int thread_id = omp_get_thread_num();

    // Sample correspondences
    std::vector<int> indices(num_samples);
    std::uniform_int_distribution<> udist(0, correspondences.first.size() - 1);
    for (int i = 0; i < num_samples; i++) {
      indices[i] = udist(mts[thread_id]);
    }

    // Estimate rotation
    Eigen::Matrix4d R_map_lidar = Eigen::Matrix4d::Zero();
    R_map_lidar.topLeftCorner<3, 3>() = find_rotation(indices);

    // Count num of inliers
    int num_inliers = 0;
    for (size_t j = 0; j < directions_lidar.size(); j++) {
      const Eigen::Vector4d direction_map = R_map_lidar * directions_lidar[j];

      if ((correspondences.first[j] - direction_map).squaredNorm() < error_thresh_sq) {
        num_inliers++;
      }
    }

#pragma omp critical
    if (num_inliers > best_num_inliers) {
      // Update the best rotation
      best_num_inliers = num_inliers;
      best_R_map_lidar = R_map_lidar;
    }
  }

  std::cout << "num_inliers: " << best_num_inliers << " / " << correspondences.first.size()
            << std::endl;

  if (inliers) {
    inliers->resize(correspondences.first.size());
    for (size_t  i = 0; i < correspondences.first.size(); i++) {
      const Eigen::Vector4d direction_map = best_R_map_lidar * directions_lidar[i];
      (*inliers)[i] = (correspondences.first[i] - direction_map).squaredNorm() < error_thresh_sq;
    }
  }
  std::cout<<"solve rotation done"<<std::endl;
  return best_R_map_lidar.block<3, 3>(0, 0);
}

Eigen::Isometry3d PoseEstimation3D::estimate_pose_lsq(
  const std::pair<std::vector<Eigen::Vector4d>, std::vector<Eigen::Vector4d>> &correspondences,
  const Eigen::Isometry3d &init_T_map_lidar) {
  auto map_pts = correspondences.first;
  auto lidar_pts = correspondences.second;
  auto size = map_pts.size();

  Sophus::SE3d T_map_lidar = Sophus::SE3d(init_T_map_lidar.matrix());

  ceres::Problem problem;
  problem.AddParameterBlock(
    T_map_lidar.data(), Sophus::SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());
  // Create reprojection error costs
  for (size_t i = 0; i < size; ++i) {
    auto proj_error = new TranformCost(map_pts[i].head<3>(), lidar_pts[i].head<3>());
    auto ad_cost =
      new ceres::AutoDiffCostFunction<TranformCost, 3, Sophus::SE3d::num_parameters>(proj_error);
    auto loss = new ceres::CauchyLoss(params.robust_kernel_width);
    problem.AddResidualBlock(ad_cost, loss, T_map_lidar.data());
  }

  // Solve!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  return Eigen::Isometry3d(T_map_lidar.matrix());
}