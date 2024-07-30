#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <common/frame_cpu.hpp>

namespace vlcal {

struct VisualLiDARData {
public:
  using Ptr = std::shared_ptr<VisualLiDARData>;
  using ConstPtr = std::shared_ptr<const VisualLiDARData>;

  VisualLiDARData(const cv::Mat& image, const FrameCPU::Ptr& points) : image(image), points(points) {}
  
public:
  cv::Mat image;
  FrameCPU::Ptr points;
};

}  // namespace vlcal
