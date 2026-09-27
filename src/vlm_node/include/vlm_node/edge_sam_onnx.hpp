#pragma once

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <mutex>
#include <string>

namespace vlm_node {

struct EdgeSamResult {
  cv::Mat mask;
  float score{0.0F};
  int mask_pixels{0};
  double encoder_ms{0.0};
  double decoder_ms{0.0};
};

// CPU-only, box-prompted EdgeSAM inference. The returned mask always has the
// same dimensions as the source image, preserving organized RGB-D indexing.
class EdgeSamOnnx {
 public:
  EdgeSamOnnx(const std::string &model_directory, int cpu_threads);
  ~EdgeSamOnnx();

  EdgeSamResult segment(const cv::Mat &bgr_image, const cv::Rect &box);

 private:
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> encoder_;
  std::unique_ptr<Ort::Session> decoder_;
  std::mutex inference_mutex_;
};

}  // namespace vlm_node
