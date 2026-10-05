#include "vlm_node/edge_sam_onnx.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace vlm_node {
namespace {
constexpr int kInputSize = 1024;

std::unique_ptr<Ort::Session> loadSession(
    Ort::Env &environment, Ort::SessionOptions &options,
    const std::filesystem::path &path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("EdgeSAM model missing: " + path.string());
  }
  return std::make_unique<Ort::Session>(
      environment, path.c_str(), options);
}
}  // namespace

EdgeSamOnnx::EdgeSamOnnx(
    const std::string &model_directory, const int cpu_threads)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "edgesam") {
  session_options_.SetIntraOpNumThreads(std::max(1, cpu_threads));
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  encoder_ = loadSession(
      environment_, session_options_,
      std::filesystem::path(model_directory) / "edge_sam_3x_encoder.onnx");
  decoder_ = loadSession(
      environment_, session_options_,
      std::filesystem::path(model_directory) / "edge_sam_3x_decoder.onnx");
}

EdgeSamOnnx::~EdgeSamOnnx() = default;

EdgeSamResult EdgeSamOnnx::segment(
    const cv::Mat &bgr_image, const cv::Rect &requested_box) {
  if (bgr_image.empty() || bgr_image.type() != CV_8UC3) {
    throw std::invalid_argument("EdgeSAM requires a non-empty CV_8UC3 image");
  }
  const cv::Rect image_bounds(0, 0, bgr_image.cols, bgr_image.rows);
  const cv::Rect box = requested_box & image_bounds;
  if (box.width < 2 || box.height < 2) {
    throw std::invalid_argument("EdgeSAM grounding box is empty");
  }

  std::lock_guard<std::mutex> lock(inference_mutex_);
  const float scale = static_cast<float>(kInputSize) /
      static_cast<float>(std::max(bgr_image.cols, bgr_image.rows));
  const int resized_width = std::min(
      kInputSize, static_cast<int>(bgr_image.cols * scale + 0.5F));
  const int resized_height = std::min(
      kInputSize, static_cast<int>(bgr_image.rows * scale + 0.5F));
  cv::Mat rgb, resized;
  cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(resized_width, resized_height),
             0.0, 0.0, cv::INTER_LINEAR);

  constexpr std::array<float, 3> mean{123.675F, 116.28F, 103.53F};
  constexpr std::array<float, 3> standard_deviation{58.395F, 57.12F, 57.375F};
  std::vector<float> input(
      static_cast<size_t>(3) * kInputSize * kInputSize, 0.0F);
  const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
  for (int y = 0; y < resized_height; ++y) {
    const auto *row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < resized_width; ++x) {
      const size_t offset = static_cast<size_t>(y) * kInputSize + x;
      for (int channel = 0; channel < 3; ++channel) {
        input[static_cast<size_t>(channel) * plane + offset] =
            (static_cast<float>(row[x][channel]) - mean[channel]) /
            standard_deviation[channel];
      }
    }
  }

  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 4> image_shape{1, 3, kInputSize, kInputSize};
  auto image_tensor = Ort::Value::CreateTensor<float>(
      memory, input.data(), input.size(), image_shape.data(), image_shape.size());
  const char *encoder_inputs[] = {"image"};
  const char *encoder_outputs[] = {"image_embeddings"};
  const auto encoder_started = std::chrono::steady_clock::now();
  auto embeddings = encoder_->Run(
      Ort::RunOptions{nullptr}, encoder_inputs, &image_tensor, 1,
      encoder_outputs, 1);
  const auto encoder_finished = std::chrono::steady_clock::now();

  // EdgeSAM/SAM uses two special box points with labels 2 and 3.
  std::array<float, 4> coordinates{
      box.x * scale, box.y * scale,
      (box.x + box.width - 1) * scale,
      (box.y + box.height - 1) * scale};
  std::array<float, 2> labels{2.0F, 3.0F};
  const std::array<int64_t, 3> coordinate_shape{1, 2, 2};
  const std::array<int64_t, 2> label_shape{1, 2};
  auto coordinate_tensor = Ort::Value::CreateTensor<float>(
      memory, coordinates.data(), coordinates.size(),
      coordinate_shape.data(), coordinate_shape.size());
  auto label_tensor = Ort::Value::CreateTensor<float>(
      memory, labels.data(), labels.size(), label_shape.data(), label_shape.size());
  std::array<Ort::Value, 3> decoder_inputs_values{
      std::move(embeddings.front()), std::move(coordinate_tensor),
      std::move(label_tensor)};
  const char *decoder_inputs[] = {
      "image_embeddings", "point_coords", "point_labels"};
  const char *decoder_outputs[] = {"scores", "masks"};
  const auto decoder_started = std::chrono::steady_clock::now();
  auto outputs = decoder_->Run(
      Ort::RunOptions{nullptr}, decoder_inputs, decoder_inputs_values.data(),
      decoder_inputs_values.size(), decoder_outputs, 2);
  const auto decoder_finished = std::chrono::steady_clock::now();

  const auto score_shape =
      outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto mask_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  if (score_shape.size() != 2 || mask_shape.size() != 4 ||
      score_shape[0] != 1 || mask_shape[0] != 1 ||
      score_shape[1] != mask_shape[1]) {
    throw std::runtime_error("Unexpected EdgeSAM decoder output shape");
  }
  const int mask_count = static_cast<int>(score_shape[1]);
  const int low_height = static_cast<int>(mask_shape[2]);
  const int low_width = static_cast<int>(mask_shape[3]);
  const float *scores = outputs[0].GetTensorData<float>();
  const float *masks = outputs[1].GetTensorData<float>();
  int best = 0;
  for (int index = 1; index < mask_count; ++index) {
    if (scores[index] > scores[best]) best = index;
  }
  const size_t mask_plane = static_cast<size_t>(low_width) * low_height;
  cv::Mat low_mask(low_height, low_width, CV_32FC1,
                   const_cast<float *>(masks + best * mask_plane));
  cv::Mat padded_mask, resized_mask, original_mask;
  cv::resize(low_mask, padded_mask, cv::Size(kInputSize, kInputSize),
             0.0, 0.0, cv::INTER_LINEAR);
  cv::resize(padded_mask(cv::Rect(0, 0, resized_width, resized_height)),
             resized_mask, bgr_image.size(), 0.0, 0.0, cv::INTER_LINEAR);
  cv::compare(resized_mask, 0.0, original_mask, cv::CMP_GT);

  // A box prompt identifies the object, and clipping prevents EdgeSAM from
  // absorbing nearby objects/background beyond RoboBrain's semantic target.
  cv::Mat box_mask(bgr_image.size(), CV_8UC1, cv::Scalar(0));
  cv::rectangle(box_mask, box, cv::Scalar(255), cv::FILLED);
  cv::bitwise_and(original_mask, box_mask, original_mask);

  // Suppress disconnected decoder speckles while retaining the largest
  // complete component (including thin handles and rims).
  cv::Mat component_labels, stats, centroids;
  const int component_count = cv::connectedComponentsWithStats(
      original_mask, component_labels, stats, centroids, 8, CV_32S);
  if (component_count > 1) {
    int largest_label = 1;
    for (int label = 2; label < component_count; ++label) {
      if (stats.at<int>(label, cv::CC_STAT_AREA) >
          stats.at<int>(largest_label, cv::CC_STAT_AREA)) {
        largest_label = label;
      }
    }
    cv::compare(component_labels, largest_label, original_mask, cv::CMP_EQ);
  }

  EdgeSamResult result;
  result.mask = original_mask;
  result.score = scores[best];
  result.mask_pixels = cv::countNonZero(result.mask);
  result.encoder_ms = std::chrono::duration<double, std::milli>(
      encoder_finished - encoder_started).count();
  result.decoder_ms = std::chrono::duration<double, std::milli>(
      decoder_finished - decoder_started).count();
  return result;
}

}  // namespace vlm_node
