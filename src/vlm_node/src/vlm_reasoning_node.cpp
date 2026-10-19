#include <my_actions/srv/verify_presence.hpp>
#include <my_actions/srv/cache_grasp_plan.hpp>
#include <my_actions/srv/get_cached_grasp.hpp>
#include <my_actions/srv/ground_object.hpp>
#include <my_actions/srv/resolve_manipulation_target.hpp>
#include <my_actions/srv/vision_inference.hpp>
#include <my_actions/srv/capture_observation.hpp>
#include "vlm_node/layout_prompt.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <my_actions/srv/chat.hpp>
#include <my_actions/srv/get_yolo_inference.hpp>
#include <my_actions/srv/append_task.hpp>
#include <my_actions/srv/execute_pipeline.hpp>
#include <my_actions/srv/find_grasp_candidates.hpp>
#include <my_actions/srv/find_placement_candidates.hpp>
#include <my_actions/srv/commit_floor_slots.hpp>
#include <my_actions/srv/plan_placement.hpp>
#include <my_actions/srv/plan_object_layout.hpp>
#include <my_actions/srv/pin_layout_observation.hpp>
#include <my_actions/srv/get3_d_coordinate.hpp>
#include <my_actions/srv/update_object_observation.hpp>
#include <my_actions/srv/resolve_object.hpp>
#include <my_actions/srv/look_at.hpp>
#include <my_actions/srv/execute_grasp.hpp>
#include <my_actions/srv/segment_grounded_image.hpp>
#include <my_actions/srv/get_world_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <Eigen/Dense>
#include "vlm_node/json.hpp"
#include "vlm_node/httplib.h"
#include "vlm_node/edge_sam_onnx.hpp"
#include "vlm_node/grounding.hpp"
#include "vlm_node/inference_capture.hpp"
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <regex>
#include <future>
#include <iterator>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <sstream>
#include <unordered_map>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include <set>

using json = nlohmann::json;

struct DetectedObject {
    std::string name;
    std::string description;
    int u;
    int v;
    float x;
    float y;
    float z;
    cv::Rect bbox;
};

struct BoundingBox {
    int x1, y1, x2, y2;
    int class_id;
    float conf;
};

struct PromptRequest {
    std::string mode = "GENERAL";
    std::string prompt;
    std::string image_path;
};

struct SynchronizedObservation {
    cv::Mat image;
    std_msgs::msg::Header header;
    sensor_msgs::msg::PointCloud2 cloud;
    geometry_msgs::msg::TransformStamped camera_transform;
};

const std::vector<std::string> COCO_CLASSES = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
  "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

class VLMReasoningNode : public rclcpp::Node {
public:
    explicit VLMReasoningNode(
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("vlm_reasoning_node", options) {
        this->declare_parameter<std::string>("api_server_ip", "10.10.10.2");
        this->declare_parameter<int>("api_server_port", 18091);
        this->declare_parameter<std::string>(
            "api_server_url", "");
        this->declare_parameter<std::string>("model_name", "robobrain2.5-4b");
        this->declare_parameter<int>("request_timeout_sec", 120);
        this->declare_parameter<std::string>("placement_support_mode", "floor_slots");
        this->declare_parameter<bool>("enable_task_dispatch", false);
        this->declare_parameter<bool>("verbose", true);
        this->declare_parameter<double>("temperature", 0.0);
        this->declare_parameter<int>("max_tokens", 256);
        this->declare_parameter<int>("image_max_dimension", 1024);
        this->declare_parameter<std::vector<std::string>>(
            "tracked_objects", {"brown cup", "red juice box"});
        this->declare_parameter<std::string>(
            "world_perception_backend", "robobrain");
        this->declare_parameter<std::string>(
            "grasp_target_mask_backend", "edgesam");
        this->declare_parameter<std::string>(
            "edgesam_model_directory",
            ament_index_cpp::get_package_share_directory("vlm_node") +
                "/models/edgesam");
        this->declare_parameter<int>("edgesam_cpu_threads", 2);
        this->declare_parameter<int>("edgesam_min_mask_pixels", 80);
        this->declare_parameter<bool>("enable_view_repositioning", false);
        this->declare_parameter<int>("grasp_plan_cache_ttl_sec", 60);
        
        api_ip_ = this->get_parameter("api_server_ip").as_string();
        api_port_ = this->get_parameter("api_server_port").as_int();
        api_server_url_ = this->get_parameter("api_server_url").as_string();
        configureApiEndpoint();
        model_name_ = this->get_parameter("model_name").as_string();
        request_timeout_sec_ = std::max<int64_t>(
            1, this->get_parameter("request_timeout_sec").as_int());
        enable_task_dispatch_ = this->get_parameter("enable_task_dispatch").as_bool();
        verbose_ = this->get_parameter("verbose").as_bool();
        temperature_ = this->get_parameter("temperature").as_double();
        max_tokens_ = std::max<int64_t>(
            1, this->get_parameter("max_tokens").as_int());
        image_max_dimension_ = std::max<int64_t>(
            64, this->get_parameter("image_max_dimension").as_int());
        tracked_objects_ = this->get_parameter("tracked_objects").as_string_array();
        world_perception_backend_ =
            this->get_parameter("world_perception_backend").as_string();
        grasp_target_mask_backend_ =
            this->get_parameter("grasp_target_mask_backend").as_string();
        std::transform(grasp_target_mask_backend_.begin(),
            grasp_target_mask_backend_.end(), grasp_target_mask_backend_.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (grasp_target_mask_backend_ != "edgesam") {
            throw std::invalid_argument("grasp_target_mask_backend only supports edgesam");
        }
        edgesam_min_mask_pixels_ = std::max<int64_t>(
            1, this->get_parameter("edgesam_min_mask_pixels").as_int());
        if (grasp_target_mask_backend_ == "edgesam") {
            try {
                edgesam_ = std::make_unique<vlm_node::EdgeSamOnnx>(
                    this->get_parameter("edgesam_model_directory").as_string(),
                    static_cast<int>(std::max<int64_t>(
                        1, this->get_parameter("edgesam_cpu_threads").as_int())));
                RCLCPP_INFO(this->get_logger(),
                            "EdgeSAM CPU box segmentation ready.");
            } catch (const std::exception &error) {
                edgesam_initialization_error_ = error.what();
                RCLCPP_ERROR(this->get_logger(),
                             "EdgeSAM initialization failed: %s",
                             error.what());
            }
        }
        std::transform(
            world_perception_backend_.begin(), world_perception_backend_.end(),
            world_perception_backend_.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (world_perception_backend_ != "yolo" &&
            world_perception_backend_ != "robobrain") {
            RCLCPP_WARN(
                this->get_logger(),
                "Unknown world_perception_backend '%s'; using robobrain.",
                world_perception_backend_.c_str());
            world_perception_backend_ = "robobrain";
        }
        enable_view_repositioning_ =
            this->get_parameter("enable_view_repositioning").as_bool();
        grasp_plan_cache_ttl_sec_ = static_cast<int>(std::clamp<int64_t>(
            this->get_parameter("grasp_plan_cache_ttl_sec").as_int(), 1, 60));
        RCLCPP_INFO(
            this->get_logger(),
            "RoboBrain endpoint: %s (model=%s, timeout=%ds, grounding uses original image; other image_limit=%dpx, "
            "task dispatch=%s, verbose=%s)",
            api_server_url_.c_str(), model_name_.c_str(), request_timeout_sec_,
            image_max_dimension_,
            enable_task_dispatch_ ? "enabled" : "disabled",
            verbose_ ? "enabled" : "disabled");


        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 1,
            std::bind(&VLMReasoningNode::imageCallback, this, std::placeholders::_1));
        observation_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/active_vision/observation_image",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&VLMReasoningNode::observationImageCallback, this, std::placeholders::_1));

        cache_grasp_client_ = this->create_client<my_actions::srv::CacheGraspPlan>(
            "/pipeline/cache_grasp_plan", rclcpp::ServicesQoS(), cb_group_);
        get_cached_grasp_client_ = this->create_client<my_actions::srv::GetCachedGrasp>(
            "/pipeline/get_cached_grasp", rclcpp::ServicesQoS(), cb_group_);
        verify_presence_srv_ = this->create_service<my_actions::srv::VerifyPresence>(
            "~/verify_presence",
            [this](const std::shared_ptr<my_actions::srv::VerifyPresence::Request> request,
                   std::shared_ptr<my_actions::srv::VerifyPresence::Response> response) {
                try {
                    const auto image = cv_bridge::toCvCopy(request->image, "bgr8")->image;
                    response->text = callVLMBackend("", prepareGroundingPresencePrompt(request->target),
                                                   16, image);
                    response->present = presenceResponseIsAffirmative(response->text);
                    response->success = response->present;
                    response->status = response->present ? "PRESENT" : "NOT_CONFIRMED_PRESENT";
                } catch (const std::exception &error) { response->status = error.what(); }
            }, rclcpp::ServicesQoS(), cb_group_);
        active_target_client_ = this->create_client<my_actions::srv::ResolveManipulationTarget>(
            "/pipeline/resolve_target", rclcpp::ServicesQoS(), cb_group_);
        ground_object_client_ = this->create_client<my_actions::srv::GroundObject>(
            "/pipeline/ground_object", rclcpp::ServicesQoS(), cb_group_);
        ground_object_srv_ = this->create_service<my_actions::srv::GroundObject>(
            "~/ground_object",
            [this](const std::shared_ptr<my_actions::srv::GroundObject::Request> request,
                   std::shared_ptr<my_actions::srv::GroundObject::Response> response) {
                try {
                    const auto image = cv_bridge::toCvCopy(request->image, "bgr8")->image;
                    if (request->verify_presence) {
                        // Legacy native callers retain their original optional
                        // check. The shared master flattens this into separately
                        // measured verify_presence and ground_object operations.
                        const auto presence = callVLMBackend("", prepareGroundingPresencePrompt(request->target),
                                                            16, image);
                        if (!presenceResponseIsAffirmative(presence)) {
                            response->status = "NOT_FOUND_OR_PRESENCE_FAILED: " + presence;
                            return;
                        }
                    }
                    response->text = callVLMBackend("", prepareOfficialPrompt("GROUNDING", vlm_node::wholeObjectGroundingTarget(request->target)),
                        48, image, request->sent_image_path);
                    std::smatch match;
                    if (!vlm_node::parseGroundingBox(response->text, match)) {
                        response->status = "GROUNDING_PARSE_FAILED: " + response->text;
                        return;
                    }
                    for (size_t i = 0; i < 4; ++i) {
                        response->normalized_box[i] = std::stoi(match[i+1]);
                        response->pixel_box[i] = normalizedToPixel(
                            response->normalized_box[i], i % 2 == 0 ? image.cols : image.rows);
                    }
                    response->success = true;
                    response->status = "GROUNDING_READY";
                    {
                        std::lock_guard<std::mutex> lock(capture_mutex_);
                        const auto role=vlm_node::captureRole(request->target);
                        capture_associations_.remember(captureFrame(request->image),response->pixel_box,role);
                        const auto directory=graspCaptureDirectory();
                        if (!directory.empty()) {
                            saveCaptureImage(directory/(role+"_input.png"),image);
                            auto boxed=image.clone();
                            const auto &b=response->pixel_box;
                            cv::rectangle(boxed,cv::Point(b[0],b[1]),cv::Point(b[2],b[3]),cv::Scalar(0,255,0),2);
                            saveCaptureImage(directory/(role+"_bounding_box.png"),boxed);
                        }
                    }
                } catch (const std::exception &error) {
                    response->status = error.what();
                }
            }, rclcpp::ServicesQoS(), cb_group_);
        vision_inference_client_ = this->create_client<my_actions::srv::VisionInference>(
            "/pipeline/infer_vision", rclcpp::ServicesQoS(), cb_group_);
        capture_observation_client_ = this->create_client<my_actions::srv::CaptureObservation>(
            "/pipeline/capture_observation", rclcpp::ServicesQoS(), cb_group_);
        segment_object_client_ = this->create_client<my_actions::srv::SegmentGroundedImage>(
            "/pipeline/segment_object", rclcpp::ServicesQoS(), cb_group_);
        vision_inference_srv_ = this->create_service<my_actions::srv::VisionInference>(
            "~/infer_vision",
            [this](const std::shared_ptr<my_actions::srv::VisionInference::Request> request,
                   std::shared_ptr<my_actions::srv::VisionInference::Response> response) {
                try {
                    cv::Mat image;
                    if (request->image.width > 0) {
                        image = cv_bridge::toCvCopy(request->image, "bgr8")->image;
                    }
                    response->text = callVLMBackend(request->system_prompt, request->user_prompt,
                        request->max_tokens, image, request->sent_image_path);
                    response->success = !response->text.empty();
                    response->status = response->success ? "OK" : "VLM_REQUEST_FAILED";
                } catch (const std::exception &error) {
                    response->success = false;
                    response->status = error.what();
                }
            }, rclcpp::ServicesQoS(), cb_group_);

        chat_srv_ = this->create_service<my_actions::srv::Chat>(
            "~/chat_prompt",
            std::bind(&VLMReasoningNode::handleChatPrompt, this, std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS(),
            cb_group_);
            
        segment_srv_ = this->create_service<my_actions::srv::SegmentGroundedImage>(
            "~/segment_grounded_image",
            [this](const std::shared_ptr<my_actions::srv::SegmentGroundedImage::Request> request,
                   std::shared_ptr<my_actions::srv::SegmentGroundedImage::Response> response) {
                try {
                    if (!edgesam_) throw std::runtime_error("EDGESAM_UNAVAILABLE");
                    const auto image = cv_bridge::toCvCopy(request->image, "bgr8")->image;
                    const auto &b = request->bbox;
                    if (b[0] < 0 || b[1] < 0 || b[2] > image.cols || b[3] > image.rows ||
                        b[0] >= b[2] || b[1] >= b[3]) {
                        throw std::runtime_error("INVALID_SEGMENTATION_BOX");
                    }
                    const auto result = edgesam_->segment(
                        image, cv::Rect(b[0], b[1], b[2]-b[0], b[3]-b[1]));
                    if (result.mask_pixels < edgesam_min_mask_pixels_) {
                        throw std::runtime_error("INSUFFICIENT_SEGMENTATION_PIXELS");
                    }
                    response->mask = *cv_bridge::CvImage(
                        request->image.header, "mono8", result.mask).toImageMsg();
                    response->score = result.score;
                    response->mask_pixels = result.mask_pixels;
                    response->encoder_ms = result.encoder_ms;
                    response->decoder_ms = result.decoder_ms;
                    response->success = true;
                    response->status = "OK";
                    {
                        std::lock_guard<std::mutex> lock(capture_mutex_);
                        const auto role=capture_associations_.find(captureFrame(request->image),request->bbox);
                        const auto directory=graspCaptureDirectory();
                        if (!role.empty() && !directory.empty()) {
                            cv::Mat segmented=cv::Mat::zeros(image.size(),image.type());
                            image.copyTo(segmented,result.mask);
                            saveCaptureImage(directory/(role+"_segmented.png"),segmented);
                            saveCaptureImage(directory/(role+"_mask.png"),result.mask);
                        }
                    }
                } catch (const std::exception &error) {
                    response->success = false;
                    response->status = error.what();
                }
            }, rclcpp::ServicesQoS(), cb_group_);

        check_api_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "~/check_api",
            std::bind(&VLMReasoningNode::handleCheckApi, this, std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS(),
            cb_group_);

        get_yolo_client_ = this->create_client<my_actions::srv::GetYoloInference>(
            "/pipeline/infer_yolo",
            rclcpp::ServicesQoS(),
            cb_group_);
        trigger_scan_client_ = this->create_client<std_srvs::srv::Trigger>(
            "/pipeline/request_scan", rclcpp::ServicesQoS(), cb_group_);
        grasp_candidates_client_ = this->create_client<my_actions::srv::FindGraspCandidates>(
            "/pipeline/calculate_active_grasps", rclcpp::ServicesQoS(), cb_group_);
        placement_candidates_client_ =
            this->create_client<my_actions::srv::FindPlacementCandidates>(
                "/pipeline/find_placements",
                rclcpp::ServicesQoS(), cb_group_);
        plan_placement_client_ =
            this->create_client<my_actions::srv::PlanPlacement>(
                "/pipeline/plan_placement",
                rclcpp::ServicesQoS(), cb_group_);
        commit_floor_slots_client_ =
            this->create_client<my_actions::srv::CommitFloorSlots>(
                "/pipeline/commit_floor_slots", rclcpp::ServicesQoS(), cb_group_);
        plan_object_layout_client_ =
            this->create_client<my_actions::srv::PlanObjectLayout>(
                "/pipeline/plan_layout",
                rclcpp::ServicesQoS(), cb_group_);
        pin_layout_observation_client_ =
            this->create_client<my_actions::srv::PinLayoutObservation>(
                "/pipeline/pin_placement_observation",
                rclcpp::ServicesQoS(), cb_group_);
        pin_vamp_layout_observation_client_ =
            this->create_client<my_actions::srv::PinLayoutObservation>(
                "/pipeline/pin_grasp_observation",
                rclcpp::ServicesQoS(), cb_group_);
        get_3d_coordinate_client_ = this->create_client<my_actions::srv::Get3DCoordinate>(
            "/pipeline/get_3d_coordinate", rclcpp::ServicesQoS(), cb_group_);
        update_object_client_ = this->create_client<my_actions::srv::UpdateObjectObservation>(
            "/pipeline/update_object", rclcpp::ServicesQoS(), cb_group_);
        resolve_object_client_ = this->create_client<my_actions::srv::ResolveObject>(
            "/pipeline/resolve_object", rclcpp::ServicesQoS(), cb_group_);
        look_at_client_ = this->create_client<my_actions::srv::LookAt>(
            "/pipeline/look_at_target", rclcpp::ServicesQoS(), cb_group_);
        execute_grasp_client_ = this->create_client<my_actions::srv::ExecuteGrasp>(
            "/pipeline/execute_grasp", rclcpp::ServicesQoS(), cb_group_);
        get_world_state_client_ = this->create_client<my_actions::srv::GetWorldState>(
            "/pipeline/get_world_state", rclcpp::ServicesQoS(), cb_group_);

        append_task_client_ = this->create_client<my_actions::srv::AppendTask>(
            "/pipeline/append_task",
            rclcpp::ServicesQoS(),
            cb_group_);
            
            
        execute_pipeline_client_ = this->create_client<my_actions::srv::ExecutePipeline>(
            "/pipeline/execute_pipeline",
            rclcpp::ServicesQoS(),
            cb_group_);

        has_active_target_ = false;
        RCLCPP_INFO(
            this->get_logger(),
            "VLM Reasoning Node ready (raw synchronized RGB-D grasp path; YOLO optional).");
    }

private:
    std::string api_ip_;
    int api_port_;
    std::string api_server_url_;
    std::string api_base_path_ = "/v1";
    std::string model_name_;
    int request_timeout_sec_;
    bool enable_task_dispatch_;
    bool verbose_;
    double temperature_;
    int max_tokens_;
    int image_max_dimension_ = 1024;
    rclcpp::Client<my_actions::srv::GroundObject>::SharedPtr ground_object_client_;
    rclcpp::Client<my_actions::srv::ResolveManipulationTarget>::SharedPtr active_target_client_;
    rclcpp::Service<my_actions::srv::GroundObject>::SharedPtr ground_object_srv_;
    rclcpp::Client<my_actions::srv::VisionInference>::SharedPtr vision_inference_client_;
    rclcpp::Client<my_actions::srv::CaptureObservation>::SharedPtr capture_observation_client_;
    rclcpp::Client<my_actions::srv::SegmentGroundedImage>::SharedPtr segment_object_client_;
    rclcpp::Service<my_actions::srv::VisionInference>::SharedPtr vision_inference_srv_;
    std::mutex inference_mutex_;
    std::vector<std::string> tracked_objects_;
    std::string world_perception_backend_;
    rclcpp::Service<my_actions::srv::SegmentGroundedImage>::SharedPtr segment_srv_;
    std::string grasp_target_mask_backend_;
    std::unique_ptr<vlm_node::EdgeSamOnnx> edgesam_;
    std::string edgesam_initialization_error_;
    int edgesam_min_mask_pixels_ = 80;
    bool enable_view_repositioning_ = false;
    int grasp_plan_cache_ttl_sec_ = 10;

    struct CachedGraspPlan {
        std::string object_name;
        std_msgs::msg::Header observation_header;
        my_actions::msg::GraspCandidate candidate;
        std::vector<my_actions::msg::GraspCandidate> candidates;
        std::chrono::steady_clock::time_point created_at;
    };
    rclcpp::Client<my_actions::srv::CacheGraspPlan>::SharedPtr cache_grasp_client_;
    rclcpp::Client<my_actions::srv::GetCachedGrasp>::SharedPtr get_cached_grasp_client_;
    rclcpp::Service<my_actions::srv::VerifyPresence>::SharedPtr verify_presence_srv_;
    

    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr observation_image_sub_;
    rclcpp::Service<my_actions::srv::Chat>::SharedPtr chat_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr check_api_srv_;
    rclcpp::Client<my_actions::srv::GetYoloInference>::SharedPtr get_yolo_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_scan_client_;
    rclcpp::Client<my_actions::srv::AppendTask>::SharedPtr append_task_client_;
    rclcpp::Client<my_actions::srv::ExecutePipeline>::SharedPtr execute_pipeline_client_;
    rclcpp::Client<my_actions::srv::FindGraspCandidates>::SharedPtr grasp_candidates_client_;
    rclcpp::Client<my_actions::srv::FindPlacementCandidates>::SharedPtr
        placement_candidates_client_;
    rclcpp::Client<my_actions::srv::PlanPlacement>::SharedPtr
        plan_placement_client_;
    rclcpp::Client<my_actions::srv::PlanObjectLayout>::SharedPtr
        plan_object_layout_client_;
    rclcpp::Client<my_actions::srv::CommitFloorSlots>::SharedPtr
        commit_floor_slots_client_;
    rclcpp::Client<my_actions::srv::PinLayoutObservation>::SharedPtr
        pin_layout_observation_client_;
    rclcpp::Client<my_actions::srv::PinLayoutObservation>::SharedPtr
        pin_vamp_layout_observation_client_;
    rclcpp::Client<my_actions::srv::Get3DCoordinate>::SharedPtr get_3d_coordinate_client_;
    rclcpp::Client<my_actions::srv::UpdateObjectObservation>::SharedPtr update_object_client_;
    rclcpp::Client<my_actions::srv::ResolveObject>::SharedPtr resolve_object_client_;
    rclcpp::Client<my_actions::srv::LookAt>::SharedPtr look_at_client_;
    rclcpp::Client<my_actions::srv::ExecuteGrasp>::SharedPtr execute_grasp_client_;
    rclcpp::Client<my_actions::srv::GetWorldState>::SharedPtr get_world_state_client_;

    cv::Mat latest_image_;
    std::mutex image_mutex_;
    SynchronizedObservation latest_observation_;
    std::mutex observation_image_mutex_;

    std::vector<DetectedObject> spatial_memory_;
    
    float active_target_x_ = 0.0f;
    float active_target_y_ = 0.0f;
    float active_target_z_ = 0.0f;
    bool has_active_target_ = false;
    my_actions::srv::GetYoloInference::Response cached_yolo_result_;
    bool has_cached_yolo_result_ = false;

    bool captureFreshObservation(SynchronizedObservation& observation, std::string& status) {
        if (!capture_observation_client_->wait_for_service(std::chrono::seconds(3))) {
            status = "PIPELINE_CAPTURE_UNAVAILABLE";
            return false;
        }
        auto request = std::make_shared<my_actions::srv::CaptureObservation::Request>();
        auto future = capture_observation_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(35)) != std::future_status::ready) {
            status = "PIPELINE_CAPTURE_TIMEOUT";
            return false;
        }
        const auto result = future.get();
        if (!result->success) {
            status = result->status;
            return false;
        }
        observation.image = cv_bridge::toCvCopy(result->image, "bgr8")->image;
        observation.header = result->image.header;
        observation.cloud = result->cloud;
        observation.camera_transform = result->camera_transform;
        return true;
    }

    my_actions::srv::ResolveManipulationTarget::Response::SharedPtr resolveActiveTarget(
        const SynchronizedObservation &observation, const std::string &name, bool destination) {
        if (!active_target_client_->wait_for_service(std::chrono::seconds(3)))
            throw std::runtime_error("ACTIVE_MEMORY_UNAVAILABLE");
        auto request=std::make_shared<my_actions::srv::ResolveManipulationTarget::Request>();
        request->name=name;
        request->role=destination ? "destination" : "object";
        request->image=*cv_bridge::CvImage(observation.header,"bgr8",observation.image).toImageMsg();
        request->observation_cloud=observation.cloud;
        request->camera_transform=observation.camera_transform;
        auto future=active_target_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(request_timeout_sec_))!=std::future_status::ready)
            throw std::runtime_error("ACTIVE_MEMORY_TIMEOUT_STATE_UNKNOWN");
        auto result=future.get();
        if (!result->success) throw std::runtime_error(result->status);
        RCLCPP_INFO(this->get_logger(),"[ACTIVE TARGET] name=%s reused=%s position=[%.4f %.4f %.4f]",
            name.c_str(),result->reused ? "true" : "false",result->position.x,result->position.y,result->position.z);
        return result;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            std::lock_guard<std::mutex> lock(image_mutex_);
            latest_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void observationImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            auto converted = cv_bridge::toCvCopy(msg, "bgr8");
            std::lock_guard<std::mutex> lock(observation_image_mutex_);
            latest_observation_.image = converted->image;
            latest_observation_.header = msg->header;
        } catch (const cv_bridge::Exception& error) {
            RCLCPP_ERROR(this->get_logger(), "Synchronized observation conversion failed: %s", error.what());
        }
    }

    void configureApiEndpoint() {
        if (api_server_url_.empty()) {
            api_server_url_ = "http://" + api_ip_ + ":" +
                std::to_string(api_port_) + "/v1";
        }
        const std::regex pattern(
            R"(^http://([^/:]+)(?::([0-9]+))?(/.*)?$)");
        std::smatch match;
        if (!std::regex_match(api_server_url_, match, pattern)) {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid api_server_url '%s'; falling back to http://%s:%d/v1.",
                api_server_url_.c_str(), api_ip_.c_str(), api_port_);
            api_base_path_ = "/v1";
            api_server_url_ = "http://" + api_ip_ + ":" +
                std::to_string(api_port_) + api_base_path_;
            return;
        }
        api_ip_ = match[1].str();
        api_port_ = match[2].matched ? std::stoi(match[2].str()) : 80;
        api_base_path_ = match[3].matched ? match[3].str() : "/v1";
        while (api_base_path_.size() > 1 && api_base_path_.back() == '/') {
            api_base_path_.pop_back();
        }
        if (api_base_path_.empty() || api_base_path_ == "/") {
            api_base_path_ = "/v1";
        }
        api_server_url_ = "http://" + api_ip_ + ":" +
            std::to_string(api_port_) + api_base_path_;
    }

    void handleCheckApi(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request;
        httplib::Client client(api_ip_, api_port_);
        client.set_connection_timeout(5);
        client.set_read_timeout(10);

        auto result = client.Get("/health");
        if (!result) {
            response->success = false;
            response->message = "Cannot connect to RoboBrain at http://" + api_ip_ + ":" +
                std::to_string(api_port_) + "/health";
            return;
        }
        if (result->status != 200) {
            response->success = false;
            response->message = "RoboBrain health returned HTTP " +
                std::to_string(result->status);
            return;
        }
        response->success = true;
        response->message = "RoboBrain health endpoint is ready at http://" +
            api_ip_ + ":" + std::to_string(api_port_) + "/health.";
    }

    static std::string trim(const std::string& value) {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        const auto last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    static std::string extractGraspTarget(const std::string& instruction) {
        std::string target = trim(instruction);
        const std::array<std::string, 10> command_prefixes{
            "find grasping pose on ",
            "find a grasping pose on ",
            "find grasp pose on ",
            "find a grasp pose on ",
            "find grasp on ",
            "grasping pose on ",
            "grasp pose on ",
            "pick up ",
            "grasp ",
            "grab "};
        const std::string lower = lowercase(target);
        for (const auto& prefix : command_prefixes) {
            if (lower.rfind(prefix, 0) == 0) {
                target = trim(target.substr(prefix.size()));
                break;
            }
        }
        return target;
    }

    PromptRequest parsePromptRequest(const std::string& input) {
        PromptRequest parsed;
        std::string remaining = trim(input);

        if (!remaining.empty() && remaining.front() == '[') {
            const auto closing = remaining.find(']');
            if (closing != std::string::npos) {
                std::string flag = remaining.substr(1, closing - 1);
                std::transform(
                    flag.begin(), flag.end(), flag.begin(),
                    [](unsigned char character) {
                        return std::toupper(character);
                    });
                const std::set<std::string> modes = {
                    "GENERAL", "POINTING", "GROUNDING", "TRAJECTORY",
                    "ACTION", "YOLO", "GRASP", "FIND_GRASP", "GRAB",
                    "PRE_GRASP", "REFINE_GRASP", "PLACE_UPRIGHT",
                    "PLACE_FREE", "PLACE", "PLAN_UPRIGHT", "PLAN",
                    "PLAN_LAYOUT_UPRIGHT", "PLAN_LAYOUT"
                };
                if (modes.count(flag) > 0) {
                    parsed.mode = flag;
                    remaining = trim(remaining.substr(closing + 1));
                }
            }
        }

        const std::string image_prefix = "[IMAGE:";
        if (remaining.rfind(image_prefix, 0) == 0) {
            const auto closing = remaining.find(']');
            if (closing != std::string::npos) {
                parsed.image_path = trim(
                    remaining.substr(image_prefix.size(),
                                     closing - image_prefix.size()));
                remaining = trim(remaining.substr(closing + 1));
            }
        }

        parsed.prompt = remaining;
        // The action template selects sequential layout whenever it names
        // more than one object; retain explicit layout aliases.
        if ((parsed.mode == "PLAN" || parsed.mode == "PLAN_UPRIGHT") &&
            vlm_node::parseLayoutPrompt(parsed.prompt).valid) {
            parsed.mode = parsed.mode == "PLAN_UPRIGHT" ?
                "PLAN_LAYOUT_UPRIGHT" : "PLAN_LAYOUT";
        }

        std::string normalized_prompt = parsed.prompt;
        std::transform(normalized_prompt.begin(), normalized_prompt.end(), normalized_prompt.begin(),
                       [](unsigned char character) { return std::tolower(character); });
        if (parsed.mode == "GENERAL" &&
            (normalized_prompt.find("grasp pose") != std::string::npos ||
             normalized_prompt.find("find grasp") != std::string::npos)) {
            parsed.mode = "FIND_GRASP";
        } else if (parsed.mode == "GENERAL" &&
                   (normalized_prompt.rfind("grab ", 0) == 0 ||
                    normalized_prompt.rfind("grasp ", 0) == 0 ||
                    normalized_prompt.rfind("pick up ", 0) == 0)) {
            parsed.mode = "GRAB";
        }
        return parsed;
    }

    std::string prepareOfficialPrompt(
        const std::string& mode, const std::string& prompt) {
        if (mode == "POINTING") {
            return prompt +
                ". Please provide its 2D coordinates. Your answer should be "
                "formatted as a tuple, i.e. [(x, y)], where the tuple contains "
                "the x and y coordinates of a point satisfying the conditions "
                "above.";
        }
        if (mode == "TRAJECTORY") {
            return "Please predict 3D end-effector-centric waypoints to complete "
                "the task successfully. The task is \"" + prompt +
                "\". Your answer should be formatted as a list of tuples, i.e., "
                "[(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains "
                "the x and y coordinates and the depth of the point.";
        }
        if (mode == "GROUNDING") {
            return vlm_node::groundingPrompt(prompt);
        }
        return prompt;
    }

    std::string prepareGroundingPresencePrompt(const std::string& target) {
        return
            "Check whether the object or region described between <target> and "
            "</target> is clearly visible in the image. Respond with exactly "
            "PRESENT if it is clearly visible. Respond with exactly ABSENT if "
            "it is not visible, occluded beyond recognition, or you are "
            "uncertain. Do not provide coordinates or any explanation. "
            "<target>" + target + "</target>";
    }

    std::string prepareGraspPresencePrompt(const std::string& instruction) {
        return
            "The text inside <instruction> is a robot task, not a literal "
            "object description. Determine whether the physical target object "
            "and specifically requested object part needed to execute the "
            "instruction are clearly visible in the image. Respond with "
            "exactly PRESENT if both the target object and requested part are "
            "clearly visible. Respond with exactly ABSENT if either is missing, "
            "unrecognizable, or uncertain. Do not evaluate whether an action "
            "itself is visible. Do not explain. <instruction>" + instruction +
            "</instruction>";
    }

    bool presenceResponseIsAffirmative(const std::string& response) {
        std::string normalized = trim(response);
        std::transform(
            normalized.begin(), normalized.end(), normalized.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::toupper(character));
            });
        return normalized == "PRESENT";
    }

    void parseAndLogOfficialResponse(
        const std::string& mode, const std::string& response) {
        if (!verbose_) {
            return;
        }

        std::regex pattern;
        std::string label;
        if (mode == "POINTING") {
            pattern = std::regex(
                "\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\)");
            label = "normalized 2D point";
        } else if (mode == "GROUNDING") {
            pattern = std::regex(
                "\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*"
                "([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
            label = "normalized bounding box";
        } else if (mode == "TRAJECTORY") {
            pattern = std::regex(
                "\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*"
                "([+-]?[0-9]*\\.?[0-9]+)\\s*\\)");
            label = "3D image/depth waypoint";
        } else {
            RCLCPP_INFO(
                this->get_logger(),
                "[ROBOBRAIN PARSE] GENERAL mode returns unstructured text; "
                "no task was constructed.");
            return;
        }

        size_t count = 0;
        for (auto match = std::sregex_iterator(
                 response.begin(), response.end(), pattern);
             match != std::sregex_iterator(); ++match) {
            ++count;
            RCLCPP_INFO(
                this->get_logger(),
                "[ROBOBRAIN PARSE] %s %zu: %s",
                label.c_str(), count, match->str().c_str());
        }
        if (count == 0) {
            RCLCPP_WARN(
                this->get_logger(),
                "[ROBOBRAIN PARSE] No %s matched the official output format. "
                "No task was constructed.", label.c_str());
        } else {
            RCLCPP_INFO(
                this->get_logger(),
                "[ROBOBRAIN PARSE] Parsed %zu result(s). Model-only modes never "
                "append robot tasks.", count);
        }
    }

    static int normalizedToPixel(double coordinate, int dimension) {
        return vlm_node::groundingPixel(coordinate, dimension);
    }

    static void expandGroundingBox(
        int& x1, int& y1, int& x2, int& y2,
        int image_width, int image_height) {
        constexpr int padding_pixels = 5;
        x1 = std::max(0, x1 - padding_pixels);
        y1 = std::max(0, y1 - padding_pixels);
        x2 = std::min(image_width - 1, x2 + padding_pixels);
        y2 = std::min(image_height - 1, y2 + padding_pixels);
    }

    bool parseNormalizedPoint(
        const std::string& reply, const cv::Mat& image, int& u, int& v) const {
        const std::regex point_pattern(
            "\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\)");
        std::smatch match;
        if (!std::regex_search(reply, match, point_pattern)) return false;
        const int normalized_u = std::clamp(std::stoi(match[1]), 0, 1000);
        const int normalized_v = std::clamp(std::stoi(match[2]), 0, 1000);
        u = normalizedToPixel(normalized_u, image.cols);
        v = normalizedToPixel(normalized_v, image.rows);
        return true;
    }

    bool pixelToWorld(
        const std_msgs::msg::Header& observation_header, int u, int v,
        geometry_msgs::msg::PointStamped& point, std::string& status) {
        if (!get_3d_coordinate_client_->wait_for_service(std::chrono::seconds(2))) {
            status = "3D_LOOKUP_UNAVAILABLE";
            return false;
        }
        auto request = std::make_shared<my_actions::srv::Get3DCoordinate::Request>();
        request->observation_header = observation_header;
        request->u = u;
        request->v = v;
        request->width = 1;
        request->height = 1;
        auto future = get_3d_coordinate_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
            status = "3D_LOOKUP_TIMEOUT";
            return false;
        }
        const auto result = future.get();
        status = result->status;
        if (!result->success) return false;
        point.header = result->result_header;
        point.point.x = result->x;
        point.point.y = result->y;
        point.point.z = result->z;
        RCLCPP_INFO(
            this->get_logger(),
            "[WORLD REFRESH] pixel=(%d,%d), corrected=(%d,%d), base_link=(%.3f,%.3f,%.3f)",
            u, v, result->corrected_u, result->corrected_v,
            result->x, result->y, result->z);
        return true;
    }

    void reportObjectObservation(
        const std::string& name, bool detected,
        const geometry_msgs::msg::PointStamped& point) {
        if (!update_object_client_->wait_for_service(std::chrono::milliseconds(100))) {
            RCLCPP_DEBUG(
                this->get_logger(),
                "[WORLD REFRESH] Master is not running; '%s' remains a perception-only result.",
                name.c_str());
            return;
        }
        auto request = std::make_shared<my_actions::srv::UpdateObjectObservation::Request>();
        request->name = name;
        request->detected = detected;
        request->position = point;
        // Do not block grasp planning on bookkeeping. The master applies
        // association and conservative missed-observation removal.
        update_object_client_->async_send_request(request);
    }

    void refreshTrackedObjects(
        const SynchronizedObservation& observation) {
        RCLCPP_INFO(
            this->get_logger(),
            "[WORLD REFRESH] Locating %zu configured object(s) in one VLM request.",
            tracked_objects_.size());
        if (tracked_objects_.empty()) {
            return;
        }
        std::ostringstream prompt;
        prompt << "Locate every clearly visible target in this image. Return "
                  "exactly one JSON object. Each key must be one of: [";
        for (size_t index = 0; index < tracked_objects_.size(); ++index) {
            if (index > 0) prompt << ", ";
            prompt << "\"" << tracked_objects_[index] << "\"";
        }
        prompt << "]. The value must be [x,y] for one point on that object's "
                  "visible physical surface, using normalized image coordinates "
                  "from 0 to 1000, or null when that object is absent or uncertain. "
                  "Include every listed key. Do not add explanation.";
        const std::string reply = callVLM("", prompt.str(), 128, observation.image, {}, observation.header);
        json locations;
        try {
            const auto begin = reply.find('{');
            const auto end = reply.rfind('}');
            if (begin == std::string::npos || end == std::string::npos || end < begin) {
                throw std::runtime_error("JSON object not found");
            }
            locations = json::parse(reply.substr(begin, end - begin + 1));
        } catch (const std::exception& error) {
            RCLCPP_WARN(
                this->get_logger(),
                "[WORLD REFRESH] Batched pointing parse failed: %s; reply=%s",
                error.what(), reply.c_str());
            return;
        }
        for (const auto& object_name : tracked_objects_) {
            geometry_msgs::msg::PointStamped point;
            point.header = observation.header;
            point.header.frame_id = "base_link";
            if (!locations.contains(object_name) || locations[object_name].is_null() ||
                !locations[object_name].is_array() ||
                locations[object_name].size() != 2 ||
                !locations[object_name][0].is_number() ||
                !locations[object_name][1].is_number()) {
                reportObjectObservation(object_name, false, point);
                RCLCPP_INFO(
                    this->get_logger(), "[WORLD REFRESH] %s: not detected",
                    object_name.c_str());
                continue;
            }
            const int u = normalizedToPixel(
                locations[object_name][0].get<double>(), observation.image.cols);
            const int v = normalizedToPixel(
                locations[object_name][1].get<double>(), observation.image.rows);
            std::string lookup_status;
            if (!pixelToWorld(observation.header, u, v, point, lookup_status)) {
                RCLCPP_WARN(
                    this->get_logger(), "[WORLD REFRESH] %s: %s",
                    object_name.c_str(), lookup_status.c_str());
                reportObjectObservation(object_name, false, point);
                continue;
            }
            reportObjectObservation(object_name, true, point);
        }
    }

    static std::string lowercase(std::string value) {
        std::transform(
            value.begin(), value.end(), value.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
        return value;
    }

    static bool candidateIsSafeToExecute(
        const my_actions::msg::GraspCandidate& candidate) {
        if (!candidate.ik_valid || !candidate.collision_checked ||
            !candidate.grasp_collision_valid ||
            !candidate.pregrasp_collision_valid ||
            candidate.pregrasp_collision_overridden || candidate.score <= 0.0F) {
            return false;
        }
        const bool body_model = candidate.validation_status.rfind(
            "CUP_BODY_MODEL_", 0) == 0;
        return candidate.measured_contacts_valid || body_model;
    }

    bool takeFreshSafeCachedPlan(
        const std::string& object_name, CachedGraspPlan& plan,
        std::string& status) {
        if (!get_cached_grasp_client_->wait_for_service(std::chrono::seconds(3))) {
            status = "MASTER_GRASP_CACHE_UNAVAILABLE";
            return false;
        }
        auto request = std::make_shared<my_actions::srv::GetCachedGrasp::Request>();
        request->object_name = object_name;
        auto future = get_cached_grasp_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            status = "MASTER_GRASP_CACHE_TIMEOUT";
            return false;
        }
        const auto result = future.get();
        status = result->status;
        if (!result->success || result->candidates.empty() ||
                !candidateIsSafeToExecute(result->candidates.front())) return false;
        plan = {result->object_name, result->observation_header, result->candidates.front(),
                result->candidates, std::chrono::steady_clock::now()};
        return true;
    }

    std::optional<std::string> singleCachedGraspObject() {
        CachedGraspPlan plan;
        std::string status;
        if (!takeFreshSafeCachedPlan("", plan, status)) return std::nullopt;
        return plan.object_name;
    }

    void cacheSafeGraspPlans(
        const std::string& object_name,
        const std_msgs::msg::Header& observation_header,
        const std::vector<my_actions::msg::GraspCandidate>& candidates) {
        std::vector<my_actions::msg::GraspCandidate> safe_candidates;
        std::copy_if(
            candidates.begin(), candidates.end(),
            std::back_inserter(safe_candidates), candidateIsSafeToExecute);
        if (safe_candidates.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "[GRASP] No candidate was cached because none meets the execution safety gate.");
            return;
        }
        if (!cache_grasp_client_->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(this->get_logger(), "MASTER_GRASP_CACHE_UNAVAILABLE");
            return;
        }
        auto request = std::make_shared<my_actions::srv::CacheGraspPlan::Request>();
        request->object_name = object_name;
        request->observation_header = observation_header;
        request->candidates = safe_candidates;
        request->ttl_seconds = grasp_plan_cache_ttl_sec_;
        auto future = cache_grasp_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "MASTER_GRASP_CACHE_TIMEOUT");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "[GRASP CACHE] %s", future.get()->status.c_str());
    }

    std::string targetWorldName(const std::string& instruction) const {
        const std::string normalized = lowercase(instruction);
        if (world_perception_backend_ == "yolo") {
            if (normalized.find("mug") != std::string::npos) return "cup";
            std::string best_match;
            for (const auto& class_name : COCO_CLASSES) {
                if (normalized.find(class_name) != std::string::npos &&
                    class_name.size() > best_match.size()) {
                    best_match = class_name;
                }
            }
            return best_match;
        }
        for (const auto& configured_name : tracked_objects_) {
            if (normalized.find(lowercase(configured_name)) != std::string::npos) {
                return configured_name;
            }
        }
        return instruction;
    }

    bool getYoloResultForObservation(
        const SynchronizedObservation& observation,
        my_actions::srv::GetYoloInference::Response& yolo_result,
        std::string& status) {
        if (!get_yolo_client_->wait_for_service(std::chrono::seconds(3))) {
            status = "YOLO_UNAVAILABLE: /active_vision/get_yolo_inference";
            return false;
        }
        const int64_t expected_stamp =
            rclcpp::Time(observation.header.stamp).nanoseconds();
        if (has_cached_yolo_result_ && 
            rclcpp::Time(cached_yolo_result_.observation_header.stamp).nanoseconds() == expected_stamp) {
            yolo_result = cached_yolo_result_;
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(15);
        while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
            auto request =
                std::make_shared<my_actions::srv::GetYoloInference::Request>();
            auto future = get_yolo_client_->async_send_request(request);
            if (future.wait_for(std::chrono::seconds(2)) !=
                std::future_status::ready) {
                continue;
            }
            const auto result = future.get();
            if (!result->success ||
                rclcpp::Time(result->observation_header.stamp).nanoseconds() !=
                    expected_stamp) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            yolo_result = *result;
            cached_yolo_result_ = *result;
            has_cached_yolo_result_ = true;
            return true;
        }
        status = "YOLO_TIMEOUT: no result for synchronized observation";
        return false;
    }

    bool targetVisibleWithYolo(
        const SynchronizedObservation& observation,
        const std::string& target_class, std::string& status) {
        if (target_class.empty()) {
            status = "YOLO_CLASS_UNKNOWN: name a class supported by the detector";
            return false;
        }
        my_actions::srv::GetYoloInference::Response result;
        if (getYoloResultForObservation(observation, result, status)) {
            const bool found = std::find(
                result.classes.begin(), result.classes.end(),
                target_class) != result.classes.end();
            status = found ? "PRESENT" : "ABSENT";
            RCLCPP_INFO(
                this->get_logger(),
                "[WORLD PERCEPTION] backend=yolo target=%s result=%s",
                target_class.c_str(), status.c_str());
            return found;
        }
        return false;
    }



    bool targetVisible(
        const SynchronizedObservation& observation,
        const std::string& instruction, const std::string& world_name,
        std::string& status) {
        if (world_perception_backend_ == "yolo") {
            return targetVisibleWithYolo(observation, world_name, status);
        }
        const std::string presence = callVLM(
            "", prepareGraspPresencePrompt(instruction), 16,
            observation.image, {}, observation.header);
        status = trim(presence);
        return presenceResponseIsAffirmative(presence);
    }

    bool repositionToStoredTarget(
        const std::string& world_name, std::string& status) {
        if (!enable_view_repositioning_) {
            status = "TARGET_OUT_OF_VIEW: view repositioning is disabled";
            return false;
        }
        if (!resolve_object_client_->wait_for_service(std::chrono::seconds(3))) {
            status = "WORLD_STATE_UNAVAILABLE: /pipeline/resolve_object";
            return false;
        }
        auto resolve_request =
            std::make_shared<my_actions::srv::ResolveObject::Request>();
        resolve_request->name = world_name;
        auto resolve_future = resolve_object_client_->async_send_request(
            resolve_request);
        if (resolve_future.wait_for(std::chrono::seconds(3)) !=
            std::future_status::ready) {
            status = "WORLD_STATE_TIMEOUT";
            return false;
        }
        const auto resolved = resolve_future.get();
        if (!resolved->success) {
            status = "TARGET_LOCATION_" + resolved->status;
            return false;
        }
        if (!look_at_client_->wait_for_service(std::chrono::seconds(3))) {
            status = "LOOK_AT_UNAVAILABLE: /master/look_at_target";
            return false;
        }
        auto look_request = std::make_shared<my_actions::srv::LookAt::Request>();
        look_request->look_at_x = resolved->position.point.x;
        look_request->look_at_y = resolved->position.point.y;
        look_request->look_at_z = resolved->position.point.z;
        RCLCPP_INFO(
            this->get_logger(),
            "[NBV] Repositioning view toward %s (%s) at (%.3f, %.3f, %.3f).",
            world_name.c_str(), resolved->object_id.c_str(),
            look_request->look_at_x, look_request->look_at_y,
            look_request->look_at_z);
        auto look_future = look_at_client_->async_send_request(look_request);
        if (look_future.wait_for(std::chrono::seconds(180)) !=
            std::future_status::ready || !look_future.get()->success) {
            status = "NBV_REPOSITION_FAILED";
            return false;
        }
        status = "NBV_REPOSITIONED";
        return true;
    }

    std::mutex capture_mutex_;
    vlm_node::CaptureAssociations capture_associations_;

    static std::string captureFrame(const sensor_msgs::msg::Image &image) {
        return image.header.frame_id+":"+std::to_string(image.header.stamp.sec)+":"+
            std::to_string(image.header.stamp.nanosec)+":"+std::to_string(image.width)+":"+std::to_string(image.height);
    }

    static std::filesystem::path graspCaptureDirectory() {
        const char* home = std::getenv("HOME");
        if (home == nullptr || *home == '\0') return {};
        return std::filesystem::path(home) / "robobrain_inference";
    }

    void saveCaptureImage(const std::filesystem::path& path,
                          const cv::Mat& image) {
        if (path.empty() || image.empty()) return;
        try {
            std::filesystem::create_directories(path.parent_path());
            if (!cv::imwrite(path.string(), image)) {
                RCLCPP_WARN(this->get_logger(), "Could not save image: %s", path.c_str());
            }
        } catch (const std::exception& error) {
            RCLCPP_WARN(this->get_logger(), "Could not save %s: %s",
                        path.c_str(), error.what());
        }
    }

    std::string encodeImageToBase64(
        const cv::Mat& img, const std::filesystem::path& sent_image_path = {}) {
        std::vector<uchar> buf;
        if (img.empty() || !cv::imencode(".jpg", img, buf) || buf.empty()) {
            throw std::runtime_error("failed to JPEG-encode RoboBrain image");
        }
        if (!sent_image_path.empty()) {
            try {
                std::filesystem::create_directories(sent_image_path.parent_path());
                std::ofstream output(sent_image_path, std::ios::binary);
                output.write(reinterpret_cast<const char*>(buf.data()),
                             static_cast<std::streamsize>(buf.size()));
                if (!output) throw std::runtime_error("write failed");
            } catch (const std::exception& error) {
                RCLCPP_WARN(this->get_logger(), "Could not save %s: %s",
                            sent_image_path.c_str(), error.what());
            }
        }
        
        static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";
        
        std::string ret;
        ret.reserve(((buf.size() + 2) / 3) * 4);
        int i = 0, j = 0;
        unsigned char char_array_3[3], char_array_4[4];

        for (size_t p = 0; p < buf.size(); p++) {
            char_array_3[i++] = buf[p];
            if (i == 3) {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;
                for(i = 0; (i <4) ; i++) ret += base64_chars[char_array_4[i]];
                i = 0;
            }
        }
        if (i) {
            for(j = i; j < 3; j++) char_array_3[j] = '\0';
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (j = 0; (j < i + 1); j++) ret += base64_chars[char_array_4[j]];
            while((i++ < 3)) ret += '=';
        }
        return ret;
    }

    std::string callVLM(const std::string& system_prompt, const std::string& user_prompt,
                        int max_tokens, cv::Mat image,
                        const std::filesystem::path& sent_image_path = {},
                        const std_msgs::msg::Header &header = std_msgs::msg::Header()) {
        if (!vision_inference_client_->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(this->get_logger(), "PIPELINE_INFERENCE_UNAVAILABLE");
            return "";
        }
        auto request = std::make_shared<my_actions::srv::VisionInference::Request>();
        request->system_prompt = system_prompt;
        request->user_prompt = user_prompt;
        request->max_tokens = max_tokens;
        request->sent_image_path = sent_image_path.string();
        if (!image.empty()) {
            request->image = *cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        }
        auto future = vision_inference_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(request_timeout_sec_ + 30)) !=
                std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "PIPELINE_INFERENCE_TIMEOUT");
            return "";
        }
        const auto response = future.get();
        if (!response->success) {
            RCLCPP_ERROR(this->get_logger(), "%s", response->status.c_str());
            return "";
        }
        return response->text;
    }

    std::string groundObject(const cv::Mat &image, const std::string &target,
                             const std_msgs::msg::Header &header,
                             const std::filesystem::path &path = {}) {
        if (!ground_object_client_->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(this->get_logger(), "PIPELINE_GROUNDING_UNAVAILABLE");
            return "";
        }
        auto request = std::make_shared<my_actions::srv::GroundObject::Request>();
        request->image = *cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        request->target = target;
        request->sent_image_path = path.string();
        auto future = ground_object_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(request_timeout_sec_ + 45)) !=
                std::future_status::ready) return "";
        const auto response = future.get();
        return response->success ? response->text : "";
    }

    vlm_node::EdgeSamResult segmentObject(const cv::Mat &image, const cv::Rect &box,
                                         const std_msgs::msg::Header &header) {
        if (!segment_object_client_->wait_for_service(std::chrono::seconds(3))) {
            throw std::runtime_error("PIPELINE_SEGMENTATION_UNAVAILABLE");
        }
        auto request = std::make_shared<my_actions::srv::SegmentGroundedImage::Request>();
        request->image = *cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        request->bbox = {box.x, box.y, std::min(image.cols, box.x + box.width),
                         std::min(image.rows, box.y + box.height)};
        auto future = segment_object_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
            throw std::runtime_error("PIPELINE_SEGMENTATION_TIMEOUT");
        }
        const auto response = future.get();
        if (!response->success) throw std::runtime_error(response->status);
        vlm_node::EdgeSamResult result;
        result.mask = cv_bridge::toCvCopy(response->mask, "mono8")->image;
        result.score = response->score;
        result.mask_pixels = response->mask_pixels;
        result.encoder_ms = response->encoder_ms;
        result.decoder_ms = response->decoder_ms;
        return result;
    }

    std::string callVLMBackend(const std::string& system_prompt, const std::string& user_prompt,
                        int max_tokens, cv::Mat current_image,
                        const std::filesystem::path& sent_image_path = {}) {
        std::unique_lock<std::mutex> inference_lock(
            inference_mutex_, std::try_to_lock);
        if (!inference_lock.owns_lock()) {
            RCLCPP_WARN(
                this->get_logger(),
                "RoboBrain is busy; rejecting the new request instead of "
                "queueing a stale or duplicate inference.");
            return "";
        }
        const auto request_started = std::chrono::steady_clock::now();
        httplib::Client cli(api_ip_, api_port_);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(request_timeout_sec_);
        cli.set_write_timeout(30);

        json payload = {
            {"model", model_name_},
            {"temperature", temperature_},
            {"max_tokens", std::max(1, max_tokens)},
            {"stream", false},
            {"chat_template_kwargs", {{"enable_thinking", false}}}
        };

        const bool grounding_request = user_prompt.rfind(
            "Please provide the bounding box coordinate of the region this sentence describes: ", 0) == 0;
        if (grounding_request) vlm_node::configureGroundingRequest(payload);

        cv::Mat sent_image = current_image;
        if (!grounding_request && !current_image.empty()) {
            const float max_dim = static_cast<float>(image_max_dimension_);
            const float scale = std::min(max_dim / current_image.cols, max_dim / current_image.rows);
            if (scale < 1.0F) {
                cv::resize(current_image, sent_image, cv::Size(), scale, scale, cv::INTER_AREA);
            }
        }
        const std::string image_url = sent_image.empty() ? "" :
            "data:image/jpeg;base64," + encodeImageToBase64(sent_image, sent_image_path);
        json content_array = vlm_node::imageFirstContent(image_url, user_prompt);

        json user_message = {
            {"role", "user"},
            {"content", content_array}
        };
        
        if (system_prompt.empty()) {
            payload["messages"] = {user_message};
        } else {
            json sys_message = {
                {"role", "system"},
                {"content", system_prompt}
            };
            payload["messages"] = {sys_message, user_message};
        }

        if (verbose_) {
            const std::string image_description = current_image.empty()
                ? "none"
                : "JPEG data URL " + std::to_string(current_image.cols) +
                    "x" + std::to_string(current_image.rows);
            RCLCPP_INFO(
                this->get_logger(),
                "[ROBOBRAIN TX] %s model=%s image=%s "
                "temperature=%.2f max_tokens=%d\nsystem: %s\nuser: %s",
                api_server_url_.c_str(), model_name_.c_str(),
                image_description.c_str(),
                payload["temperature"].get<double>(), payload["max_tokens"].get<int>(),
                system_prompt.empty() ? "<none>" : system_prompt.c_str(),
                user_prompt.c_str());
        }

        const std::string endpoint = api_base_path_ + "/chat/completions";
        auto res = cli.Post(endpoint, payload.dump(), "application/json");
        RCLCPP_INFO(this->get_logger(),
            "[VLM TIMING] request_seconds=%.6f http_status=%d model=%s",
            std::chrono::duration<double>(std::chrono::steady_clock::now()-request_started).count(),
            res ? res->status : 0, model_name_.c_str());
        
        if (!res || res->status != 200) {
            const std::string err = res
                ? "HTTP " + std::to_string(res->status)
                : "connection failed or timed out";
            RCLCPP_ERROR(
                this->get_logger(), "RoboBrain request to %s failed: %s",
                endpoint.c_str(), err.c_str());
            if (res && !res->body.empty() &&
                res->body.find("data:image") == std::string::npos) {
                const std::string detail = res->body.substr(0, 512);
                RCLCPP_ERROR(
                    this->get_logger(), "RoboBrain response: %s",
                    detail.c_str());
            }
            return "";
        }

        try {
            json res_json = json::parse(res->body);
            if (!res_json.contains("choices") || res_json["choices"].size() == 0) {
                RCLCPP_ERROR(this->get_logger(), "Unexpected JSON structure.");
                return "";
            }
            const auto& content = res_json["choices"][0]["message"]["content"];
            if (!content.is_string()) {
                RCLCPP_ERROR(this->get_logger(), "RoboBrain returned non-text message content.");
                return "";
            }
            const std::string reply = content.get<std::string>();
            if (verbose_) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "[ROBOBRAIN RX] HTTP %d\n%s",
                    res->status, reply.c_str());
            }
            return reply;
        } catch (const json::exception& e) {
            RCLCPP_ERROR(
                this->get_logger(), "Invalid RoboBrain response JSON: %s",
                e.what());
            return "";
        }
    }

    void handleChatPrompt(const std::shared_ptr<my_actions::srv::Chat::Request> request,
                          std::shared_ptr<my_actions::srv::Chat::Response> response) 
    {
        const PromptRequest parsed = parsePromptRequest(request->prompt);
        if (parsed.prompt.empty() && parsed.mode != "GRASP" &&
            parsed.mode != "GRAB") {
            response->response =
                "Prompt is empty. Use [GENERAL], [POINTING], [GROUNDING], "
                "[TRAJECTORY], [ACTION], [YOLO], [FIND_GRASP], "
                "[PRE_GRASP], [REFINE_GRASP], [PLACE_UPRIGHT], "
                "[PLACE], [PLACE_FREE], [PLAN_UPRIGHT], [PLAN], "
                "[PLAN_LAYOUT_UPRIGHT], [PLAN_LAYOUT], or [GRAB].";
            return;
        }

        const std::string normalized_prompt = lowercase(trim(parsed.prompt));
        if (normalized_prompt == "world state" ||
            normalized_prompt == "show world state" ||
            normalized_prompt == "get world state") {
            if (!get_world_state_client_->wait_for_service(
                    std::chrono::seconds(2))) {
                response->response =
                    "WORLD_STATE_UNAVAILABLE: /pipeline/get_world_state";
                return;
            }
            auto state_request =
                std::make_shared<my_actions::srv::GetWorldState::Request>();
            auto state_future = get_world_state_client_->async_send_request(
                state_request);
            if (state_future.wait_for(std::chrono::seconds(3)) !=
                std::future_status::ready) {
                response->response = "WORLD_STATE_TIMEOUT";
                return;
            }
            response->response = state_future.get()->state_json;
            return;
        }

        if (parsed.mode == "PLAN_LAYOUT_UPRIGHT" ||
            parsed.mode == "PLAN_LAYOUT") {
            const bool upright_policy =
                parsed.mode == "PLAN_LAYOUT_UPRIGHT";
            const auto layout = vlm_node::parseLayoutPrompt(parsed.prompt);
            if (!layout.valid) {
                response->response =
                    "LAYOUT_PROMPT_INVALID: use 'place <object>, <object>, ... onto <destination>'";
                return;
            }
            const auto &object_names = layout.objects;
            const auto &destination_name = layout.destination;

            SynchronizedObservation observation;
            std::string observation_status;
            if (!captureFreshObservation(observation, observation_status)) {
                response->response = observation_status;
                return;
            }
            if (!pin_layout_observation_client_->wait_for_service(
                    std::chrono::seconds(3))) {
                response->response =
                    "LAYOUT_PIN_SERVICE_UNAVAILABLE: /pipeline/pin_placement_observation";
                return;
            }
            if (!pin_vamp_layout_observation_client_->wait_for_service(
                    std::chrono::seconds(3))) {
                response->response = "LAYOUT_VAMP_PIN_SERVICE_UNAVAILABLE";
                return;
            }
            bool observation_pinned = false;
            for (int attempt = 0; attempt < 30 && !observation_pinned; ++attempt) {
                auto pin_request = std::make_shared<
                    my_actions::srv::PinLayoutObservation::Request>();
                pin_request->observation_header = observation.header;
                auto pin_future = pin_layout_observation_client_->async_send_request(
                    pin_request);
                if (pin_future.wait_for(std::chrono::seconds(1)) ==
                    std::future_status::ready) {
                    observation_pinned = pin_future.get()->success;
                    if (observation_pinned) {
                        auto vamp_pin = pin_vamp_layout_observation_client_->async_send_request(
                            pin_request);
                        observation_pinned =
                            vamp_pin.wait_for(std::chrono::seconds(1)) ==
                            std::future_status::ready && vamp_pin.get()->success;
                    }
                }
                if (!observation_pinned) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            if (!observation_pinned) {
                response->response =
                    "LAYOUT_OBSERVATION_PIN_FAILED: matching point cloud did not arrive";
                return;
            }
            if (!edgesam_) {
                response->response =
                    "EDGESAM_UNAVAILABLE: " + edgesam_initialization_error_;
                return;
            }
            const auto capture_dir = graspCaptureDirectory();
            if (!capture_dir.empty()) {
                for (const char *name : {"destination_bounding_box.png",
                                         "destination_mask.png",
                                         "destination_segmented.png"}) {
                    std::error_code error;
                    std::filesystem::remove(capture_dir / name, error);
                    if (error) {
                        RCLCPP_WARN(this->get_logger(),
                                    "Could not clear previous %s: %s",
                                    name, error.message().c_str());
                    }
                }
            }
            auto segment_named_region = [this, &observation, &capture_dir](
                const std::string &name, bool destination,
                sensor_msgs::msg::Image &mask_message,
                std::string &error) {
                try {
                    const auto target=resolveActiveTarget(observation,name,destination);
                    mask_message=target->mask;
                    if (destination && !capture_dir.empty()) {
                        const auto mask=cv_bridge::toCvCopy(mask_message,"mono8")->image;
                        saveCaptureImage(capture_dir / "destination_mask.png",mask);
                        auto boxed=observation.image.clone();
                        cv::rectangle(boxed,cv::Point(target->pixel_box[0],target->pixel_box[1]),
                            cv::Point(target->pixel_box[2],target->pixel_box[3]),cv::Scalar(0,255,0),2);
                        saveCaptureImage(capture_dir / "destination_bounding_box.png",boxed);
                        cv::Mat segmented=cv::Mat::zeros(observation.image.size(),observation.image.type());
                        observation.image.copyTo(segmented,mask);
                        saveCaptureImage(capture_dir / "destination_segmented.png",segmented);
                    }
                    return true;
                } catch (const std::exception &exception) {
                    error=exception.what();return false;
                }
            };

            sensor_msgs::msg::Image destination_mask;
            std::string segmentation_error;
            if (!segment_named_region(
                    destination_name, true, destination_mask,
                    segmentation_error)) {
                response->response = "LAYOUT_SEGMENTATION_FAILED: " +
                    segmentation_error;
                return;
            }
            std::vector<sensor_msgs::msg::Image> object_masks;
            object_masks.reserve(object_names.size());
            std::vector<std::string> segmented_names;
            std::vector<std::string> segmentation_failures;
            for (const auto &name : object_names) {
                sensor_msgs::msg::Image mask;
                if (!segment_named_region(
                        name, false, mask, segmentation_error)) {
                    segmentation_failures.push_back(name + ": " + segmentation_error);
                    RCLCPP_WARN(this->get_logger(), "[LAYOUT SKIPPED] %s: %s",
                        name.c_str(), segmentation_error.c_str());
                    continue;
                }
                segmented_names.push_back(name);
                object_masks.push_back(std::move(mask));
            }
            if (segmented_names.empty()) {
                response->response = "LAYOUT_FAILED: no objects could be segmented";
                for (const auto &failure : segmentation_failures) {
                    response->response += "\nSKIPPED: " + failure;
                }
                return;
            }
            if (!plan_object_layout_client_->wait_for_service(
                    std::chrono::seconds(5))) {
                response->response =
                    "LAYOUT_BACKEND_UNAVAILABLE: /pipeline/plan_layout";
                return;
            }
            auto layout_request =
                std::make_shared<my_actions::srv::PlanObjectLayout::Request>();
            layout_request->observation_header = observation.header;
            layout_request->destination_mask = std::move(destination_mask);
            layout_request->object_masks = std::move(object_masks);
            layout_request->object_names = segmented_names;
            layout_request->destination_name = destination_name;
            layout_request->orientation_policy =
                upright_policy ? "upright" : "free";
            auto layout_future = plan_object_layout_client_->async_send_request(
                layout_request);
            if (layout_future.wait_for(std::chrono::seconds(
                    request_timeout_sec_ * static_cast<int>(object_names.size()))) != std::future_status::ready) {
                response->response = "LAYOUT_BACKEND_TIMEOUT";
                return;
            }
            const auto layout_result = layout_future.get();
            response->response = (layout_result->success ||
                layout_result->status.rfind("LAYOUT_PARTIAL", 0) == 0) ?
                layout_result->status : "LAYOUT_PLAN_FAILED: " +
                layout_result->status;
            for (const auto &failure : segmentation_failures) {
                response->response += "\nSKIPPED: " + failure;
            }
            RCLCPP_INFO(this->get_logger(), "[LAYOUT SUMMARY] %s", response->response.c_str());
            return;
        }

        if (parsed.mode == "PLACE_UPRIGHT" || parsed.mode == "PLACE_FREE" ||
            parsed.mode == "PLACE" || parsed.mode == "PLAN_UPRIGHT" ||
            parsed.mode == "PLAN") {
            const bool placement_plan_requested =
                parsed.mode == "PLAN_UPRIGHT" || parsed.mode == "PLAN";
            const bool upright_policy =
                parsed.mode == "PLACE_UPRIGHT" ||
                parsed.mode == "PLAN_UPRIGHT";
            const std::array<std::string, 4> separators{
                " onto ", " into ", " on ", " in "};
            size_t separator_position = std::string::npos;
            std::string separator;
            for (const auto &candidate : separators) {
                const size_t position = normalized_prompt.find(candidate);
                if (position != std::string::npos &&
                    (separator_position == std::string::npos ||
                     position < separator_position)) {
                    separator_position = position;
                    separator = candidate;
                }
            }
            if (separator_position == std::string::npos) {
                response->response =
                    "PLACEMENT_PROMPT_INVALID: use 'place <object> onto <destination>'";
                return;
            }
            auto remove_prefix = [](std::string value) {
                value = trim(value);
                const std::array<std::string, 3> prefixes{
                    "place ", "the ", "a "};
                bool changed = true;
                while (changed) {
                    changed = false;
                    const std::string lower = lowercase(value);
                    for (const auto &prefix : prefixes) {
                        if (lower.rfind(prefix, 0) == 0) {
                            value = trim(value.substr(prefix.size()));
                            changed = true;
                            break;
                        }
                    }
                }
                return value;
            };
            const std::string object_name = remove_prefix(
                parsed.prompt.substr(0, separator_position));
            const std::string destination_name = remove_prefix(
                parsed.prompt.substr(separator_position + separator.size()));
            if (object_name.empty() || destination_name.empty()) {
                response->response =
                    "PLACEMENT_PROMPT_INVALID: object or destination is empty";
                return;
            }

            SynchronizedObservation observation;
            std::string observation_status;
            if (!captureFreshObservation(observation, observation_status)) {
                response->response = observation_status;
                return;
            }
            if (!edgesam_) {
                response->response =
                    "EDGESAM_UNAVAILABLE: " + edgesam_initialization_error_;
                return;
            }
            const cv::Mat image = observation.image;
            const auto capture_dir = graspCaptureDirectory();
            if (!capture_dir.empty()) {
                for (const char* name : {"placement_robobrain_input.jpg",
                                         "placement_bounding_box.png",
                                         "placement_target_mask.png",
                                         "placement_segmented.png",
                                         "destination_bounding_box.png",
                                         "destination_mask.png",
                                         "destination_segmented.png"}) {
                    std::error_code error;
                    std::filesystem::remove(capture_dir / name, error);
                    if (error) {
                        RCLCPP_WARN(this->get_logger(),
                                    "Could not clear previous %s: %s",
                                    name, error.message().c_str());
                    }
                }
                saveCaptureImage(capture_dir / "placement_raw.png", image);
            }
            vlm_node::EdgeSamResult segmentation;
            try {
                const auto target=resolveActiveTarget(observation,destination_name,true);
                segmentation.mask=cv_bridge::toCvCopy(target->mask,"mono8")->image;
                segmentation.mask_pixels=cv::countNonZero(segmentation.mask);
                segmentation.score=0.F;  // Resolver does not claim a new segmentation confidence.
                if (!capture_dir.empty()) {
                    auto boxed=image.clone();
                    cv::rectangle(boxed,cv::Point(target->pixel_box[0],target->pixel_box[1]),
                        cv::Point(target->pixel_box[2],target->pixel_box[3]),cv::Scalar(0,255,0),2);
                    saveCaptureImage(capture_dir / "placement_bounding_box.png",boxed);
                    saveCaptureImage(capture_dir / "destination_bounding_box.png",boxed);
                }
            } catch (const std::exception &error) {
                response->response=std::string("PLACEMENT_TARGET_FAILED: ")+error.what();return;
            }
            if (!capture_dir.empty()) {
                saveCaptureImage(capture_dir / "placement_target_mask.png",
                                 segmentation.mask);
                saveCaptureImage(capture_dir / "destination_mask.png",
                                 segmentation.mask);
                cv::Mat segmented_image = cv::Mat::zeros(image.size(), image.type());
                image.copyTo(segmented_image, segmentation.mask);
                saveCaptureImage(capture_dir / "placement_segmented.png",
                                 segmented_image);
                saveCaptureImage(capture_dir / "destination_segmented.png",
                                 segmented_image);
                RCLCPP_INFO(this->get_logger(), "[PLACE IMAGES] %s",
                            capture_dir.c_str());
            }
            if (segmentation.mask_pixels < edgesam_min_mask_pixels_) {
                response->response =
                    "PLACEMENT_MASK_TOO_SMALL: pixels=" +
                    std::to_string(segmentation.mask_pixels);
                return;
            }
            RCLCPP_INFO(
                this->get_logger(),
                "[PLACE] destination=%s mask=%d px score=%.3f policy=%s encoder_ms=%.3f decoder_ms=%.3f",
                destination_name.c_str(), segmentation.mask_pixels,
                segmentation.score,
                upright_policy ? "upright" : "free", segmentation.encoder_ms, segmentation.decoder_ms);

            if (!placement_candidates_client_->wait_for_service(
                    std::chrono::seconds(5))) {
                response->response =
                    "PLACEMENT_BACKEND_UNAVAILABLE: "
                    "/pipeline/find_placements";
                return;
            }
            auto placement_request = std::make_shared<
                my_actions::srv::FindPlacementCandidates::Request>();
            placement_request->observation_header = observation.header;
            placement_request->destination_mask = *cv_bridge::CvImage(
                observation.header, "mono8", segmentation.mask).toImageMsg();
            placement_request->orientation_policy =
                upright_policy ? "upright" : "free";
            placement_request->object_name = object_name;
            placement_request->destination_name = destination_name;
            placement_request->support_mode =
                this->get_parameter("placement_support_mode").as_string();
            if (!upright_policy && placement_request->support_mode == "floor_slots") {
                placement_request->support_mode = "auto";
            }
            auto future = placement_candidates_client_->async_send_request(
                placement_request);
            if (future.wait_for(std::chrono::seconds(request_timeout_sec_)) !=
                std::future_status::ready) {
                response->response = "PLACEMENT_BACKEND_TIMEOUT";
                return;
            }
            const auto result = future.get();
            if (!result->success) {
                response->response = result->status;
                return;
            }
            if (!placement_plan_requested) {
                response->response =
                    "PLACEMENT_CANDIDATES_READY: " + result->status;
                return;
            }

            CachedGraspPlan cached_plan;
            std::string cache_status;
            std::string cached_object_name = object_name;
            if (!takeFreshSafeCachedPlan(
                    cached_object_name, cached_plan, cache_status)) {
                const auto implicit = singleCachedGraspObject();
                if (!implicit || !takeFreshSafeCachedPlan(
                        *implicit, cached_plan, cache_status)) {
                    response->response =
                        "PLACEMENT_" + cache_status;
                    return;
                }
                cached_object_name = *implicit;
            }
            if (!plan_placement_client_->wait_for_service(
                    std::chrono::seconds(5))) {
                response->response =
                    "PLACEMENT_PLANNER_UNAVAILABLE: /pipeline/plan_placement";
                return;
            }
            auto plan_request =
                std::make_shared<my_actions::srv::PlanPlacement::Request>();
            plan_request->grasp_candidate = cached_plan.candidate;
            plan_request->source_object_center = result->source_object_center;
            plan_request->candidates = result->candidates;
            plan_request->object_name = cached_object_name;
            plan_request->destination_name = destination_name;
            plan_request->orientation_policy =
                upright_policy ? "upright" : "free";
            plan_request->validation_batch_size = 8;
            plan_request->retry_motion_failures = true;
            auto plan_future = plan_placement_client_->async_send_request(
                plan_request);
            if (plan_future.wait_for(std::chrono::seconds(30)) !=
                std::future_status::ready) {
                response->response = "PLACEMENT_PLANNER_TIMEOUT";
                return;
            }
            const auto plan_result = plan_future.get();
            if (plan_result->success && placement_request->support_mode == "floor_slots") {
                if (!commit_floor_slots_client_->wait_for_service(std::chrono::seconds(3))) {
                    response->response = "FLOOR_SLOT_COMMIT_UNAVAILABLE";
                    return;
                }
                auto commit = std::make_shared<my_actions::srv::CommitFloorSlots::Request>();
                commit->selected_pose = plan_result->selected_candidate.object_pose;
                auto committed = commit_floor_slots_client_->async_send_request(commit);
                if (committed.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
                    response->response = "FLOOR_SLOT_COMMIT_TIMEOUT_STATE_UNKNOWN";
                    return;
                }
                const auto reservation = committed.get();
                if (!reservation->success) {
                    response->response = reservation->status;
                    return;
                }
            }
            response->response = plan_result->success ?
                plan_result->status : "PLACEMENT_PLAN_FAILED: " +
                plan_result->status;
            return;
        }

        if (parsed.mode == "GRASP" || parsed.mode == "FIND_GRASP" ||
            parsed.mode == "GRAB" || parsed.mode == "PRE_GRASP" ||
            parsed.mode == "REFINE_GRASP") {
            const bool execute_requested =
                parsed.mode == "GRASP" || parsed.mode == "GRAB";
            const bool pregrasp_requested = parsed.mode == "PRE_GRASP";
            const bool refine_requested =
                pregrasp_requested || parsed.mode == "REFINE_GRASP";
            const std::optional<std::string> implicit_object =
                parsed.prompt.empty() ? singleCachedGraspObject() :
                std::nullopt;
            if (parsed.prompt.empty() && !implicit_object) {
                response->response =
                    "AMBIGUOUS_CACHED_GRASP: include the object name";
                return;
            }
            const std::string requested_object = implicit_object ?
                *implicit_object : extractGraspTarget(parsed.prompt);
            const std::string world_name = implicit_object ?
                *implicit_object : targetWorldName(requested_object);
            const std::string object_name =
                world_name.empty() ? requested_object : world_name;
            // GRASP (with GRAB retained as an alias) is intentionally a
            // two-step operation: it executes only
            // the #1 plan produced by a recent FIND_GRASP.  It never silently
            // re-grounds and replaces the RViz-reviewed pose.
            if (execute_requested || refine_requested) {
                CachedGraspPlan cached_plan;
                std::string cache_status;
                if (!takeFreshSafeCachedPlan(
                        object_name, cached_plan, cache_status)) {
                    response->response = cache_status;
                    return;
                }
                if (!execute_grasp_client_->wait_for_service(
                        std::chrono::seconds(3))) {
                    response->response =
                        "MASTER_UNAVAILABLE: /pipeline/execute_grasp; no motion was started";
                    return;
                }
                if (refine_requested) {
                    std::ostringstream summary;
                    summary << cache_status;
                    // PRE_GRASP is deliberately deterministic: move only to
                    // the RViz-reviewed, rank-#1 HGGD pose.  REFINE_GRASP is
                    // retained as a diagnostic mode which may try the other
                    // safe cached seeds when a local cross-section fails.
                    const std::vector<my_actions::msg::GraspCandidate> candidates =
                        pregrasp_requested ?
                        std::vector<my_actions::msg::GraspCandidate>{
                            cached_plan.candidate} : cached_plan.candidates;
                    const std::string operation =
                        pregrasp_requested ? "PRE_GRASP" : "REFINE_GRASP";
                    for (const auto& candidate : candidates) {
                        auto command = std::make_shared<
                            my_actions::srv::ExecuteGrasp::Request>();
                        command->object_name = cached_plan.object_name;
                        command->observation_header =
                            cached_plan.observation_header;
                        command->candidate = candidate;
                        command->execute = true;
                        command->refine_only = true;
                        auto command_future =
                            execute_grasp_client_->async_send_request(command);
                        if (command_future.wait_for(std::chrono::seconds(
                                std::max(request_timeout_sec_, 300))) !=
                            std::future_status::ready) {
                            response->response =
                                operation +
                                "_TIMEOUT: motion state is unknown; "
                                "inspect the arm before another command";
                            return;
                        }
                        const auto result = command_future.get();
                        summary << "\nCANDIDATE #" << candidate.id << ": "
                                << result->status;
                        if (result->success) {
                            summary << "\n" << operation << "_SUCCESS";
                            response->response = summary.str();
                            return;
                        }
                    }
                    summary << "\n" << operation << "_FAILED: "
                            << (pregrasp_requested ?
                                "rank-#1 HGGD seed failed local refinement or "
                                "strict final IK validation" :
                                "no cached HGGD seed passed local refinement "
                                "and strict final IK validation");
                    response->response = summary.str();
                    return;
                }
                auto command =
                    std::make_shared<my_actions::srv::ExecuteGrasp::Request>();
                command->object_name = cached_plan.object_name;
                command->observation_header = cached_plan.observation_header;
                command->candidate = cached_plan.candidate;
                command->execute = true;
                command->refine_only = false;
                auto command_future =
                    execute_grasp_client_->async_send_request(command);
                if (command_future.wait_for(
                        std::chrono::seconds(std::max(request_timeout_sec_, 300))) !=
                    std::future_status::ready) {
                    response->response =
                        "GRAB_TIMEOUT: execution state is unknown; inspect the arm before another command";
                    return;
                }
                const auto command_result = command_future.get();
                std::ostringstream summary;
                summary << cache_status << "\nMASTER: " << command_result->status;
                if (command_result->motion_completed) {
                    const bool verified = command_result->gripper_contact;
                    summary << "\nVERIFY: gripper_contact="
                            << command_result->gripper_contact
                            << " robobrain=BYPASSED"
                            << (verified ? "\nGRAB_SUCCESS" : "\nGRAB_FAILED");
                }
                response->response = summary.str();
                return;
            }
            SynchronizedObservation observation;
            std::string fresh_status;
            if (!captureFreshObservation(observation, fresh_status)) {
                response->response = fresh_status;
                return;
            }
            cv::Mat image = observation.image;
            std::string presence_status;
            if (!targetVisible(
                    observation, parsed.prompt, world_name,
                    presence_status)) {
                if (!execute_requested) {
                    response->response =
                        "NOT_IN_CURRENT_VIEW: " + presence_status;
                    return;
                }
                std::string reposition_status;
                if (!repositionToStoredTarget(world_name, reposition_status)) {
                    response->response = reposition_status;
                    return;
                }
                if (!captureFreshObservation(observation, fresh_status)) {
                    response->response = fresh_status;
                    return;
                }
                image = observation.image;
                if (!targetVisible(
                        observation, parsed.prompt, world_name,
                        presence_status)) {
                    response->response =
                        "NOT_FOUND_AFTER_NBV: " + presence_status;
                    return;
                }
            }
            if (world_perception_backend_ == "robobrain") {
                refreshTrackedObjects(observation);
            }

            const std::string lower_prompt = lowercase(parsed.prompt);
            std::string grounding_target =
                "the complete visible object " + object_name;
            if (lower_prompt.find("handle") != std::string::npos) {
                grounding_target =
                    "the complete visible handle of " + object_name;
            } else if (lower_prompt.find("rim") != std::string::npos ||
                       lower_prompt.find("lip") != std::string::npos ||
                       lower_prompt.find("edge") != std::string::npos) {
                std::string whole_object = object_name;
                for (const std::string phrase : {
                         " of the ", " of an ", " of a "}) {
                    const auto position = lower_prompt.rfind(phrase);
                    if (position != std::string::npos) {
                        whole_object = trim(parsed.prompt.substr(
                            position + phrase.size()));
                        break;
                    }
                }
                grounding_target =
                    "Locate the whole " + whole_object +
                    ", including its body, handle, and rim. Return only "
                    "its bounding box [x1,y1,x2,y2] in normalized "
                    "coordinates 0 to 1000.";
            } else if (lower_prompt.find("body") != std::string::npos ||
                       lower_prompt.find("side") != std::string::npos) {
                grounding_target =
                    "the complete visible body of " + object_name;
            }
            const bool whole_object_for_rim =
                lower_prompt.find("rim") != std::string::npos ||
                lower_prompt.find("lip") != std::string::npos ||
                lower_prompt.find("edge") != std::string::npos;
            if (!whole_object_for_rim) {
                grounding_target +=
                    ". Use one tight bounding box enclosing the entire named "
                    "target, not a smaller grasp or contact patch";
            }
            const auto capture_dir = graspCaptureDirectory();
            if (!capture_dir.empty()) {
                for (const char* name : {"robobrain_input.jpg", "bounding_box.png",
                                         "target_mask.png", "segmented.png",
                                         "hggd_heatmap.png"}) {
                    std::error_code error;
                    std::filesystem::remove(capture_dir / name, error);
                    if (error) {
                        RCLCPP_WARN(this->get_logger(),
                                    "Could not clear previous %s: %s",
                                    name, error.message().c_str());
                    }
                }
                saveCaptureImage(capture_dir / "raw.png", image);
            }
            my_actions::srv::ResolveManipulationTarget::Response::SharedPtr active_target;
            try { active_target=resolveActiveTarget(observation,object_name,false); }
            catch (const std::exception &error) {response->response=error.what();return;}
            int x1=active_target->pixel_box[0],y1=active_target->pixel_box[1];
            int x2=active_target->pixel_box[2],y2=active_target->pixel_box[3];
            cv::Mat boxed_image = image.clone();
            cv::rectangle(boxed_image, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);
            if (!capture_dir.empty()) {
                saveCaptureImage(capture_dir / "bounding_box.png", boxed_image);
            }
            // HGGD supplies the spatial grasp seeds. Keep only a mechanical
            // ROI anchor for the existing service contract; the planner uses
            // depth continuity to select foreground and never treats this
            // rectangle centre as a semantic grasp point.
            int u = (x1 + x2) / 2;
            int v = (y1 + y2) / 2;

            cv::Mat target_mask=cv_bridge::toCvCopy(active_target->mask,"mono8")->image;
            std::string target_mask_source = active_target->reused ? "active memory" : "EdgeSAM box-prompted mask";
            if (target_mask.empty() && grasp_target_mask_backend_ == "edgesam") {
                if (!edgesam_) {
                    response->response =
                        "EDGESAM_UNAVAILABLE: " + edgesam_initialization_error_;
                    return;
                }
                try {
                    const cv::Rect grounding_roi(
                        x1, y1, std::max(1, x2 - x1 + 1),
                        std::max(1, y2 - y1 + 1));
                    const auto segmentation =
                        segmentObject(image, grounding_roi, observation.header);
                    if (segmentation.mask_pixels < edgesam_min_mask_pixels_) {
                        response->response =
                            "EDGESAM_MASK_TOO_SMALL: pixels=" +
                            std::to_string(segmentation.mask_pixels);
                        return;
                    }
                    target_mask = segmentation.mask;
                    target_mask_source = "EdgeSAM box-prompted mask";
                    RCLCPP_INFO(
                        this->get_logger(),
                        "[EDGESAM] mask=%d px score=%.3f encoder=%.1fms "
                        "decoder=%.1fms",
                        segmentation.mask_pixels, segmentation.score,
                        segmentation.encoder_ms, segmentation.decoder_ms);
                } catch (const std::exception &error) {
                    response->response =
                        std::string("EDGESAM_INFERENCE_FAILED: ") + error.what();
                    return;
                }
            }
            RCLCPP_INFO(
                this->get_logger(),
                "[GRASP] Grounded ROI=[%d,%d]-[%d,%d] mask=%s; "
                "HGGD will provide spatial seeds.",
                x1, y1, x2, y2, target_mask_source.c_str());
            if (!capture_dir.empty()) {
                saveCaptureImage(capture_dir / "target_mask.png", target_mask);
                cv::Mat segmented_image = cv::Mat::zeros(image.size(), image.type());
                image.copyTo(segmented_image, target_mask);
                saveCaptureImage(capture_dir / "segmented.png", segmented_image);
                RCLCPP_INFO(this->get_logger(), "[GRASP IMAGES] %s",
                            capture_dir.c_str());
            }

            if (!grasp_candidates_client_->wait_for_service(std::chrono::seconds(3))) {
                response->response = "VAMP_UNAVAILABLE: /pipeline/calculate_grasps";
                return;
            }
            auto grasp_request = std::make_shared<my_actions::srv::FindGraspCandidates::Request>();
            grasp_request->observation_header = observation.header;
            grasp_request->seed_u = u; grasp_request->seed_v = v;
            grasp_request->image_width = image.cols; grasp_request->image_height = image.rows;
            grasp_request->target_mask = *cv_bridge::CvImage(
                observation.header, "mono8", target_mask).toImageMsg();
            grasp_request->target_description = object_name;
            grasp_request->affordance_type = lower_prompt.find("handle") != std::string::npos ? "HANDLE" :
                (lower_prompt.find("rim") != std::string::npos ||
                 lower_prompt.find("lip") != std::string::npos ||
                 lower_prompt.find("edge") != std::string::npos) ? "RIM" :
                (lower_prompt.find("body") != std::string::npos ||
                 lower_prompt.find("side") != std::string::npos) ? "BODY" : "AUTO";
            auto future = grasp_candidates_client_->async_send_request(grasp_request);
            if (future.wait_for(std::chrono::seconds(request_timeout_sec_)) != std::future_status::ready) {
                response->response = "GRASP_TIMEOUT";
                return;
            }
            const auto result = future.get();
            const auto &corrected_target = result->corrected_seed_base;
            if (std::isfinite(corrected_target.x) &&
                std::isfinite(corrected_target.y) &&
                std::isfinite(corrected_target.z) &&
                (std::abs(corrected_target.x) + std::abs(corrected_target.y) +
                 std::abs(corrected_target.z)) > 1.0e-6) {
                geometry_msgs::msg::PointStamped target_point;
                target_point.header = observation.header;
                target_point.header.frame_id = "base_link";
                target_point.point = corrected_target;
                reportObjectObservation(object_name, true, target_point);
                RCLCPP_INFO(
                    this->get_logger(),
                    "[WORLD REFRESH] Using VAMP-corrected target "
                    "base_link=(%.3f,%.3f,%.3f)",
                    corrected_target.x, corrected_target.y,
                    corrected_target.z);
            }
            std::ostringstream summary;
            summary << result->status << ": " << result->candidates.size() << " dry-run candidates";
            for (const auto &candidate : result->candidates) {
                summary << "\n#" << candidate.id << " base_link pose=["
                        << candidate.grasp_pose.pose.position.x << ", "
                        << candidate.grasp_pose.pose.position.y << ", "
                        << candidate.grasp_pose.pose.position.z << "] width="
                        << candidate.jaw_opening_m << "m proposal_score=" << candidate.score
                        << " rank=" << candidate.ranking_score
                        << " ik=" << candidate.ik_valid
                        << " measured_contacts="
                        << candidate.measured_contacts_valid
                        << " silhouette_contacts="
                        << candidate.silhouette_contacts_valid
                        << " validation=" << candidate.validation_status
                        << " grasp_collision="
                        << (candidate.collision_checked ?
                            (candidate.grasp_collision_valid ? "valid" : "invalid") :
                            "not_checked")
                        << " pregrasp_collision="
                        << (candidate.collision_checked ?
                            (candidate.pregrasp_collision_valid ? "valid" : "override") :
                            "not_checked");
                summary << " grasp_joints=[";
                for (size_t index = 0; index < candidate.grasp_joints.size(); ++index) {
                    if (index > 0) summary << ",";
                    summary << candidate.grasp_joints[index];
                }
                summary << "]";
            }

            if (!result->success || result->candidates.empty()) {
                response->response = summary.str();
                return;
            }

            // GRAB still executes the RViz-reviewed #1 candidate. Keep the
            // remaining safe seeds so REFINE_GRASP can advance to the next
            // one when local reconstruction rejects a cross-section.
            cacheSafeGraspPlans(
                object_name, observation.header, result->candidates);

            if (execute_grasp_client_->wait_for_service(std::chrono::milliseconds(500))) {
                auto command = std::make_shared<my_actions::srv::ExecuteGrasp::Request>();
                command->object_name = object_name;
                command->observation_header = observation.header;
                command->candidate = result->candidates.front();
                command->execute = execute_requested;
                command->refine_only = false;
                auto command_future = execute_grasp_client_->async_send_request(command);
                const auto command_timeout = std::chrono::seconds(
                    execute_requested ? std::max(request_timeout_sec_, 300) : 3);
                if (command_future.wait_for(command_timeout) ==
                    std::future_status::ready) {
                    const auto command_result = command_future.get();
                    summary << "\nMASTER: " << command_result->status;
                    if (execute_requested) {
                        bool visual_success = false;
                        std::string visual_reply = "NOT_RUN";
                        if (command_result->motion_completed) {
                            SynchronizedObservation verification_observation;
                            std::string verification_status;
                            if (captureFreshObservation(
                                    verification_observation,
                                    verification_status)) {
                                visual_reply = trim(callVLM(
                                    "",
                                    "Check whether the robot gripper is visibly "
                                    "holding the requested object: " + object_name +
                                    ". Reply with exactly SUCCESS or FAILURE. "
                                    "If uncertain, reply FAILURE.",
                                    16, verification_observation.image));
                                std::string normalized_reply = visual_reply;
                                std::transform(
                                    normalized_reply.begin(),
                                    normalized_reply.end(),
                                    normalized_reply.begin(),
                                    [](unsigned char value) {
                                      return static_cast<char>(std::toupper(value));
                                    });
                                visual_success = normalized_reply == "SUCCESS";
                            } else {
                                visual_reply = verification_status;
                            }
                        }
                        const bool verified =
                            command_result->motion_completed &&
                            command_result->gripper_contact && visual_success;
                        summary << "\nVERIFY: gripper_contact="
                                << command_result->gripper_contact
                                << " robobrain=" << visual_reply;
                        summary << (verified ?
                            "\nGRAB_SUCCESS" : "\nGRAB_FAILED");
                    }
                } else {
                    summary <<
                        "\nMASTER: grasp request timed out; execution state is "
                        "unknown. Do not send another motion command until "
                        "the arm and /pipeline/execute_grasp are checked.";
                }
            } else if (execute_requested) {
                summary << "\nGRAB_FAILED: master execution service is unavailable; no motion was started";
            } else {
                summary << "\nPLAN_NOT_CACHED: master is not running (RViz result remains valid)";
            }
            response->response = summary.str();
            return;
        }

        const std::set<std::string> official_modes = {
            "GENERAL", "POINTING", "GROUNDING", "TRAJECTORY"
        };
        if (official_modes.count(parsed.mode) > 0) {
            cv::Mat current_image;
            if (!parsed.image_path.empty()) {
                current_image = cv::imread(parsed.image_path, cv::IMREAD_COLOR);
                if (current_image.empty()) {
                    response->response =
                        "Could not read image path: " + parsed.image_path;
                    RCLCPP_ERROR(
                        this->get_logger(), "%s", response->response.c_str());
                    return;
                }
            } else {
                std::lock_guard<std::mutex> lock(image_mutex_);
                if (!latest_image_.empty()) {
                    current_image = latest_image_;
                }
            }

            if (current_image.empty()) {
                response->response =
                    parsed.mode + " requires an image. Add "
                    "[IMAGE:/absolute/path.jpg] after the mode flag or start "
                    "the camera driver.";
                RCLCPP_ERROR(
                    this->get_logger(), "%s", response->response.c_str());
                return;
            }

            RCLCPP_INFO(
                this->get_logger(),
                "[ROBOBRAIN MODE] flag=[%s], image_source=%s",
                parsed.mode.c_str(),
                parsed.image_path.empty()
                    ? (current_image.empty() ? "none" : "camera")
                    : parsed.image_path.c_str());

            if (parsed.mode == "GROUNDING") {
                RCLCPP_INFO(
                    this->get_logger(),
                    "[ROBOBRAIN GROUNDING] Stage 1/2: checking whether target "
                    "is present before requesting coordinates.");
                const std::string presence_reply = callVLM(
                    "", prepareGroundingPresencePrompt(parsed.prompt),
                    16, current_image);
                if (presence_reply.empty()) {
                    response->response =
                        "GROUNDING_ABORTED: object-presence check failed to "
                        "return a response.";
                    RCLCPP_ERROR(
                        this->get_logger(), "[ROBOBRAIN GROUNDING] %s",
                        response->response.c_str());
                    return;
                }
                if (!presenceResponseIsAffirmative(presence_reply)) {
                    response->response =
                        "NOT_FOUND: grounding was not attempted because the "
                        "object-presence check returned: " +
                        trim(presence_reply);
                    RCLCPP_WARN(
                        this->get_logger(), "[ROBOBRAIN GROUNDING] %s",
                        response->response.c_str());
                    return;
                }
                RCLCPP_INFO(
                    this->get_logger(),
                    "[ROBOBRAIN GROUNDING] Presence check returned PRESENT. "
                    "Stage 2/2: requesting the bounding box.");
            }

            const std::string prepared_prompt = prepareOfficialPrompt(
                parsed.mode, parsed.prompt);

            const std::string reply = parsed.mode == "GROUNDING" ?
                groundObject(current_image, parsed.prompt, std_msgs::msg::Header{}) :
                callVLM("", prepared_prompt, max_tokens_, current_image);
            if (reply.empty()) {
                response->response = "RoboBrain failed to return a response.";
                return;
            }
            if (parsed.mode == "GROUNDING") {
                std::smatch box;
                if (!vlm_node::parseGroundingBox(reply, box)) {
                    response->response = "GROUNDING_PARSE_FAILED: " + reply;
                    return;
                }
            }
            response->response = reply;
            parseAndLogOfficialResponse(parsed.mode, reply);
            return;
        }

        const std::string user_prompt = parsed.prompt;
        std::string prompt_lower = user_prompt;
        std::transform(prompt_lower.begin(), prompt_lower.end(), prompt_lower.begin(), [](unsigned char c){ return std::tolower(c); });

        if (parsed.mode == "YOLO")
        {
            // Mode 1: Scene Analysis via YOLO + VLM (Refactored to use vision_node service)
            RCLCPP_INFO(this->get_logger(), "=== MODE: SCENE ANALYSIS (VISION NODE YOLO+VLM) ===");
            spatial_memory_.clear();
            
            cv::Mat current_image;
            {
                std::lock_guard<std::mutex> lock(image_mutex_);
                if (!latest_image_.empty()) {
                    latest_image_.copyTo(current_image);
                }
            }
            
            if (current_image.empty()) {
                response->response = "No image available.";
                return;
            }

            if (!get_yolo_client_->wait_for_service(std::chrono::seconds(3))) {
                response->response = "Failed to connect to /active_vision/get_yolo_inference service.";
                return;
            }

            auto req = std::make_shared<my_actions::srv::GetYoloInference::Request>();
            auto future = get_yolo_client_->async_send_request(req);
            
            std::string summary = "Detection Results:\n";
            int count = 0;
            
            if (future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
                auto res = future.get();
                if (res->success) {
                    for (size_t i = 0; i < res->classes.size(); ++i) {
                        std::string cls_name = res->classes[i];
                        
                        bool match = false;
                        if (prompt_lower.find("all") != std::string::npos || prompt_lower.find(cls_name) != std::string::npos) match = true;
                        if (prompt_lower.find("mug") != std::string::npos && cls_name == "cup") match = true;
                        if (prompt_lower.find("detect") != std::string::npos && prompt_lower.find("all") == std::string::npos) match = true;

                        if (!match) continue;

                        DetectedObject obj;
                        obj.name = cls_name + "_" + std::to_string(count + 1);
                        
                        int x1 = res->bbox_x_min[i];
                        int y1 = res->bbox_y_min[i];
                        int x2 = res->bbox_x_max[i];
                        int y2 = res->bbox_y_max[i];
                        
                        x1 = std::max(0, x1);
                        y1 = std::max(0, y1);
                        x2 = std::min(current_image.cols - 1, x2);
                        y2 = std::min(current_image.rows - 1, y2);
                        
                        cv::Rect bbox(x1, y1, x2 - x1, y2 - y1);
                        obj.bbox = bbox;
                        
                        if (bbox.area() > 0) {
                            cv::Mat crop = current_image(bbox).clone();
                            std::string sys_prompt = "You are a helpful assistant. Be brief.";
                            std::string user_prompt = "Describe the main object in this image briefly in one short sentence. Focus on its color and type.";
                            std::string desc = callVLM(sys_prompt, user_prompt, 50, crop);
                            if (desc.empty()) desc = "Unknown object";
                            std::replace(desc.begin(), desc.end(), '\n', ' ');
                            obj.description = desc;
                        } else {
                            obj.description = "Unknown object";
                        }
                        
                        obj.x = res->position_x[i];
                        obj.y = res->position_y[i];
                        obj.z = res->position_z[i];
                        
                        active_target_x_ = obj.x;
                        active_target_y_ = obj.y;
                        active_target_z_ = obj.z;
                        has_active_target_ = true;
                        
                        spatial_memory_.push_back(obj);
                        summary += "- " + obj.name + ": " + obj.description + "\n";
                        summary += "  Location: [" + std::to_string(obj.x) + ", " + std::to_string(obj.y) + ", " + std::to_string(obj.z) + "]\n";
                        count++;
                    }
                } else {
                    summary = "YOLO Inference failed.\n";
                }
            } else {
                summary = "YOLO Inference timed out.\n";
            }
            response->response = summary;
        } else {
            // Mode 2: Action
            RCLCPP_INFO(this->get_logger(), "=== MODE: ACTION ===");
            std::string memory_str = "Known Objects in Spatial Memory:\n";
            if (spatial_memory_.empty()) {
                memory_str += "None.\n";
            } else {
                for (const auto& obj : spatial_memory_) {
                    memory_str += "- " + obj.name + ": " + obj.description + "\n";
                    memory_str += "  Location: " + std::to_string(obj.x) + " " + std::to_string(obj.y) + " " + std::to_string(obj.z) + "\n";
                }
            }

            std::string sys_prompt = 
                memory_str + "\n"
                "Task: " + user_prompt + "\n"
                "You must pick ONE object from the Known Objects that best matches the Task description.\n"
                "Output EXACTLY ONE action based on the chosen object's location. Do NOT write code. You MUST use the exact x, y, and z coordinates of the object from the Known Objects list.\n"
                "Available actions:\n"
                "- LOOK_AT_3D <x> <y> <z>\n"
                "- LOOK_AT <pixel_x> <pixel_y>\n"
                "- RELATIVE_MOVE <dx> <dy> <dz>\n"
                "- GRAB\n"
                "\n"
                "Now, output the Action:\n";
                
            RCLCPP_INFO(this->get_logger(), "Sending Prompt to VLM:\n%s", sys_prompt.c_str());
            
            cv::Mat action_image;
            {
                std::lock_guard<std::mutex> lock(image_mutex_);
                if (!latest_image_.empty()) {
                    action_image = latest_image_;
                }
            }
            if (action_image.empty()) {
                response->response =
                    "ACTION requires an image from /camera/color/image_raw.";
                return;
            }
            std::string sys_instruction = "You are a robotic task planner. Choose the next action based on the spatial memory provided.";
            std::string reply = callVLM(
                sys_instruction, sys_prompt, 200, action_image);
            
            if (reply.empty()) {
                response->response = "VLM failed to return a response.";
                return;
            }

            RCLCPP_INFO(this->get_logger(), "VLM Action Response:\n%s", reply.c_str());

            // Prompt-only mode is the safe default. Parsing remains visible in
            // logs, but dispatch and execution stay disabled.
            response->response = reply;

            std::string action_str = reply;
            
            if (action_str.find("RELATIVE_MOVE") != std::string::npos && has_active_target_) {
                std::regex rel_rgx("RELATIVE_MOVE\\s+([+-]?\\d*\\.?\\d+)\\s+([+-]?\\d*\\.?\\d+)\\s+([+-]?\\d*\\.?\\d+)");
                std::smatch rel_match;
                if (std::regex_search(action_str, rel_match, rel_rgx)) {
                    float dx = std::stof(rel_match[1]);
                    float dy = std::stof(rel_match[2]);
                    float dz = std::stof(rel_match[3]);
                    
                    float new_x = active_target_x_ + dx;
                    float new_y = active_target_y_ + dy;
                    float new_z = active_target_z_ + dz;
                    
                    action_str = "LOOK_AT_3D " + std::to_string(new_x) + " " + std::to_string(new_y) + " " + std::to_string(new_z);
                    RCLCPP_INFO(this->get_logger(), "Translated RELATIVE_MOVE to absolute: %s", action_str.c_str());
                }
            }
            
            if (action_str.find("LOOK_AT_3D") != std::string::npos) {
                std::regex look_rgx("LOOK_AT_3D\\s+([+-]?\\d*\\.?\\d+)\\s+([+-]?\\d*\\.?\\d+)\\s+([+-]?\\d*\\.?\\d+)");
                std::smatch look_match;
                if (std::regex_search(action_str, look_match, look_rgx)) {
                    float tx = std::stof(look_match[1]);
                    float ty = std::stof(look_match[2]);
                    float tz = std::stof(look_match[3]);
                    
                    // Find the best matching object name in memory based on distance
                    std::string best_name = "target object";
                    float min_dist = 999.0f;
                    for (const auto& obj : spatial_memory_) {
                        float dist = std::sqrt(std::pow(obj.x - tx, 2) + std::pow(obj.y - ty, 2) + std::pow(obj.z - tz, 2));
                        if (dist < min_dist && dist < 0.2f) { // Within 20cm
                            min_dist = dist;
                            best_name = obj.name;
                        }
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "Prepared LOOK_AT task for Master Node Pipeline: [%.3f, %.3f, %.3f] (Target: %s)", tx, ty, tz, best_name.c_str());

                    RCLCPP_INFO(
                        this->get_logger(),
                        "[ROBOBRAIN PARSE] Constructed task candidate: "
                        "OBSERVE_OBJECT target=%s position=[%.3f, %.3f, %.3f] "
                        "dispatch=%s",
                        best_name.c_str(), tx, ty, tz,
                        enable_task_dispatch_ ? "enabled" : "disabled");

                    if (!enable_task_dispatch_) {
                        return;
                    }
                    
                    if (append_task_client_->wait_for_service(std::chrono::seconds(2))) {
                        auto req = std::make_shared<my_actions::srv::AppendTask::Request>();
                        req->task_type = "LOOK_AT";
                        req->target_name = best_name;
                        req->coordinates = {tx, ty, tz};
                        
                        auto future_append = append_task_client_->async_send_request(req);
                        if (future_append.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
                            RCLCPP_INFO(this->get_logger(), "Task appended successfully. Triggering pipeline execution...");
                            if (execute_pipeline_client_->wait_for_service(std::chrono::seconds(2))) {
                                auto req_exec = std::make_shared<my_actions::srv::ExecutePipeline::Request>();
                                execute_pipeline_client_->async_send_request(req_exec);
                                response->response = "Executed Action: Appended LOOK_AT for " + best_name + " and triggered Pipeline execution.";
                            } else {
                                response->response = "Appended LOOK_AT task, but failed to connect to /pipeline/execute_pipeline.";
                            }
                        } else {
                            response->response = "Failed to append task. Service timed out.";
                        }
                    } else {
                        response->response = "Failed to connect to /pipeline/append_task service. Simulated Action: " + action_str;
                    }
                } else {
                    response->response = action_str;
                }
            } else {
                std::regex pixel_rgx(
                    "LOOK_AT\\s+([0-9]+)\\s+([0-9]+)");
                std::smatch pixel_match;
                if (std::regex_search(action_str, pixel_match, pixel_rgx)) {
                    RCLCPP_INFO(
                        this->get_logger(),
                        "[ROBOBRAIN PARSE] Parsed LOOK_AT pixel candidate: "
                        "x=%s y=%s. No pipeline task constructed because a "
                        "world-frame 3D target is required.",
                        pixel_match[1].str().c_str(),
                        pixel_match[2].str().c_str());
                    return;
                }
                response->response = "No recognized action found in output: " + reply;
                RCLCPP_WARN(
                    this->get_logger(),
                    "[ROBOBRAIN PARSE] No supported action matched; no task "
                    "was constructed.");
            }
        }
    }
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(VLMReasoningNode)

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VLMReasoningNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
