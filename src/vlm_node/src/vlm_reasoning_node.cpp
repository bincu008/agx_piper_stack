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
#include <my_actions/srv/plan_placement.hpp>
#include <my_actions/srv/get3_d_coordinate.hpp>
#include <my_actions/srv/update_object_observation.hpp>
#include <my_actions/srv/resolve_object.hpp>
#include <my_actions/srv/look_at.hpp>
#include <my_actions/srv/execute_grasp.hpp>
#include <my_actions/srv/get_world_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <Eigen/Dense>
#include "vlm_node/json.hpp"
#include "vlm_node/httplib.h"
#include "vlm_node/edge_sam_onnx.hpp"
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
        this->declare_parameter<std::string>("api_server_ip", "127.0.0.1");
        this->declare_parameter<int>("api_server_port", 1234);
        this->declare_parameter<std::string>("model_name", "robobrain2.5-4b");
        this->declare_parameter<int>("request_timeout_sec", 120);
        this->declare_parameter<bool>("enable_task_dispatch", false);
        this->declare_parameter<bool>("verbose", true);
        this->declare_parameter<double>("temperature", 0.1);
        this->declare_parameter<int>("max_tokens", 768);
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
        model_name_ = this->get_parameter("model_name").as_string();
        request_timeout_sec_ = this->get_parameter("request_timeout_sec").as_int();
        enable_task_dispatch_ = this->get_parameter("enable_task_dispatch").as_bool();
        verbose_ = this->get_parameter("verbose").as_bool();
        temperature_ = this->get_parameter("temperature").as_double();
        max_tokens_ = this->get_parameter("max_tokens").as_int();
        tracked_objects_ = this->get_parameter("tracked_objects").as_string_array();
        world_perception_backend_ =
            this->get_parameter("world_perception_backend").as_string();
        grasp_target_mask_backend_ =
            this->get_parameter("grasp_target_mask_backend").as_string();
        std::transform(grasp_target_mask_backend_.begin(),
            grasp_target_mask_backend_.end(), grasp_target_mask_backend_.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (grasp_target_mask_backend_ != "edgesam" &&
            grasp_target_mask_backend_ != "robobrain" &&
            grasp_target_mask_backend_ != "yolo_intersection") {
            RCLCPP_WARN(this->get_logger(),
                "Unknown grasp_target_mask_backend '%s'; using edgesam.",
                grasp_target_mask_backend_.c_str());
            grasp_target_mask_backend_ = "edgesam";
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
            "LM Studio endpoint: http://%s:%d (model=%s, task dispatch=%s, verbose=%s)",
            api_ip_.c_str(), api_port_, model_name_.c_str(),
            enable_task_dispatch_ ? "enabled" : "disabled",
            verbose_ ? "enabled" : "disabled");


        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10,
            std::bind(&VLMReasoningNode::imageCallback, this, std::placeholders::_1));
        observation_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/active_vision/observation_image",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&VLMReasoningNode::observationImageCallback, this, std::placeholders::_1));

        chat_srv_ = this->create_service<my_actions::srv::Chat>(
            "~/chat_prompt",
            std::bind(&VLMReasoningNode::handleChatPrompt, this, std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS(),
            cb_group_);
            
        check_api_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "~/check_api",
            std::bind(&VLMReasoningNode::handleCheckApi, this, std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS(),
            cb_group_);

        get_yolo_client_ = this->create_client<my_actions::srv::GetYoloInference>(
            "/active_vision/get_yolo_inference",
            rclcpp::ServicesQoS(),
            cb_group_);
        trigger_scan_client_ = this->create_client<std_srvs::srv::Trigger>(
            "/active_vision/trigger_scan", rclcpp::ServicesQoS(), cb_group_);
        grasp_candidates_client_ = this->create_client<my_actions::srv::FindGraspCandidates>(
            "/vamp_planner/find_grasp_candidates", rclcpp::ServicesQoS(), cb_group_);
        placement_candidates_client_ =
            this->create_client<my_actions::srv::FindPlacementCandidates>(
                "/anyplace/find_placement_candidates",
                rclcpp::ServicesQoS(), cb_group_);
        plan_placement_client_ =
            this->create_client<my_actions::srv::PlanPlacement>(
                "/vamp_planner/plan_placement",
                rclcpp::ServicesQoS(), cb_group_);
        get_3d_coordinate_client_ = this->create_client<my_actions::srv::Get3DCoordinate>(
            "/active_vision/get_3d_coordinate", rclcpp::ServicesQoS(), cb_group_);
        update_object_client_ = this->create_client<my_actions::srv::UpdateObjectObservation>(
            "/master/update_object_observation", rclcpp::ServicesQoS(), cb_group_);
        resolve_object_client_ = this->create_client<my_actions::srv::ResolveObject>(
            "/master/resolve_object", rclcpp::ServicesQoS(), cb_group_);
        look_at_client_ = this->create_client<my_actions::srv::LookAt>(
            "/master/look_at_target", rclcpp::ServicesQoS(), cb_group_);
        execute_grasp_client_ = this->create_client<my_actions::srv::ExecuteGrasp>(
            "/master/execute_grasp", rclcpp::ServicesQoS(), cb_group_);
        get_world_state_client_ = this->create_client<my_actions::srv::GetWorldState>(
            "/master/get_world_state", rclcpp::ServicesQoS(), cb_group_);

        append_task_client_ = this->create_client<my_actions::srv::AppendTask>(
            "/master/append_task",
            rclcpp::ServicesQoS(),
            cb_group_);
            
            
        execute_pipeline_client_ = this->create_client<my_actions::srv::ExecutePipeline>(
            "/master/execute_pipeline",
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
    std::string model_name_;
    int request_timeout_sec_;
    bool enable_task_dispatch_;
    bool verbose_;
    double temperature_;
    int max_tokens_;
    std::vector<std::string> tracked_objects_;
    std::string world_perception_backend_;
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
    std::mutex grasp_plan_cache_mutex_;
    std::unordered_map<std::string, CachedGraspPlan> cached_grasp_plans_;
    

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
        if (!trigger_scan_client_->wait_for_service(std::chrono::seconds(3))) {
            status = "VISION_UNAVAILABLE: /active_vision/trigger_scan";
            return false;
        }
        int64_t previous_stamp_ns = 0;
        {
            std::lock_guard<std::mutex> lock(observation_image_mutex_);
            if (!latest_observation_.image.empty()) {
                previous_stamp_ns = rclcpp::Time(latest_observation_.header.stamp).nanoseconds();
            }
        }
        auto trigger = trigger_scan_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
        if (trigger.wait_for(std::chrono::seconds(3)) != std::future_status::ready || !trigger.get()->success) {
            status = "FRESH_SCAN_TRIGGER_FAILED";
            return false;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(observation_image_mutex_);
                const int64_t stamp_ns = latest_observation_.image.empty() ? 0 :
                    rclcpp::Time(latest_observation_.header.stamp).nanoseconds();
                if (stamp_ns > previous_stamp_ns) {
                    observation.image = latest_observation_.image.clone();
                    observation.header = latest_observation_.header;
                    RCLCPP_INFO(this->get_logger(),
                                "[GRASP] Using fresh synchronized raw RGB-D observation at %ld.",
                                stamp_ns);
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        status = "FRESH_OBSERVATION_TIMEOUT: scan triggered but no newer synchronized RGB image arrived";
        return false;
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

    void handleCheckApi(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request;
        httplib::Client client(api_ip_, api_port_);
        client.set_connection_timeout(5);
        client.set_read_timeout(10);

        auto result = client.Get("/v1/models");
        if (!result) {
            response->success = false;
            response->message = "Cannot connect to LM Studio at http://" + api_ip_ + ":" +
                std::to_string(api_port_);
            return;
        }
        if (result->status != 200) {
            response->success = false;
            response->message = "LM Studio returned HTTP " + std::to_string(result->status);
            return;
        }

        try {
            const auto body = json::parse(result->body);
            bool model_found = false;
            if (body.contains("data") && body["data"].is_array()) {
                for (const auto& model : body["data"]) {
                    if (model.contains("id") && model["id"] == model_name_) {
                        model_found = true;
                        break;
                    }
                }
            }
            response->success = model_found;
            response->message = model_found
                ? "LM Studio is reachable and model '" + model_name_ + "' is available."
                : "LM Studio is reachable, but model '" + model_name_ + "' is not available.";
        } catch (const json::exception& error) {
            response->success = false;
            response->message = "Invalid /v1/models response: " + std::string(error.what());
        }
    }

    static std::string trim(const std::string& value) {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        const auto last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
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
                    "PLACE_FREE", "PLACE", "PLAN_UPRIGHT", "PLAN"
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
            return "Please provide the bounding box coordinate of the region "
                "this sentence describes: " + prompt + ".";
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

    bool saveOutputImage(
        const std::string& filename, const cv::Mat& image,
        const std::string& description) {
        (void)filename;
        (void)description;
        return !image.empty();
    }

    void saveGroundingStatusImage(
        const cv::Mat& source_image, const std::string& status) {
        if (source_image.empty()) {
            return;
        }
        cv::Mat annotated = source_image.clone();
        cv::putText(
            annotated, status, cv::Point(20, 45),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
        saveOutputImage(
            "robobrain_grounding.png", annotated,
            "grounding status annotation");
    }

    static int normalizedToPixel(double coordinate, int dimension) {
        const int pixel = static_cast<int>(
            std::lround(coordinate * static_cast<double>(dimension) / 1000.0));
        return std::clamp(pixel, 0, dimension - 1);
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
        const std::string reply = callVLM("", prompt.str(), 128, observation.image);
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
        std::lock_guard<std::mutex> lock(grasp_plan_cache_mutex_);
        const auto entry = cached_grasp_plans_.find(object_name);
        if (entry == cached_grasp_plans_.end()) {
            status = "NO_CACHED_GRASP: run [FIND_GRASP] first";
            return false;
        }
        const auto age = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - entry->second.created_at).count();
        if (age > grasp_plan_cache_ttl_sec_) {
            cached_grasp_plans_.erase(entry);
            status = "CACHED_GRASP_EXPIRED: run [FIND_GRASP] again";
            return false;
        }
        if (!candidateIsSafeToExecute(entry->second.candidate)) {
            status = "CACHED_GRASP_UNSAFE: run [FIND_GRASP] again";
            return false;
        }
        plan = entry->second;
        status = "SAFE_CACHED_GRASP age=" + std::to_string(age) + "s";
        return true;
    }

    std::optional<std::string> singleCachedGraspObject() {
        std::lock_guard<std::mutex> lock(grasp_plan_cache_mutex_);
        std::optional<std::string> object_name;
        const auto now = std::chrono::steady_clock::now();
        for (const auto& entry : cached_grasp_plans_) {
            const auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - entry.second.created_at).count();
            if (age > grasp_plan_cache_ttl_sec_ ||
                !candidateIsSafeToExecute(entry.second.candidate)) {
                continue;
            }
            if (object_name) return std::nullopt;
            object_name = entry.first;
        }
        return object_name;
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
        std::lock_guard<std::mutex> lock(grasp_plan_cache_mutex_);
        cached_grasp_plans_[object_name] = {
            object_name, observation_header, safe_candidates.front(),
            safe_candidates,
            std::chrono::steady_clock::now()};
        RCLCPP_INFO(
            this->get_logger(),
            "[GRASP] Cached %zu safe candidate(s) for '%s' for %d seconds.",
            safe_candidates.size(), object_name.c_str(),
            grasp_plan_cache_ttl_sec_);
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

    bool selectYoloTargetMask(
        const my_actions::srv::GetYoloInference::Response& result,
        const std::string& target_class, const cv::Rect& grounding_roi,
        int& seed_u, int& seed_v, cv::Mat& selected_mask,
        std::string& diagnostic) const {
        int selected_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < result.classes.size() &&
             index < result.masks.size(); ++index) {
            if (result.classes[index] != target_class) continue;
            try {
                const cv::Mat mask = cv_bridge::toCvCopy(
                    std::make_shared<sensor_msgs::msg::Image>(result.masks[index]),
                    "mono8")->image;
                if (mask.empty() ||
                    mask.cols < grounding_roi.x + grounding_roi.width ||
                    mask.rows < grounding_roi.y + grounding_roi.height) {
                    continue;
                }
                const cv::Rect image_bounds(0, 0, mask.cols, mask.rows);
                const cv::Rect overlap = grounding_roi & image_bounds;
                const int roi_mask_pixels = overlap.empty() ? 0 :
                    cv::countNonZero(mask(overlap));
                const bool contains_seed = seed_u >= 0 && seed_u < mask.cols &&
                    seed_v >= 0 && seed_v < mask.rows &&
                    mask.at<uint8_t>(seed_v, seed_u) != 0;
                const double confidence = index < result.confidences.size() ?
                    result.confidences[index] : 0.0;
                // The semantic point is the strongest association.  ROI
                // overlap resolves multiple cups when the point is in a gap.
                const double score = (contains_seed ? 1.0e9 : 0.0) +
                    static_cast<double>(roi_mask_pixels) + confidence;
                if (score > best_score) {
                    best_score = score;
                    selected_index = static_cast<int>(index);
                }
            } catch (const cv_bridge::Exception&) {
                continue;
            }
        }
        if (selected_index < 0) return false;
        try {
            selected_mask = cv_bridge::toCvCopy(
                std::make_shared<sensor_msgs::msg::Image>(
                    result.masks[static_cast<size_t>(selected_index)]),
                "mono8")->image;
        } catch (const cv_bridge::Exception&) {
            return false;
        }
        if (seed_u < 0 || seed_u >= selected_mask.cols || seed_v < 0 ||
            seed_v >= selected_mask.rows ||
            selected_mask.at<uint8_t>(seed_v, seed_u) == 0) {
            double best_distance_sq = std::numeric_limits<double>::infinity();
            cv::Point nearest(seed_u, seed_v);
            for (int v = 0; v < selected_mask.rows; ++v) {
                const auto* row = selected_mask.ptr<uint8_t>(v);
                for (int u = 0; u < selected_mask.cols; ++u) {
                    if (row[u] == 0) continue;
                    const double du = static_cast<double>(u - seed_u);
                    const double dv = static_cast<double>(v - seed_v);
                    const double distance_sq = du * du + dv * dv;
                    if (distance_sq < best_distance_sq) {
                        best_distance_sq = distance_sq;
                        nearest = cv::Point(u, v);
                    }
                }
            }
            if (!std::isfinite(best_distance_sq)) return false;
            seed_u = nearest.x;
            seed_v = nearest.y;
        }
        std::ostringstream detail;
        detail << "YOLO " << target_class << " mask index=" << selected_index
               << " pixels=" << cv::countNonZero(selected_mask)
               << " seed=(" << seed_u << "," << seed_v << ")";
        diagnostic = detail.str();
        return true;
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
            observation.image);
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
            status = "WORLD_STATE_UNAVAILABLE: /master/resolve_object";
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

    void annotateOfficialResponse(
        const std::string& mode, const std::string& response,
        const cv::Mat& source_image) {
        if (source_image.empty() || mode == "GENERAL") {
            return;
        }

        cv::Mat annotated = source_image.clone();
        bool drew_result = false;

        if (mode == "GROUNDING") {
            const std::regex box_pattern(
                "\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*"
                "([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
            for (auto match = std::sregex_iterator(
                     response.begin(), response.end(), box_pattern);
                 match != std::sregex_iterator(); ++match) {
                const cv::Point top_left(
                    normalizedToPixel(std::stod((*match)[1]), annotated.cols),
                    normalizedToPixel(std::stod((*match)[2]), annotated.rows));
                const cv::Point bottom_right(
                    normalizedToPixel(std::stod((*match)[3]), annotated.cols),
                    normalizedToPixel(std::stod((*match)[4]), annotated.rows));
                cv::rectangle(
                    annotated, top_left, bottom_right,
                    cv::Scalar(0, 255, 0), 3);
                cv::putText(
                    annotated, "grounding", top_left + cv::Point(0, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
                drew_result = true;
            }
            if (drew_result) {
                saveOutputImage(
                    "robobrain_grounding.png", annotated,
                    "grounding annotation");
            }
            return;
        }

        if (mode == "POINTING") {
            const std::regex point_pattern(
                "\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\)");
            for (auto match = std::sregex_iterator(
                     response.begin(), response.end(), point_pattern);
                 match != std::sregex_iterator(); ++match) {
                const cv::Point point(
                    normalizedToPixel(std::stod((*match)[1]), annotated.cols),
                    normalizedToPixel(std::stod((*match)[2]), annotated.rows));
                cv::drawMarker(
                    annotated, point, cv::Scalar(0, 0, 255),
                    cv::MARKER_CROSS, 30, 3);
                cv::circle(
                    annotated, point, 10, cv::Scalar(0, 0, 255), 3);
                drew_result = true;
            }
            if (drew_result) {
                saveOutputImage(
                    "robobrain_pointing.png", annotated,
                    "pointing annotation");
            }
            return;
        }

        if (mode == "TRAJECTORY") {
            const std::regex trajectory_pattern(
                "\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*"
                "([+-]?[0-9]*\\.?[0-9]+)\\s*\\)");
            std::vector<cv::Point> points;
            std::vector<std::string> depths;
            for (auto match = std::sregex_iterator(
                     response.begin(), response.end(), trajectory_pattern);
                 match != std::sregex_iterator(); ++match) {
                points.emplace_back(
                    normalizedToPixel(std::stod((*match)[1]), annotated.cols),
                    normalizedToPixel(std::stod((*match)[2]), annotated.rows));
                depths.push_back((*match)[3]);
            }
            for (size_t index = 0; index < points.size(); ++index) {
                if (index > 0) {
                    cv::line(
                        annotated, points[index - 1], points[index],
                        cv::Scalar(255, 0, 0), 3);
                }
                cv::circle(
                    annotated, points[index], 7,
                    index + 1 == points.size()
                        ? cv::Scalar(0, 0, 255)
                        : cv::Scalar(0, 255, 0),
                    -1);
                cv::putText(
                    annotated,
                    std::to_string(index + 1) + ":d=" + depths[index],
                    points[index] + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 0, 0), 2);
            }
            if (!points.empty()) {
                saveOutputImage(
                    "robobrain_trajectory.png", annotated,
                    "trajectory annotation");
            }
        }
    }



    std::string encodeImageToBase64(const cv::Mat& img) {
        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf);
        
        static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";
        
        std::string ret;
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

    std::string callVLM(const std::string& system_prompt, const std::string& user_prompt, int max_tokens, cv::Mat current_image) {
        httplib::Client cli(api_ip_, api_port_);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(request_timeout_sec_);

        json payload = {
            {"model", model_name_},
            {"temperature", temperature_},
            {"max_tokens", max_tokens}
        };

        json content_array = json::array();
        content_array.push_back({
            {"type", "text"},
            {"text", user_prompt}
        });

        if (!current_image.empty()) {
            cv::Mat resized_img;
            float max_dim = 640.0f;
            float scale = std::min(max_dim / current_image.cols, max_dim / current_image.rows);
            if (scale < 1.0f) {
                cv::resize(current_image, resized_img, cv::Size(), scale, scale, cv::INTER_AREA);
            } else {
                resized_img = current_image;
            }
            std::string base64_image = encodeImageToBase64(resized_img);
            content_array.push_back({
                {"type", "image_url"},
                {"image_url", {{"url", "data:image/jpeg;base64," + base64_image}}}
            });
        }

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
                "[ROBOBRAIN TX] http://%s:%d model=%s image=%s "
                "temperature=%.2f max_tokens=%d\nsystem: %s\nuser: %s",
                api_ip_.c_str(), api_port_, model_name_.c_str(),
                image_description.c_str(),
                temperature_, max_tokens,
                system_prompt.empty() ? "<none>" : system_prompt.c_str(),
                user_prompt.c_str());
        }

        auto res = cli.Post("/v1/chat/completions", payload.dump(), "application/json");
        
        if (!res || res->status != 200) {
            std::string err = res ? "HTTP " + std::to_string(res->status) : "Connection Failed";
            RCLCPP_ERROR(this->get_logger(), "API Error: %s", err.c_str());
            if (res && !res->body.empty()) {
                RCLCPP_ERROR(this->get_logger(), "LM Studio response: %s", res->body.c_str());
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
                RCLCPP_ERROR(this->get_logger(), "LM Studio returned non-text message content.");
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
            RCLCPP_WARN(this->get_logger(), "Error reading response: %s", e.what());
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
                "[PLACE], [PLACE_FREE], [PLAN_UPRIGHT], [PLAN], or [GRAB].";
            return;
        }

        const std::string normalized_prompt = lowercase(trim(parsed.prompt));
        if (normalized_prompt == "world state" ||
            normalized_prompt == "show world state" ||
            normalized_prompt == "get world state") {
            if (!get_world_state_client_->wait_for_service(
                    std::chrono::seconds(2))) {
                response->response =
                    "WORLD_STATE_UNAVAILABLE: /master/get_world_state";
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
            const std::string grounding_target =
                "the complete visible placement surface of the " +
                destination_name +
                ". Use one tight bounding box enclosing the entire named "
                "destination surface";
            const std::string grounding_reply = callVLM(
                "", prepareOfficialPrompt("GROUNDING", grounding_target),
                64, image);
            const std::regex box_pattern(
                "\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*"
                "([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
            std::smatch box_match;
            if (!std::regex_search(
                    grounding_reply, box_match, box_pattern)) {
                response->response =
                    "PLACEMENT_GROUNDING_PARSE_FAILED: " + grounding_reply;
                return;
            }
            int x1 = normalizedToPixel(std::stod(box_match[1]), image.cols);
            int y1 = normalizedToPixel(std::stod(box_match[2]), image.rows);
            int x2 = normalizedToPixel(std::stod(box_match[3]), image.cols);
            int y2 = normalizedToPixel(std::stod(box_match[4]), image.rows);
            if (x1 > x2) std::swap(x1, x2);
            if (y1 > y2) std::swap(y1, y2);
            vlm_node::EdgeSamResult segmentation;
            try {
                segmentation = edgesam_->segment(
                    image, cv::Rect(
                        x1, y1, std::max(1, x2 - x1 + 1),
                        std::max(1, y2 - y1 + 1)));
            } catch (const std::exception &error) {
                response->response =
                    std::string("PLACEMENT_EDGESAM_FAILED: ") + error.what();
                return;
            }
            if (segmentation.mask_pixels < edgesam_min_mask_pixels_) {
                response->response =
                    "PLACEMENT_MASK_TOO_SMALL: pixels=" +
                    std::to_string(segmentation.mask_pixels);
                return;
            }
            cv::Mat segmented = cv::Mat::zeros(image.size(), image.type());
            image.copyTo(segmented, segmentation.mask);
            saveOutputImage(
                "placement_destination_rgb.png", segmented,
                "segmented placement destination RGB image");
            saveOutputImage(
                "placement_destination_mask.png", segmentation.mask,
                "segmented placement destination mask");
            RCLCPP_INFO(
                this->get_logger(),
                "[PLACE] destination=%s mask=%d px score=%.3f policy=%s",
                destination_name.c_str(), segmentation.mask_pixels,
                segmentation.score,
                upright_policy ? "upright" : "free");

            if (!placement_candidates_client_->wait_for_service(
                    std::chrono::seconds(5))) {
                response->response =
                    "PLACEMENT_BACKEND_UNAVAILABLE: "
                    "/anyplace/find_placement_candidates";
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
                    "PLACEMENT_PLANNER_UNAVAILABLE: /vamp_planner/plan_placement";
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
            auto plan_future = plan_placement_client_->async_send_request(
                plan_request);
            if (plan_future.wait_for(std::chrono::seconds(30)) !=
                std::future_status::ready) {
                response->response = "PLACEMENT_PLANNER_TIMEOUT";
                return;
            }
            const auto plan_result = plan_future.get();
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
            const std::string world_name = implicit_object ?
                *implicit_object : targetWorldName(parsed.prompt);
            const std::string object_name =
                world_name.empty() ? parsed.prompt : world_name;
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
                        "MASTER_UNAVAILABLE: /master/execute_grasp; no motion was started";
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
            saveOutputImage("robobrain_raw.png", image, "raw synchronized grasp image");
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
                saveOutputImage(
                    "robobrain_raw.png", image,
                    "raw synchronized post-NBV grasp image");
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
                grounding_target =
                    "the complete visible rim of " + object_name;
            } else if (lower_prompt.find("body") != std::string::npos ||
                       lower_prompt.find("side") != std::string::npos) {
                grounding_target =
                    "the complete visible body of " + object_name;
            }
            grounding_target +=
                ". Use one tight bounding box enclosing the entire named "
                "target, not a smaller grasp or contact patch";
            const std::string grounding_reply = callVLM(
                "", prepareOfficialPrompt(
                    "GROUNDING", grounding_target),
                64, image);
            const std::regex box_pattern(
                "\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*"
                "([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
            std::smatch box_match;
            if (!std::regex_search(grounding_reply, box_match, box_pattern)) {
                response->response =
                    "VLM_GROUNDING_PARSE_FAILED: expected normalized [x1, y1, x2, y2], got: " +
                    grounding_reply;
                return;
            }
            int x1 = normalizedToPixel(std::stod(box_match[1]), image.cols);
            int y1 = normalizedToPixel(std::stod(box_match[2]), image.rows);
            int x2 = normalizedToPixel(std::stod(box_match[3]), image.cols);
            int y2 = normalizedToPixel(std::stod(box_match[4]), image.rows);
            if (x1 > x2) std::swap(x1, x2);
            if (y1 > y2) std::swap(y1, y2);
            // HGGD supplies the spatial grasp seeds. Keep only a mechanical
            // ROI anchor for the existing service contract; the planner uses
            // depth continuity to select foreground and never treats this
            // rectangle centre as a semantic grasp point.
            int u = (x1 + x2) / 2;
            int v = (y1 + y2) / 2;

            cv::Mat rectangular_mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
            cv::rectangle(rectangular_mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), cv::FILLED);
            cv::Mat target_mask = rectangular_mask;
            std::string target_mask_source = "RoboBrain grounding rectangle";
            if (grasp_target_mask_backend_ == "edgesam") {
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
                        edgesam_->segment(image, grounding_roi);
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
            } else if (grasp_target_mask_backend_ == "yolo_intersection" && !world_name.empty()) {
                my_actions::srv::GetYoloInference::Response yolo_result;
                std::string yolo_status;
                const cv::Rect grounding_roi(
                    x1, y1, std::max(1, x2 - x1 + 1),
                    std::max(1, y2 - y1 + 1));
                if (getYoloResultForObservation(
                        observation, yolo_result, yolo_status) &&
                    selectYoloTargetMask(
                        yolo_result, world_name, grounding_roi, u, v,
                        target_mask, target_mask_source)) {
                    cv::bitwise_and(target_mask, rectangular_mask, target_mask);
                    target_mask_source += " intersected with RoboBrain ROI";
                    RCLCPP_INFO(this->get_logger(),
                                "[GRASP] Using %s instead of the grounding rectangle.",
                                target_mask_source.c_str());
                } else {
                    RCLCPP_WARN(this->get_logger(),
                                "[GRASP] YOLO target mask unavailable (%s); using the grounding rectangle.",
                                yolo_status.c_str());
                }
            }
            cv::Mat segmented_rgb = cv::Mat::zeros(image.size(), image.type());
            image.copyTo(segmented_rgb, target_mask);
            saveOutputImage(
                "segmented_target_rgb.png", segmented_rgb,
                "segmented target RGB image");
            saveOutputImage(
                "segmented_target_mask.png", target_mask,
                "segmented target binary mask");
            cv::Mat annotated = image.clone();
            cv::Mat target_overlay(image.size(), image.type(), cv::Scalar(0, 255, 255));
            cv::Mat blended;
            cv::addWeighted(annotated, 0.55, target_overlay, 0.45, 0.0, blended);
            blended.copyTo(annotated, target_mask);
            cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);
            saveOutputImage(
                "robobrain_grasp_roi.png", annotated,
                "grounded and segmented grasp target annotation");
            RCLCPP_INFO(
                this->get_logger(),
                "[GRASP] Grounded ROI=[%d,%d]-[%d,%d] mask=%s; "
                "HGGD will provide spatial seeds.",
                x1, y1, x2, y2, target_mask_source.c_str());

            if (!grasp_candidates_client_->wait_for_service(std::chrono::seconds(3))) {
                response->response = "VAMP_UNAVAILABLE: /vamp_planner/find_grasp_candidates";
                return;
            }
            auto grasp_request = std::make_shared<my_actions::srv::FindGraspCandidates::Request>();
            grasp_request->observation_header = observation.header;
            grasp_request->seed_u = u; grasp_request->seed_v = v;
            grasp_request->image_width = image.cols; grasp_request->image_height = image.rows;
            grasp_request->target_mask = *cv_bridge::CvImage(
                observation.header, "mono8", target_mask).toImageMsg();
            grasp_request->target_description = parsed.prompt;
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
                        << candidate.jaw_opening_m << "m gpd_score=" << candidate.score
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
                        "the arm and /master/execute_grasp are checked.";
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
                    current_image = latest_image_.clone();
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
            saveOutputImage(
                "robobrain_raw.png", current_image,
                "raw RoboBrain input image");

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
                    saveGroundingStatusImage(
                        current_image, "GROUNDING ABORTED");
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
                    saveGroundingStatusImage(current_image, "NOT FOUND");
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

            const std::string reply = callVLM(
                "", prepared_prompt, max_tokens_, current_image);
            if (reply.empty()) {
                response->response = "RoboBrain failed to return a response.";
                return;
            }
            response->response = reply;
            parseAndLogOfficialResponse(parsed.mode, reply);
            annotateOfficialResponse(parsed.mode, reply, current_image);
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
            saveOutputImage(
                "robobrain_raw.png", current_image,
                "raw RoboBrain input image");

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
                    action_image = latest_image_.clone();
                }
            }
            if (action_image.empty()) {
                response->response =
                    "ACTION requires an image from /camera/color/image_raw.";
                return;
            }
            saveOutputImage(
                "robobrain_raw.png", action_image,
                "raw RoboBrain input image");
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
                                response->response = "Appended LOOK_AT task, but failed to connect to /master/execute_pipeline.";
                            }
                        } else {
                            response->response = "Failed to append task. Service timed out.";
                        }
                    } else {
                        response->response = "Failed to connect to /master/append_task service. Simulated Action: " + action_str;
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
