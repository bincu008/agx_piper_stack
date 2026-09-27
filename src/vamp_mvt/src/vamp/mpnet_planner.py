import numpy as np
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import logging
import torch
import torch.nn as nn

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
except ImportError:
    RKNN = None
    RKNN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MAX_POINTCLOUD_SIZE = 11978
ROBOT_DOF = 8

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MAX_POINTCLOUD_SIZE*3, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 28)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class Planner(nn.Module):
    def __init__(self, input_size, output_size):
        super(Planner, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
            nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
            nn.Linear(896, 768), nn.PReLU(), nn.Dropout(),
            nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
            nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out

class HardwareBackend(ABC):
    """Hardware abstraction for neural network inference"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        pass
    
    @abstractmethod
    def predict(self, model: Any, inputs: np.ndarray) -> Tuple[np.ndarray, float]:
        pass

class CPUBackend(HardwareBackend):
    """CPU inference backend"""
    
    def load_model(self, model_path: str) -> Any:
        path = Path(model_path)
        if path.suffix == '.pkl':
            import pickle
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                # If pickle fails, try PyTorch state_dict
                import torch
                import os
                state_dict = torch.load(path, map_location='cpu')
                filename = os.path.basename(path)
                if "encoder" in filename.lower():
                    model = Encoder()
                    model.load_state_dict(state_dict)
                elif "planner" in filename.lower():
                    model = Planner(28 + ROBOT_DOF + ROBOT_DOF, ROBOT_DOF)  # latent + start + goal -> next_config
                    model.load_state_dict(state_dict)
                model.eval()
                return model
        elif path.suffix in ['.pt', '.pth']:
            import torch
            model = torch.load(path, map_location='cpu')
            model.eval()
            return model
        elif path.suffix == '.onnx':
            import onnxruntime as ort
            return ort.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def predict(self, model: Any, inputs: np.ndarray) -> Tuple[np.ndarray, float]:
        if hasattr(model, 'predict'):  # sklearn/pickle
            start_time = time.time()
            output = model.predict(inputs)
            return output, (time.time() - start_time) * 1000
        elif hasattr(model, '__call__'):  # pytorch
            import torch
            with torch.no_grad():
                tensor_input = torch.from_numpy(inputs).float()
                if len(tensor_input.shape) == 1:
                    tensor_input = tensor_input.unsqueeze(0)
                start_time = time.time()
                output = model(tensor_input)
                return output.numpy(), (time.time() - start_time) * 1000
        elif hasattr(model, 'run'):  # onnx
            input_name = model.get_inputs()[0].name
            start_time = time.time()
            outputs = model.run(None, {input_name: inputs})
            return outputs[0], (time.time() - start_time) * 1000
        else:
            raise ValueError("Unknown model type")

class OrangePiNPUBackend(HardwareBackend):
    """OrangePi NPU backend with CPU fallback"""
    
    def __init__(self):
        self.cpu_backend = CPUBackend()
        self.rknn_models = {}  # Cache for loaded RKNN models
        logger.info("OrangePi NPU backend initialized")
        
        # Check if RKNN is available
        if RKNN_AVAILABLE:
            try:
                # Test RKNN functionality
                test_rknn = RKNN(verbose=False)
                test_rknn.release()
                self.rknn_available = True
                logger.info("RKNN library detected - NPU acceleration enabled")
            except Exception as e:
                self.rknn_available = False
                logger.warning(f"RKNN library found but not functional: {e}")
        else:
            self.rknn_available = False
            logger.info("RKNN library not available - using CPU fallback")

    def load_model(self, model_path: str) -> Any:
        """Load model and convert to RKNN if needed"""
        if not self.rknn_available:
            return self.cpu_backend.load_model(model_path)
        
        try:
            path = Path(model_path)
            model_name = path.stem
            model_dir = path.parent
            
            # Define paths for converted models in same directory
            onnx_path = model_dir / f"{model_name}.onnx"
            rknn_path = model_dir / f"{model_name}.rknn"
            
            # Check if already converted and cached in memory
            if model_name in self.rknn_models:
                logger.debug(f"Using cached RKNN model: {model_name}")
                return self.rknn_models[model_name]
            
            # Check if RKNN file already exists on disk
            if rknn_path.exists():
                logger.info(f"Loading existing RKNN model: {rknn_path}")
                rknn_model = self._load_existing_rknn(str(rknn_path))
                self.rknn_models[model_name] = rknn_model
                return rknn_model
            
            # Check if ONNX file exists, if not convert from PyTorch
            if not onnx_path.exists():
                logger.info(f"Converting {model_name} from PKL to ONNX...")
                pytorch_model = self._load_pytorch_from_pkl(model_path)
                self._convert_to_onnx(pytorch_model, str(onnx_path), model_path)
            else:
                logger.info(f"Using existing ONNX model: {onnx_path}")
            
            # Convert ONNX to RKNN
            logger.info(f"Converting {model_name} from ONNX to RKNN...")
            rknn_model = self._convert_to_rknn(str(onnx_path), str(rknn_path))
            
            # Cache the model
            self.rknn_models[model_name] = rknn_model
            logger.info(f"Successfully loaded {model_name} on NPU")
            
            return rknn_model
            
        except Exception as e:
            logger.error(f"Failed to load model on NPU: {e}")
            logger.info("Falling back to CPU backend")
            return self.cpu_backend.load_model(model_path)
    
    def predict(self, model: Any, inputs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference on NPU or fall back to CPU"""
        if not self.rknn_available or not hasattr(model, 'inference'):
            return self.cpu_backend.predict(model, inputs)
        
        try:
            # Ensure input is the right shape and type
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)
            inputs = inputs.astype(np.float32)
            
            # Run NPU inference
            start_time = time.time()
            outputs = model.inference(inputs=[inputs])
            return outputs[0], (time.time() - start_time) * 1000  # RKNN returns list of outputs
            
        except Exception as e:
            logger.error(f"NPU inference failed: {e}")
            logger.info("Falling back to CPU")
            return self.cpu_backend.predict(model, inputs)

    def _load_pytorch_from_pkl(self, model_path: str):
        """Load PyTorch model from .pkl file"""
        path = Path(model_path)
        
        # Try loading the pickle file
        try:
            import pickle
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
        except:
            # If pickle fails, try PyTorch load
            state_dict = torch.load(path, map_location='cpu')
        
        # Determine model type and create appropriate model
        filename = path.name.lower()
        if "encoder" in filename:
            model = Encoder()
            model.load_state_dict(state_dict)
        elif "planner" in filename:
            model = Planner(28 + ROBOT_DOF + ROBOT_DOF, ROBOT_DOF)  # latent + start + goal -> next_config
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unknown model type: {filename}")
        
        model.eval()
        return model
   
    def _convert_to_onnx(self, pytorch_model, onnx_path: str, original_path: str) -> None:
        """Convert PyTorch model to ONNX"""
        model_name = Path(original_path).stem
        
        # Determine input shape based on model type
        if "encoder" in model_name.lower():
            # Encoder: point cloud input
            dummy_input = torch.randn(1, MAX_POINTCLOUD_SIZE * 3)
            input_names = ['point_cloud']
            output_names = ['env_encoding']
        else:
            # Planner: latent + start + goal
            dummy_input = torch.randn(1, 28 + ROBOT_DOF + ROBOT_DOF)
            input_names = ['planning_features']
            output_names = ['waypoint']
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names
        )
        
        logger.debug(f"Converted {model_name} to ONNX: {onnx_path}")

    def _convert_to_rknn(self, onnx_path: str, rknn_path: str):
        """Convert ONNX model to RKNN"""
        # Create RKNN instance
        if not RKNN_AVAILABLE:
            raise RuntimeError("RKNN not available for conversion")
        
        # Create RKNN instance
        rknn = RKNN(verbose=False)
        
        # Configure for RK3588 (OrangePi 5/5B)
        ret = rknn.config(
            target_platform='rk3588',
            quantized_dtype='w8a8',  # 8-bit weights, 8-bit activations
            optimization_level=1,
            quantized_algorithm='normal'
        )
        if ret != 0:
            raise RuntimeError(f"RKNN config failed with code {ret}")
        
        # Load ONNX model
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
        
        # Build the model (quantization happens here)
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed with code {ret}")
        
        # Export RKNN model
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"RKNN export failed with code {ret}")
        
        # Initialize runtime for inference
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with code {ret}")
        
        logger.debug(f"Converted and saved RKNN model: {rknn_path}")
        return rknn

    def _load_existing_rknn(self, rknn_path: str):
        """Load existing RKNN model from disk"""
        if not RKNN_AVAILABLE:
            raise RuntimeError("RKNN not available for loading")
            
        rknn = RKNN(verbose=False)
        
        # Load the RKNN model
        ret = rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {rknn_path}, code {ret}")
        
        # Initialize runtime for inference
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with code {ret}")
        
        logger.debug(f"Loaded existing RKNN model: {rknn_path}")
        return rknn

    def __del__(self):
        """Cleanup RKNN models"""
        # Release RKNN models
        for model_name, rknn_model in self.rknn_models.items():
            try:
                if hasattr(rknn_model, 'release'):
                    rknn_model.release()
                    logger.debug(f"Released RKNN model: {model_name}")
            except:
                pass

class RaspberryPiNPUBackend(HardwareBackend):
    """Raspberry Pi NPU backend with CPU fallback"""
    
    def __init__(self):
        self.cpu_backend = CPUBackend()
        logger.info("Raspberry Pi NPU: Using CPU fallback (NPU implementation pending)")
    
    def load_model(self, model_path: str) -> Any:
        # TODO: Replace with Raspberry Pi NPU model loading
        return self.cpu_backend.load_model(model_path)
    
    def predict(self, model: Any, inputs: np.ndarray) -> Tuple[np.ndarray, float]:
        # TODO: Replace with Raspberry Pi NPU inference
        return self.cpu_backend.predict(model, inputs), 0.0

class MPNetPlanner:
    """Compact MPNet planner using Python orchestration"""
    
    BACKENDS = {
        'cpu': CPUBackend,
        'opi_npu': OrangePiNPUBackend,
        'rpi_npu': RaspberryPiNPUBackend,
    }

    def __init__(self, encoder_path: str, planner_path: str, hardware: str = 'cpu'):
        """Initialize MPNet with specified hardware backend"""
        if hardware not in self.BACKENDS:
            logger.warning(f"Unknown hardware '{hardware}', using CPU")
            hardware = 'cpu'
        
        self.backend = self.BACKENDS[hardware]()
        self.hardware = hardware
        
        self.accum_val_time = 0
        self.pnet_inference_cnt = 0
        self.accum_enet_infer_time = 0
        self.accum_pnet_infer_time = 0

        # Load models
        logger.info(f"Loading encoder: {encoder_path}")
        self.encoder = self.backend.load_model(encoder_path)
        logger.info(f"Loading planner: {planner_path}")
        self.planner = self.backend.load_model(planner_path)
        
        # Cache for encoded environment
        self.latent_code = None
        logger.info(f"MPNet initialized with {hardware} backend")
    
    def encode_environment(self, pointcloud: list) -> bool:
        """Encode point cloud to latent representation"""
        try:
            # Preprocess if needed (normalize, filter, etc.)
            logger.debug(f"Point cloud shape before preprocessing: {np.array(pointcloud).shape}")
            processed_cloud = self._preprocess_pointcloud(pointcloud)
            logger.debug(f"Point cloud shape after preprocessing: {np.array(processed_cloud).shape}")
            
            # Encode
            np.savetxt('enet_input.txt', processed_cloud, delimiter=',')
            self.latent_code, self.accum_enet_infer_time = self.backend.predict(self.encoder, processed_cloud)
            logger.debug(f"Encoded {len(pointcloud)} points to latent shape: {self.latent_code.shape}")
            return True
        except Exception as e:
            logger.error(f"Environment encoding failed: {e}")
            return False
    
    def plan(self, start: np.ndarray, goal: np.ndarray, robot_module, environment,
             max_iterations: int = 50, max_planning_steps: int = 50) -> Optional[List[np.ndarray]]:
        """
        Plan path from start to goal using MPNet
        
        Args:
            start: Start configuration
            goal: Goal configuration  
            robot_module: VAMP robot module (e.g., vamp.panda)
            environment: VAMP environment
            max_iterations: Max planning iterations
            max_planning_steps: Max steps per planning attempt
            
        Returns:
            List of configurations forming path, or None if failed
        """
        if self.latent_code is None:
            logger.error("Environment not encoded. Call encode_environment() first.")
            return None
        
        logger.info(f"Start shape: {len(start)}, Goal shape: {len(goal)}")
        
        start_time = time.time()
        
        # Check direct connection first
        val_start_time = time.time()
        if robot_module.validate_motion(robot_module.Configuration(start), robot_module.Configuration(goal), environment):
            self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
            logger.info("Direct connection found")
            return [start, goal]
        self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000

        best_path = None
        best_distance_to_goal = float('inf')
        
        for iteration in range(max_iterations):
            # Try bidirectional planning (core MPNet algorithm)
            path = self._bidirectional_planning_attempt(
                start, goal, robot_module, environment, max_planning_steps
            )
            # bookshelf_thin, fetch, problem index 3
            # path = [[0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
            #         [0.1130716415169656, 0.6701656662448738, 0.5437118304124002, -0.9573185442113827, 1.192729119155159, -0.6570224510957486, 1.389650191854703, 0.5608505081960977],
            #         [0.09654528291702527, 0.5147561598239914, 0.4493044145424446, -0.4633838174423112, 0.1591385738381093, -0.8442762895832869, 0.5673636381620717, -0.6596020426390228],
            #         [0.08840026260867515, -0.1049776956095018, 0.2490675657194144, -0.255429554351454, -1.351346779291686, -1.259964889635087, -0.2211360368887943, -2.91253456987987],
            #         [0.06580220830306079, -1.002794560663853, -0.272744465625363, -1.431699601650493, -1.561234119725792, -1.744681742687049, -0.04365573011807782, 1.117031525941965],
            #         [0.2162197405043042, -0.2457764831227006, 0.02184441132129261, -2.045275807819388, -0.009591529666930394, 0.3332992301497282, 0.1652488400061966, 1.713909096307712]]
            if path and len(path) > 1:
                # Check if we reached goal
                final_config = path[-1]
                logger.debug(f"Check distance between last config {final_config} and goal {goal}")
                distance_to_goal = np.linalg.norm(final_config - goal)
                if distance_to_goal < 1.0:
                    path.append(goal)
                    print(f"Path found in iteration {iteration + 1}: {len(path)} waypoints")
                    print(f"Planning time: {(time.time() - start_time) * 1000:.1f}ms")
                    print(f"Planning Network Inference Count: {self.pnet_inference_cnt}")
                    print(f"Accum. motion validation time: {self.accum_val_time}")
                    print(f"Accum. ENet inference time: {self.accum_enet_infer_time}")
                    print(f"Accum. PNet inference time: {self.accum_pnet_infer_time}")
                    return path
                
                # Track best partial path
                if distance_to_goal < best_distance_to_goal:
                    best_distance_to_goal = distance_to_goal
                    best_path = path.copy()
        
        if best_path:
            logger.warning(f"Partial path found (distance to goal: {best_distance_to_goal:.3f})")
            return best_path
        
        logger.info("No path found")
        return None
    
    def _bidirectional_planning_attempt(self, start: np.ndarray, goal: np.ndarray,
                                      robot_module, environment, max_steps: int) -> Optional[List[np.ndarray]]:
        """MPNet bidirectional planning with divide-and-conquer strategy"""
        
        # Forward planning: start -> intermediate
        forward_path = self._single_planning_attempt(start, goal, robot_module, environment, max_steps // 2)
        if not forward_path or len(forward_path) < 2:
            return None
        
        # Backward planning: goal -> intermediate  
        backward_path = self._single_planning_attempt(goal, start, robot_module, environment, max_steps // 2)
        if not backward_path or len(backward_path) < 2:
            return forward_path
        
        # Try to connect forward and backward paths
        forward_end = forward_path[-1]
        backward_end = backward_path[-1]
        
        # Check if paths can be connected
        val_start_time = time.time()
        if robot_module.validate_motion(robot_module.Configuration(forward_end), robot_module.Configuration(backward_end), environment):
            self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
            # Merge paths (reverse backward path since it was planned goal->start)
            merged_path = forward_path + list(reversed(backward_path[:-1]))
            return merged_path
        self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
        
        # If direct connection fails, try recursive planning between end points
        bridge_path = self._single_planning_attempt(forward_end, backward_end, robot_module, environment, max_steps // 4)
        if bridge_path and len(bridge_path) > 1:
            merged_path = forward_path + bridge_path[1:] + list(reversed(backward_path[:-1]))
            return merged_path
        
        # Return the longer path if connection fails
        return forward_path if len(forward_path) >= len(backward_path) else backward_path
    
    def _single_planning_attempt(self, start: np.ndarray, goal: np.ndarray,
                               robot_module, environment, max_steps: int) -> Optional[List[np.ndarray]]:
        """Single direction MPNet planning attempt"""
        current = start.copy()
        path = [current.copy()]
        
        for step in range(max_steps):
            # Direct neural prediction from current to goal
            next_config = self._predict_next_config(current, goal)
            
            if next_config is None:
                break
            
            # Validate motion
            val_start_time = time.time()
            if robot_module.validate_motion(robot_module.Configuration(current), robot_module.Configuration(next_config), environment):
                self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
                path.append(next_config.copy())
                current = next_config
                
                # Check if reached goal
                if np.linalg.norm(current - goal) < 1:  # Goal tolerance
                    return path
            else:
                # Collision detected - try to recover with slight perturbation
                self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
                perturbed_config = self._add_noise_to_config(next_config, noise_scale=0.25)
                val_start_time = time.time()
                if robot_module.validate_motion(robot_module.Configuration(current), robot_module.Configuration(perturbed_config), environment):
                    self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
                    logger.debug(f"Recovered with perturbed config: {perturbed_config}")
                    path.append(perturbed_config.copy())
                    current = perturbed_config
                else:
                    # Failed to recover, stop this planning attempt
                    self.accum_val_time = self.accum_val_time + (time.time() - val_start_time) * 1000
                    break
        
        return path if len(path) > 1 else None
    
    def _predict_next_config(self, current: np.ndarray, goal: np.ndarray) -> Optional[np.ndarray]:
        """Predict next configuration using planning network"""
        try:
            # Prepare input: [latent_code, current, goal]
            planning_input = self._prepare_planning_input(current, goal)
            
            # Predict
            prediction, infer_time = self.backend.predict(self.planner, planning_input)
            self.accum_pnet_infer_time = self.accum_pnet_infer_time + infer_time
            self.pnet_inference_cnt = self.pnet_inference_cnt + 1
            
            # Post-process prediction
            return self._postprocess_prediction(prediction, current)
        except Exception as e:
            logger.debug(f"Prediction failed: {e}")
            return None
    
    def _preprocess_pointcloud(self, pointcloud: list) -> np.ndarray:
        """Preprocess pointcloud for encoder"""
        pointcloud = np.array(pointcloud)
        
        # Handle different input formats
        if pointcloud.ndim == 1:
            # Already flattened
            if len(pointcloud) == MAX_POINTCLOUD_SIZE * 3:
                return pointcloud.astype(np.float32)
            else:
                # Reshape and process
                pointcloud = pointcloud.reshape(-1, 3)
        
        # Subsample if too many points
        if len(pointcloud) > MAX_POINTCLOUD_SIZE:
            indices = np.random.choice(len(pointcloud), MAX_POINTCLOUD_SIZE, replace=False)
            pointcloud = pointcloud[indices]
        
        # Pad if too few points
        elif len(pointcloud) < MAX_POINTCLOUD_SIZE:
            padding = np.zeros((MAX_POINTCLOUD_SIZE - len(pointcloud), 3))
            pointcloud = np.vstack([pointcloud, padding])

        return pointcloud.astype(np.float32).flatten()
    
    def _prepare_planning_input(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Prepare input for planning network"""
        # Flatten latent code if needed
        latent_flat = self.latent_code.flatten() if self.latent_code.ndim > 1 else self.latent_code
        
        # Concatenate: [latent_code, current_config, goal_config]
        planning_input = np.concatenate([latent_flat, current, goal])
        
        # Add batch dimension if needed
        if planning_input.ndim == 1:
            planning_input = planning_input.reshape(1, -1)
        
        return planning_input.astype(np.float32)
    
    def _postprocess_prediction(self, prediction: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Post-process network prediction"""
        # Remove batch dimension if present
        if prediction.ndim > 1:
            prediction = prediction.flatten()
        
        # Apply movement constraints (limit step size for stability)
        max_step_size = 0.3  # radians
        step = prediction - current
        step_norm = np.linalg.norm(step)
        
        if step_norm > max_step_size:
            step = step * (max_step_size / step_norm)
            prediction = current + step
        
        return prediction
    
    def _add_noise_to_config(self, config: np.ndarray, noise_scale: float = 0.05) -> np.ndarray:
        """Add small random noise to configuration for collision recovery"""
        noise = np.random.normal(0, noise_scale, config.shape)
        noisy_config = config + noise
        return np.clip(noisy_config, -np.pi, np.pi)

def plan_with_mpnet(robot_name: str, start: List[float], goal: List[float],
                   environment, pointcloud, encoder_path: str, planner_path: str,
                   hardware: str = 'cpu') -> Optional[List[np.ndarray]]:
    """
    MPNet planning
    
    Args:
        robot_name: Robot name ('panda', 'ur5', 'fetch', etc.)
        start: Start configuration as list
        goal: Goal configuration as list
        environment: VAMP environment
        pointcloud: Point cloud as ndarray
        encoder_path: Path to encoder model
        planner_path: Path to planner model
        hardware: Hardware backend ('cpu', 'opi_npu', 'rpi_npu')
        
    Returns:
        Path as list of configurations, or None if planning failed
    """
    import vamp
    
    # Get robot module
    robot_module = getattr(vamp, robot_name)
    
    # Initialize MPNet
    mpnet = MPNetPlanner(encoder_path, planner_path, hardware)
    
    # Extract and encode environment
    if not mpnet.encode_environment(pointcloud):
        logger.error("Failed to encode environment")
        return None
    
    # Convert to numpy arrays
    start_np = np.array(start, dtype=np.float32)
    goal_np = np.array(goal, dtype=np.float32)
    
    # Plan
    path = mpnet.plan(start_np, goal_np, robot_module, environment)
    
    return path