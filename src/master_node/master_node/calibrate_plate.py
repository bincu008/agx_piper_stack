"""Capture a known plate through the shared ROS APIs and save scale evidence."""

import argparse
import json
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from my_actions.srv import CaptureObservation, GroundObject, SegmentGroundedImage
import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation
from sensor_msgs_py import point_cloud2

from master_node.plate_scale import measure_plate
from task_pipeline.registry import endpoint


def main():
    """Calculate calibration without commanding the arm or changing live parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--diameter', type=float, default=.174)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 20:
        parser.error('samples must be between 1 and 20')
    rclpy.init()
    node = rclpy.create_node('plate_scale_calibration')
    bridge = CvBridge()
    directory = args.output.parent / (args.output.stem+'_captures')
    directory.mkdir(parents=True, exist_ok=True)

    def call(schema, address, request):
        client = node.create_client(schema, address)
        try:
            if not client.wait_for_service(timeout_sec=15):
                raise RuntimeError('Unavailable: '+address)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=130)
            if not future.done():
                raise RuntimeError('Timeout: '+address)
            response = future.result()
            if hasattr(response, 'success') and not response.success:
                raise RuntimeError(response.status)
            return response
        finally:
            node.destroy_client(client)

    try:
        params = call(GetParameters, '/active_vision_server/get_parameters',
                      GetParameters.Request(names=['perception_measurement_ratio']))
        ratio = params.values[0].double_value
        frames = []
        for index in range(args.samples):
            obs = call(CaptureObservation, endpoint('capture_observation'),
                       CaptureObservation.Request())
            ground = call(GroundObject, endpoint('ground_object'), GroundObject.Request(
                image=obs.image, target='the entire round plate, including its outer rim',
                verify_presence=False))
            segmented = call(SegmentGroundedImage, endpoint('segment_object'),
                             SegmentGroundedImage.Request(image=obs.image, bbox=ground.pixel_box))
            prefix = directory / f'frame_{index:02d}'
            prefix.with_suffix('.cdr').write_bytes(serialize_message(obs))
            cv2.imwrite(str(prefix.with_suffix('.png')), bridge.imgmsg_to_cv2(obs.image, 'bgr8'))
            mask = bridge.imgmsg_to_cv2(segmented.mask).astype(bool)
            cv2.imwrite(str(directory/f'mask_{index:02d}.png'), mask.astype(np.uint8)*255)
            world = point_cloud2.read_points_numpy(
                obs.cloud, field_names=('x', 'y', 'z'), skip_nans=False).reshape(
                    obs.cloud.height, obs.cloud.width, 3)
            transform = obs.camera_transform.transform
            rotation = Rotation.from_quat([transform.rotation.x, transform.rotation.y,
                                           transform.rotation.z, transform.rotation.w]).as_matrix()
            translation = np.array([transform.translation.x, transform.translation.y,
                                    transform.translation.z])
            camera = (world-translation) @ rotation
            measured = measure_plate(camera, mask, args.diameter, ratio)
            measured['source_stamp'] = {'sec': obs.cloud.header.stamp.sec,
                                        'nanosec': obs.cloud.header.stamp.nanosec}
            measured['grounding_response'] = ground.text
            frames.append(measured)
            np.savez_compressed(directory/f'plate_{index:02d}.npz', camera_xyz=camera, mask=mask)
            print(index+1, measured['perception_measurement_ratio'], flush=True)
        ratios = np.array([f['perception_measurement_ratio'] for f in frames])
        if np.ptp(ratios)/np.median(ratios) > .02:
            raise RuntimeError('Capture ratios vary by more than 2%; calibration not saved')
        result = {'schema_version': 1, 'perception_measurement_ratio': float(np.median(ratios)),
                  'known_diameter_m': args.diameter, 'previous_ratio': ratio,
                  'status': 'plate_in_plane_uniform_scale; absolute_depth_unverified',
                  'axis_calibration_independent': False, 'frames': frames}
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print('Saved:', args.output, flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
