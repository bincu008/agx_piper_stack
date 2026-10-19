#!/usr/bin/env python3
"""ROS2 bridge node: expose INSID3 grounding as typed ROS services.

This node follows the public-API conventions of the agx_piper_stack
codebase (namespaced services, typed .srv definitions, per-call timing):

    /insid3/infer      (insid3_bridge_msgs/srv/Insid3Infer)
    /insid3/visualize  (insid3_bridge_msgs/srv/Insid3Visualize)

Each infer call is routed to the INSID3 HTTP grounding server
(insid3_eval/server/insid3_server.py), which owns the template registry
and the model. This node only translates between ROS messages and the
server's HTTP API.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

import numpy as np
import rclpy
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

from insid3_bridge_msgs.srv import Insid3Infer, Insid3Visualize


def ros_image_to_pil(msg: RosImage) -> Image.Image:
    """Convert sensor_msgs/Image (rgb8/bgr8/mono8) to PIL RGB."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    if enc == 'rgb8':
        arr = arr.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
    elif enc == 'bgr8':
        arr = arr.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)[:, :, ::-1]
    elif enc == 'mono8':
        arr = arr.reshape(h, msg.step)[:, :w]
        arr = np.stack([arr] * 3, axis=-1)
    else:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    return Image.fromarray(arr.copy())


def pil_to_ros_image(img: Image.Image, encoding: str) -> RosImage:
    msg = RosImage()
    msg.height, msg.width = img.height, img.width
    msg.encoding = encoding
    msg.step = img.width * (1 if encoding == 'mono8' else 3)
    msg.data = np.asarray(img).tobytes()
    return msg


def draw_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    green = Image.new('RGB', image.size, (46, 204, 64))
    blended = Image.blend(image, green, 0.45)
    return Image.composite(blended, image,
                           Image.fromarray((mask * 255).astype(np.uint8)))


class Insid3BridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('insid3_bridge_node')
        self.declare_parameter('insid3_server_url', 'http://127.0.0.1:18092')
        self.declare_parameter('request_timeout_s', 300.0)
        self.server_url = self.get_parameter(
            'insid3_server_url').get_parameter_value().string_value
        self.timeout = self.get_parameter(
            'request_timeout_s').get_parameter_value().double_value
        self.create_service(Insid3Infer, '/insid3/infer', self._on_infer)
        self.create_service(Insid3Visualize, '/insid3/visualize',
                            self._on_visualize)
        self.get_logger().info(
            f'insid3_bridge_node ready: /insid3/infer, /insid3/visualize '
            f'-> {self.server_url}')

    # -- HTTP client ----------------------------------------------------
    def _post_server(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.server_url + '/v1/chat/completions',
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.load(resp)
        return body, time.time() - t0

    # -- /insid3/infer ---------------------------------------------------
    def _on_infer(self, request, response):
        t_call = time.time()
        try:
            image = ros_image_to_pil(request.image)
        except Exception as exc:
            response.success = False
            response.message = f'bad input image: {exc}'
            return response
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=90)
        data_url = ('data:image/jpeg;base64,' +
                    base64.b64encode(buf.getvalue()).decode())
        # The template name selects the reference pair server-side; the
        # prompt text is kept for servers that parse it (back-compat).
        prompt = ('Please provide the bounding box coordinate of the region '
                  f'this sentence describes: {request.template_name}. '
                  'Return only [x1, y1, x2, y2] in normalized 0-1000 '
                  'coordinates.')
        payload = {
            'model': 'insid3-grounding',
            'insid3_template': request.template_name,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url',
                 'image_url': {'url': data_url}},
                {'type': 'text', 'text': prompt}]}],
        }
        try:
            body, http_s = self._post_server(payload)
        except Exception as exc:
            response.success = False
            response.message = f'INSID3 server unreachable: {exc}'
            return response
        detail = body.get('insid3', {})
        mask_b64 = detail.get('mask_png_b64', '')
        if not mask_b64:
            response.success = False
            response.message = ('INSID3 server returned no mask '
                                f'(detail={detail.get("target")})')
            return response
        mask_img = Image.open(io.BytesIO(
            base64.b64decode(mask_b64))).convert('L')
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.NEAREST)
        mask = np.asarray(mask_img) > 127
        response.success = True
        response.message = (f"template={request.template_name} "
                            f"mask_px={int(mask.sum())}")
        response.mask = pil_to_ros_image(Image.fromarray(
            (mask * 255).astype(np.uint8)), 'mono8')
        ys, xs = np.nonzero(mask)
        if len(xs):
            response.bbox_xyxy = [float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())]
        response.infer_time_s = float(detail.get(
            "inference_seconds", detail.get("infer_s", 0.0)))
        total = time.time() - t_call
        self.get_logger().info(
            f'[INSID3 API] infer template={request.template_name} '
            f'infer_s={response.infer_time_s:.2f} total_s={total:.2f} '
            f'mask_px={int(mask.sum())} http_s={http_s:.2f}')
        return response

    # -- /insid3/visualize -----------------------------------------------
    def _on_visualize(self, request, response):
        try:
            image = ros_image_to_pil(request.image).convert('RGB')
            mask_img = ros_image_to_pil(request.mask).convert('L')
        except Exception as exc:
            response.success = False
            response.message = f'bad input: {exc}'
            return response
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.NEAREST)
        mask = np.asarray(mask_img) > 127
        overlay = draw_overlay(image, mask)
        response.success = True
        response.message = f'overlay drawn, mask_px={int(mask.sum())}'
        response.overlay = pil_to_ros_image(overlay, 'rgb8')
        buf = io.BytesIO()
        overlay.save(buf, format='PNG')
        response.overlay_png_b64 = base64.b64encode(buf.getvalue()).decode()
        return response


def main() -> None:
    rclpy.init()
    node = Insid3BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
