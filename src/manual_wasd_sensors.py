#!/usr/bin/env python3
"""
Drive a CARLA vehicle with WASD and visualize sensors in OpenCV windows:
- Front RGB camera (window: Front RGB)
- Rear RGB camera (window: Rear RGB)
- Top LiDAR bird's-eye projection (window: LiDAR Top-Down)

Keys (focus an OpenCV window to control):
- W/S: throttle/brake
- A/D: steer left/right
- R: toggle reverse gear
- Space: handbrake (momentary)
- Q or ESC: quit

Notes:
- Requires CARLA server running (e.g., ./CarlaUE4.sh -RenderOffScreen or ./carla.sh here)
- Tested with CARLA 0.9.15/0.9.16. Other versions may require minor adjustments.
"""

import argparse
import math
import os
import random
import sys
import threading
from typing import Optional, Tuple

try:
    import carla
except Exception as e:  # pragma: no cover - environment specific
    print("Failed to import carla module. Is the CARLA Python API installed and on PYTHONPATH?")
    print("Error:", e)
    sys.exit(1)

import numpy as np
import cv2


# ------------- Utility conversions -------------
def image_to_array(image: "carla.Image") -> np.ndarray:
    """Convert a CARLA RGBA image to an HxWx3 RGB numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # Convert BGRA to RGB
    rgb_array = array[:, :, :3][:, :, ::-1].copy()
    return rgb_array


def array_to_surface(arr: np.ndarray):
    """Deprecated: no longer used (pygame surface conversion)."""
    return arr


def lidar_to_topdown(points_xy: np.ndarray, size: Tuple[int, int], pixels_per_meter: float = 8.0,
                     offset_ahead_m: float = 10.0) -> np.ndarray:
    """
    Project lidar XY points (in vehicle frame: +X forward, +Y left) to a top-down grayscale image.

    - size: (width, height) in pixels of output image
    - pixels_per_meter: scale factor for meters->pixels
    - offset_ahead_m: moves origin forward so the vehicle sits slightly below center
    """
    w, h = size
    img = np.zeros((h, w), dtype=np.uint8)
    if points_xy.size == 0:
        return img

    # Translate so that some meters ahead maps to vertical center
    x = points_xy[:, 0] + offset_ahead_m
    y = points_xy[:, 1]

    # Convert meters to pixels (origin at image center)
    u = (w / 2) + (y * pixels_per_meter)
    v = (h / 2) - (x * pixels_per_meter)

    # Filter points inside image
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[mask].astype(np.int32)
    v = v[mask].astype(np.int32)
    img[v, u] = 255
    return img


# ------------- Sensor holders -------------
class SensorBuffers:
    def __init__(self, cam_w: int, cam_h: int, lidar_w: int, lidar_h: int):
        self.lock = threading.Lock()
        self.front_rgb: Optional[np.ndarray] = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        self.rear_rgb: Optional[np.ndarray] = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        self.lidar_topdown: Optional[np.ndarray] = np.zeros((lidar_h, lidar_w), dtype=np.uint8)

    def update_front(self, rgb: np.ndarray):
        with self.lock:
            self.front_rgb = rgb

    def update_rear(self, rgb: np.ndarray):
        with self.lock:
            self.rear_rgb = rgb

    def update_lidar(self, topdown: np.ndarray):
        with self.lock:
            self.lidar_topdown = topdown


# ------------- Main driving app -------------
class WasdDriveApp:
    def __init__(self, host: str, port: int, town: Optional[str], cam_res: Tuple[int, int], lidar_res: Tuple[int, int],
                 fov: int = 90, sync: bool = True, fixed_delta: float = 1 / 30.0):
        self.host = host
        self.port = port
        self.town = town
        self.cam_w, self.cam_h = cam_res
        self.lidar_w, self.lidar_h = lidar_res
        self.fov = fov
        self.sync = sync
        self.fixed_delta = fixed_delta

        self.client: carla.Client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        self.world: carla.World = self.client.get_world() if not town else self.client.load_world(town)
        self.original_settings: Optional[carla.WorldSettings] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.sensors: list[carla.Sensor] = []
        self.sensor_buf = SensorBuffers(self.cam_w, self.cam_h, self.lidar_w, self.lidar_h)
        self.reverse = False
        self.steer_cache = 0.0
        self.throttle_cache = 0.0
        self.brake_cache = 0.0
        self._momentary_handbrake = False

    # ---------- Setup and teardown ----------
    def setup_world(self):
        if self.sync:
            self.original_settings = self.world.get_settings()
            new_settings = carla.WorldSettings()
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = self.fixed_delta
            new_settings.no_rendering_mode = False
            self.world.apply_settings(new_settings)

    def restore_world(self):
        # Restore async settings
        if self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass

    def spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp_candidates = bp_lib.filter('vehicle.*')
        if not vehicle_bp_candidates:
            raise RuntimeError('No vehicle blueprints found')
        vehicle_bp = random.choice(vehicle_bp_candidates)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError('No spawn points found')
        spawn_pt = random.choice(spawn_points)

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_pt)
        if self.vehicle is None:
            raise RuntimeError('Failed to spawn vehicle (spawn point occupied?)')

    def attach_sensors(self):
        assert self.vehicle is not None
        bp_lib = self.world.get_blueprint_library()

        # Front RGB camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.cam_w))
        cam_bp.set_attribute('image_size_y', str(self.cam_h))
        cam_bp.set_attribute('fov', str(self.fov))
        cam_front_tf = carla.Transform(carla.Location(x=1.6, z=1.4), carla.Rotation(yaw=0))
        cam_front = self.world.spawn_actor(cam_bp, cam_front_tf, attach_to=self.vehicle)
        cam_front.listen(self._on_front_cam)
        self.sensors.append(cam_front)

        # Rear RGB camera (flip yaw by 180 degrees)
        cam_rear_tf = carla.Transform(carla.Location(x=-1.6, z=1.4), carla.Rotation(yaw=180))
        cam_rear = self.world.spawn_actor(cam_bp, cam_rear_tf, attach_to=self.vehicle)
        cam_rear.listen(self._on_rear_cam)
        self.sensors.append(cam_rear)

        # Top LiDAR (ray_cast)
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('points_per_second', '1000000')
        lidar_bp.set_attribute('rotation_frequency', '10.0')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        lidar_tf = carla.Transform(carla.Location(z=2.5))
        lidar = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        lidar.listen(self._on_lidar)
        self.sensors.append(lidar)

    def destroy_actors(self):
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self.sensors.clear()
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None

    # ---------- Sensor callbacks ----------
    def _on_front_cam(self, image: "carla.Image"):
        try:
            rgb = image_to_array(image)
            self.sensor_buf.update_front(rgb)
        except Exception:
            pass

    def _on_rear_cam(self, image: "carla.Image"):
        try:
            rgb = image_to_array(image)
            self.sensor_buf.update_rear(rgb)
        except Exception:
            pass

    def _on_lidar(self, lidar: "carla.LidarMeasurement"):
        try:
            # Parse raw data: interleaved float32 (x, y, z, intensity) per point in 0.9.16
            pts = np.frombuffer(lidar.raw_data, dtype=np.float32)
            if pts.size == 0:
                return
            pts = pts.reshape((-1, 4))
            xy = pts[:, :2]
            topdown = lidar_to_topdown(xy, (self.lidar_w, self.lidar_h))
            self.sensor_buf.update_lidar(topdown)
        except Exception:
            pass

    # ---------- OpenCV draw ----------
    def draw_cv(self):
        with self.sensor_buf.lock:
            if self.sensor_buf.front_rgb is not None:
                img_f = cv2.cvtColor(self.sensor_buf.front_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(img_f, 'Front RGB', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow('Front RGB', img_f)
            if self.sensor_buf.rear_rgb is not None:
                img_r = cv2.cvtColor(self.sensor_buf.rear_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(img_r, 'Rear RGB', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow('Rear RGB', img_r)
            if self.sensor_buf.lidar_topdown is not None:
                img_l = self.sensor_buf.lidar_topdown
                img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
                cv2.putText(img_l, f'LiDAR Top-Down (rev={"ON" if self.reverse else "OFF"})', (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
                cv2.imshow('LiDAR Top-Down', img_l)

    # ---------- Input and control ----------
    def process_key(self, key_code: int):
        """Process a single OpenCV key code; update control state caches."""
        if key_code == -1:
            # decay when no new key events come (simulates hold smoothness)
            self.throttle_cache *= 0.9
            self.brake_cache *= 0.9
            self.steer_cache *= 0.8
            return
        # normalize to ASCII lower
        try:
            key = chr(key_code & 0xFF).lower()
        except Exception:
            return
        if key in ('\x1b', 'q'):  # ESC or q
            raise KeyboardInterrupt()
        if key == 'r':
            self.reverse = not self.reverse
        if key == 'w':
            self.throttle_cache = min(1.0, self.throttle_cache + 0.2)
            self.brake_cache *= 0.5
        if key == 's':
            self.brake_cache = min(1.0, self.brake_cache + 0.3)
            self.throttle_cache *= 0.5
        if key == 'a':
            self.steer_cache = max(-1.0, self.steer_cache - 0.15)
        if key == 'd':
            self.steer_cache = min(1.0, self.steer_cache + 0.15)
        if key == ' ':
            # momentary handbrake; applied only for current frame in compute_control
            self._momentary_handbrake = True

    def compute_control(self) -> carla.VehicleControl:
        control = carla.VehicleControl()
        control.reverse = self.reverse
        control.throttle = float(max(0.0, min(1.0, self.throttle_cache)))
        control.brake = float(max(0.0, min(1.0, self.brake_cache)))
        control.hand_brake = getattr(self, '_momentary_handbrake', False)
        control.steer = float(max(-1.0, min(1.0, self.steer_cache)))
        # reset momentary flags
        self._momentary_handbrake = False
        return control

    # ---------- Main loop ----------
    def run(self):
        # Create OpenCV windows
        cv2.namedWindow('Front RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Rear RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('LiDAR Top-Down', cv2.WINDOW_NORMAL)
        try:
            self.setup_world()
            self.spawn_vehicle()
            self.attach_sensors()

            while True:
                if self.sync:
                    # Tick world first so sensors produce new frames
                    self.world.tick()

                # Render latest frames
                self.draw_cv()
                # Read key (focus any CV window for input)
                key_code = cv2.waitKey(1)
                try:
                    self.process_key(key_code)
                except KeyboardInterrupt:
                    break

                if self.vehicle is not None:
                    control = self.compute_control()
                    self.vehicle.apply_control(control)

        except KeyboardInterrupt:
            pass
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.destroy_actors()
            self.restore_world()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Drive CARLA with WASD + show sensors")
    p.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    p.add_argument('--port', type=int, default=2000, help='CARLA server port')
    p.add_argument('--town', default=None, help='Optional CARLA town to load (e.g., Town02)')
    p.add_argument('--cam-width', type=int, default=640)
    p.add_argument('--cam-height', type=int, default=360)
    p.add_argument('--lidar-width', type=int, default=1280)
    p.add_argument('--lidar-height', type=int, default=360)
    p.add_argument('--fov', type=int, default=90)
    p.add_argument('--async', dest='sync', action='store_false', help='Use asynchronous mode (default: sync on)')
    p.add_argument('--fps', type=float, default=30.0, help='Simulation FPS in sync mode (default: 30)')
    args = p.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    app = WasdDriveApp(
        host=args.host,
        port=args.port,
        town=args.town,
        cam_res=(args.cam_width, args.cam_height),
        lidar_res=(args.lidar_width, args.lidar_height),
        fov=args.fov,
        sync=args.sync,
        fixed_delta=1.0 / max(1.0, args.fps),
    )
    app.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
