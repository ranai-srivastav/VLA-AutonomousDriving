#!/usr/bin/env python3
"""
CARLA scripted scenario: Overtake a stopped vehicle with oncoming traffic.

What it does:
- Loads a town (default: Town02) and switches to synchronous mode.
- Finds a straight, bidirectional two-lane road (non-junction) deterministically.
- Spawns 3 vehicles at fixed positions on that road:
  - Car1 (agent): starts behind, approaches a stopped car ahead, waits for oncoming traffic (Car3), overtakes, and returns to lane.
  - Car2 (lead): in the same lane, slows to a stop ahead of Car1.
  - Car3 (oncoming): opposite lane, drives through the scene towards Car1.
- Optionally spawns background NPC vehicles away from the scene to keep town alive without interfering.
- Runs for configurable duration then cleans up and restores settings.

Usage:
  python3 src/scenario_overtake_oncoming.py --town Town02 --duration 35 --bg-vehicles 10 --seed 42
"""

import argparse
import math
import threading
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2

try:
    import carla
except Exception as e:
    print('Failed to import carla module. Ensure CARLA Python API is available on PYTHONPATH.')
    print('Error:', e)
    sys.exit(1)


# ----------------------- Utilities -----------------------
def get_speed(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def yaw_to_rad(yaw_deg: float) -> float:
    return math.radians(yaw_deg)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def advance_waypoint(wp: carla.Waypoint, distance: float, forward: bool = True) -> carla.Waypoint:
    remaining = abs(distance)
    current = wp
    step = 2.0
    while remaining > 0.0:
        if forward:
            nxt = current.next(step)
            if not nxt:
                break
            current = nxt[0]
        else:
            prv = current.previous(step)
            if not prv:
                break
            current = prv[0]
        remaining -= step
    return current


def shift_wp_along(base: carla.Waypoint, delta_m: float) -> carla.Waypoint:
    """Shift a waypoint forward (+) or backward (-) along its lane by |delta_m| meters."""
    if delta_m >= 0:
        return advance_waypoint(base, abs(delta_m), True)
    else:
        return advance_waypoint(base, abs(delta_m), False)


def find_4lane_bidirectional_segment(world: carla.World) -> Tuple[carla.Waypoint, carla.Waypoint, carla.Waypoint]:
    """Find a 4-lane (2 per direction) segment such that:
    - car1_lane: a driving lane that is adjacent (on its left) to an opposite-direction lane (oncoming close by),
      and has a right neighbor lane in the same direction (outer lane).
    Returns (car1_lane_wp, same_dir_right_wp, opposite_adjacent_wp).
    Deterministic by scanning sorted waypoints.
    """
    amap = world.get_map()
    wps = amap.generate_waypoints(2.0)
    wps.sort(key=lambda w: (w.road_id, round(w.transform.location.x, 1), round(w.transform.location.y, 1)))
    for wp in wps:
        if wp.is_junction or wp.lane_type != carla.LaneType.Driving:
            continue
        left = wp.get_left_lane()
        right = wp.get_right_lane()
        if not right or right.lane_type != carla.LaneType.Driving:
            continue
        if not left or left.lane_type != carla.LaneType.Driving:
            continue
        # Require right lane same direction, left lane opposite direction
        if (right.lane_id * wp.lane_id > 0) and (left.lane_id * wp.lane_id < 0):
            # Also check headings are roughly opposite between wp and left
            a = wp.transform.rotation.yaw
            b = left.transform.rotation.yaw
            if abs(((a - b + 180) % 360) - 180) > 30:
                continue
            return (wp, right, left)
    # Fallback to any bidirectional pair if 4-lane not found
    amap = world.get_map()
    wps = amap.generate_waypoints(2.0)
    wps.sort(key=lambda w: (w.road_id, round(w.transform.location.x, 1), round(w.transform.location.y, 1)))
    for wp in wps:
        if wp.is_junction or wp.lane_type != carla.LaneType.Driving:
            continue
        left = wp.get_left_lane()
        right = wp.get_right_lane()
        for nb in (left, right):
            if nb and nb.lane_type == carla.LaneType.Driving and (nb.lane_id * wp.lane_id < 0):
                return (wp, right if right and right.lane_id * wp.lane_id > 0 else wp, nb)
    raise RuntimeError('Could not find a suitable 4-lane or bidirectional road segment')

def spawn_construction_block(world: carla.World, bp_lib: carla.BlueprintLibrary, lane_wp: carla.Waypoint,
                             start_m: float = 10.0, end_m: float = 35.0, step_m: float = 5.0) -> List[carla.Actor]:
    """Spawn a series of cones/barriers along the given lane to block it for construction.
    Returns list of spawned static actors.
    """
    props: List[carla.Actor] = []
    # Prefer traffic cones, fallback to any barrier
    candidates = list(bp_lib.filter('static.prop.trafficcone')) or list(bp_lib.filter('static.prop.*barrier*')) or []
    if not candidates:
        return props
    prop_bp = sorted(candidates, key=lambda b: b.id)[0]
    d = start_m
    while d <= end_m:
        wp = advance_waypoint(lane_wp, d, True)
        tr = wp.transform
        # Place slightly off-center to mimic a line of cones
        tr.location.z += 0.05
        try:
            actor = world.try_spawn_actor(prop_bp, tr)
            if actor:
                props.append(actor)
        except Exception:
            pass
        d += step_m
    return props


def safe_destroy(actors: List[carla.Actor]):
    for a in actors:
        try:
            if hasattr(a, 'is_alive') and not a.is_alive:
                continue
            a.destroy()
        except Exception:
            pass


def clear_vehicles_near(world: carla.World, center: carla.Location, radius_m: float = 60.0):
    """Remove existing vehicles near the scene to reduce spawn collisions, leaving walkers alone."""
    try:
        vehs = world.get_actors().filter('vehicle.*')
    except Exception:
        return
    to_rm = []
    for v in vehs:
        try:
            if v.get_location().distance(center) <= radius_m:
                to_rm.append(v)
        except Exception:
            pass
    safe_destroy(to_rm)


def try_spawn_vehicle(world: carla.World, bp: carla.ActorBlueprint, base_wp: carla.Waypoint, attempts: int = 16) -> Optional[carla.Vehicle]:
    """Try to spawn a vehicle around a base waypoint by nudging along-lane to avoid collisions."""
    # Offsets to try (m): center, then increasing range forward/backward
    offsets = [
        0.0,
        -3.0, 3.0,
        -6.0, 6.0,
        -9.0, 9.0,
        -12.0, 12.0,
        -15.0, 15.0,
        -18.0, 18.0,
        -21.0, 21.0,
        -24.0, 24.0,
        -27.0, 27.0,
        -30.0, 30.0,
    ]
    for i, d in enumerate(offsets[:attempts]):
        wp = shift_wp_along(base_wp, d)
        actor = world.try_spawn_actor(bp, wp.transform)
        if actor:
            return actor
    return None


# ----------------------- Sensor buffers -----------------------
class Cam4Buffers:
    def __init__(self, w: int, h: int):
        self.lock = threading.Lock()
        self.fl = np.zeros((h, w, 3), dtype=np.uint8)
        self.fr = np.zeros((h, w, 3), dtype=np.uint8)
        self.rl = np.zeros((h, w, 3), dtype=np.uint8)
        self.rr = np.zeros((h, w, 3), dtype=np.uint8)

    def set_fl(self, arr: np.ndarray):
        with self.lock:
            self.fl = arr

    def set_fr(self, arr: np.ndarray):
        with self.lock:
            self.fr = arr

    def set_rl(self, arr: np.ndarray):
        with self.lock:
            self.rl = arr

    def set_rr(self, arr: np.ndarray):
        with self.lock:
            self.rr = arr


def image_to_rgb_np(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    # BGRA -> RGB
    rgb = arr[:, :, :3][:, :, ::-1].copy()
    return rgb


# ----------------------- Bird's-eye renderer -----------------------
class BirdsEye:
    def __init__(self, width_m: float, height_m: float, ppm: float = 4.0):
        self.width_m = width_m
        self.height_m = height_m
        self.ppm = ppm
        self.w = int(width_m * ppm)
        self.h = int(height_m * ppm)

    def world_to_img(self, x: float, y: float, cx: float, cy: float) -> Tuple[int, int]:
        u = int(self.w / 2 + (x - cx) * self.ppm)
        v = int(self.h / 2 - (y - cy) * self.ppm)
        return u, v

    def draw_lane_lines(self, img: np.ndarray, lane_pts: List[carla.Transform], opp_pts: List[carla.Transform]):
        # Draw lane centerlines in white
        def to_pts(transforms: List[carla.Transform]):
            return np.array([[t.location.x, t.location.y] for t in transforms], dtype=np.float32)

        # Compute midline and normals for double yellow visualization
        pts_a = to_pts(lane_pts)
        pts_b = to_pts(opp_pts)
        if len(pts_a) < 2 or len(pts_b) < 2:
            return
        mid = (pts_a + pts_b) / 2.0
        # Approx normal from A->B direction rotated 90 degrees
        dir_vec = pts_b - pts_a
        norms = np.stack([-dir_vec[:, 1], dir_vec[:, 0]], axis=1)
        nlen = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6
        norms = norms / nlen
        # Two yellow lines offset by ~0.15m
        y1 = mid + 0.15 * norms
        y2 = mid - 0.15 * norms
        # Helper to draw polyline after projection (center provided at runtime)
        self._y1 = y1
        self._y2 = y2
        self._a = pts_a
        self._b = pts_b

    def render(self, world: carla.World, center: carla.Location, lane_pts: List[carla.Transform], opp_pts: List[carla.Transform],
               actors: List[carla.Actor]) -> np.ndarray:
        cx, cy = center.x, center.y
        canvas = np.full((self.h, self.w, 3), 30, dtype=np.uint8)  # dark gray background

        # Project helper for polylines
        def proj_poly(points_xy: np.ndarray) -> np.ndarray:
            if points_xy is None or len(points_xy) == 0:
                return np.zeros((0, 1, 2), dtype=np.int32)
            pts = []
            for x, y in points_xy:
                u, v = self.world_to_img(float(x), float(y), cx, cy)
                pts.append([u, v])
            return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        # Draw lane centerlines
        a_xy = np.array([[t.location.x, t.location.y] for t in lane_pts], dtype=np.float32)
        b_xy = np.array([[t.location.x, t.location.y] for t in opp_pts], dtype=np.float32)
        cv2.polylines(canvas, [proj_poly(a_xy)], False, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [proj_poly(b_xy)], False, (220, 220, 220), 1, cv2.LINE_AA)

        # Draw double yellow between lanes, if prepared via draw_lane_lines
        if hasattr(self, '_y1') and hasattr(self, '_y2'):
            cv2.polylines(canvas, [proj_poly(self._y1)], False, (0, 220, 220), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [proj_poly(self._y2)], False, (0, 220, 220), 2, cv2.LINE_AA)

        # Draw vehicles as oriented rectangles
        for actor in actors:
            if not isinstance(actor, carla.Vehicle):
                continue
            tf = actor.get_transform()
            bb = actor.bounding_box
            ext = bb.extent
            # Local bbox corners in vehicle frame
            corners = np.array([
                [ ext.x,  ext.y],
                [ ext.x, -ext.y],
                [-ext.x, -ext.y],
                [-ext.x,  ext.y],
            ], dtype=np.float32)
            # Rotate by yaw and translate
            yaw = math.radians(tf.rotation.yaw)
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=np.float32)
            world_xy = (corners @ rot.T) + np.array([tf.location.x, tf.location.y])
            poly = np.array([[self.world_to_img(x, y, cx, cy)] for x, y in world_xy], dtype=np.int32)
            color = (60, 180, 255) if actor.attributes.get('role_name', '') == 'hero' else (255, 180, 60)
            cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
        return canvas


# ----------------------- Controllers -----------------------
@dataclass
class PIDConfig:
    kp_steer: float = 0.04
    kp_speed: float = 0.25


class SimpleLaneFollower:
    def __init__(self, vehicle: carla.Vehicle, pid: PIDConfig):
        self.vehicle = vehicle
        self.pid = pid

    def control_towards(self, target_loc: carla.Location, target_speed: float) -> carla.VehicleControl:
        tf = self.vehicle.get_transform()
        vloc = tf.location
        # heading error
        dx = target_loc.x - vloc.x
        dy = target_loc.y - vloc.y
        desired_yaw = math.degrees(math.atan2(dy, dx))
        yaw_err = ((desired_yaw - tf.rotation.yaw + 180) % 360) - 180
        steer = clamp(self.pid.kp_steer * yaw_err, -1.0, 1.0)
        # speed control
        speed = get_speed(self.vehicle)
        sp_err = target_speed - speed
        throttle = clamp(self.pid.kp_speed * sp_err, 0.0, 1.0)
        brake = 0.0
        if sp_err < -0.5:
            throttle = 0.0
            brake = clamp(-self.pid.kp_speed * sp_err, 0.0, 1.0)
        ctrl = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        return ctrl


# ----------------------- Scenario -----------------------
def run_scenario(host: str, port: int, town: str, duration_s: float, bg_vehicles: int, seed: int,
                 cam_w: int, cam_h: int, fov: int, view: bool, bev_width_m: float, bev_height_m: float, bev_ppm: float,
                 config: Optional[Dict[str, Any]] = None) -> int:
    random.seed(seed)
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    # If config specifies town and user didn't pass one, use config
    if config and config.get('town') and not town:
        town = config['town']
    world = client.load_world(town) if town else client.get_world()

    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.0)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)

    original_settings = world.get_settings()
    settings = carla.WorldSettings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 30.0
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    actors_to_destroy: List[carla.Actor] = []
    sensors_to_destroy: List[carla.Actor] = []

    try:
        map_obj = world.get_map()
        bp_lib = world.get_blueprint_library()

        # If an external JSON config is provided, use it to spawn cars; otherwise use built-in deterministic logic
        if config and 'vehicles' in config:
            spawn_points = map_obj.get_spawn_points()

            def waypoint_from_spawn_index(idx: int) -> carla.Waypoint:
                sp = spawn_points[idx]
                return map_obj.get_waypoint(sp.location)

            def offset_wp(wp: carla.Waypoint, along: float = 0.0, right: float = 0.0, up: float = 0.0, yaw_offset: float = 0.0) -> carla.Transform:
                base_wp = shift_wp_along(wp, along)
                yaw_r = math.radians(base_wp.transform.rotation.yaw)
                right_vec = carla.Vector3D(x=math.cos(yaw_r), y=math.sin(yaw_r))
                loc = base_wp.transform.location + right_vec * right + carla.Vector3D(z=up)
                rot = base_wp.transform.rotation
                rot.yaw += yaw_offset
                return carla.Transform(loc, rot)

            vehicles_cfg = config['vehicles']
            planned: List[Tuple[Dict[str, Any], carla.ActorBlueprint, carla.Transform]] = []
            vehicle_bps = list(bp_lib.filter('vehicle.*'))
            vehicle_bps = sorted(vehicle_bps, key=lambda b: b.id)

            def resolve_bp(bpid: Optional[str]) -> carla.ActorBlueprint:
                if bpid:
                    try:
                        return bp_lib.find(bpid)
                    except Exception:
                        pass
                return vehicle_bps[0]

            for v in vehicles_cfg:
                bp = resolve_bp(v.get('blueprint'))
                if 'role_name' in v:
                    try:
                        bp.set_attribute('role_name', str(v['role_name']))
                    except Exception:
                        pass
                if 'spawn_point_index' in v:
                    idx = int(v['spawn_point_index'])
                    idx = max(0, min(idx, len(spawn_points) - 1))
                    base_wp = waypoint_from_spawn_index(idx)
                    t = offset_wp(
                        base_wp,
                        along=float(v.get('offset_along', 0.0)),
                        right=float(v.get('offset_right', 0.0)),
                        up=float(v.get('offset_up', 0.0)),
                        yaw_offset=float(v.get('yaw_offset', 0.0)),
                    )
                elif 'transform' in v:
                    tr = v['transform']
                    loc = tr.get('location', {})
                    rot = tr.get('rotation', {})
                    t = carla.Transform(
                        carla.Location(x=float(loc.get('x', 0.0)), y=float(loc.get('y', 0.0)), z=float(loc.get('z', 0.0))),
                        carla.Rotation(pitch=float(rot.get('pitch', 0.0)), yaw=float(rot.get('yaw', 0.0)), roll=float(rot.get('roll', 0.0)))
                    )
                else:
                    t = spawn_points[0]
                planned.append((v, bp, t))

            created: List[carla.Vehicle] = []
            for v, bp, t in planned:
                # Clear vicinity for each planned spawn
                try:
                    clear_vehicles_near(world, t.location, radius_m=60.0)
                except Exception:
                    pass
                base_wp = map_obj.get_waypoint(t.location)
                # Try exact transform first (slightly lifted), then fallback along-lane
                t_lift = carla.Transform(
                    carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + 0.1),
                    t.rotation
                )
                veh = world.try_spawn_actor(bp, t_lift)
                if not veh:
                    veh = try_spawn_vehicle(world, bp, base_wp, attempts=16)
                if not veh:
                    print(f"Failed to spawn {v.get('role_name','vehicle')} after retries")
                else:
                    try:
                        veh.set_transform(t)
                    except Exception:
                        pass
                    actors_to_destroy.append(veh)
                    created.append(veh)
            car1 = created[0] if len(created) > 0 else None
            car2 = created[1] if len(created) > 1 else None
            car3 = created[2] if len(created) > 2 else None
        else:
            # Original deterministic lane-based setup
            car1_lane_wp, same_dir_right_wp, opp_adjacent_wp = find_4lane_bidirectional_segment(world)
            car1_wp = advance_waypoint(car1_lane_wp, 30.0, forward=False)
            car2_wp = advance_waypoint(car1_lane_wp, 25.0, forward=True)
            car3_wp = advance_waypoint(opp_adjacent_wp, 60.0, forward=True)

            vehicle_bps = list(bp_lib.filter('vehicle.*'))
            if not vehicle_bps:
                print('No vehicle blueprints found')
                return 1
            vehicle_bps = sorted(vehicle_bps, key=lambda b: b.id)
            car1_bp = vehicle_bps[3 % len(vehicle_bps)]
            car2_bp = vehicle_bps[7 % len(vehicle_bps)]
            car3_bp = vehicle_bps[11 % len(vehicle_bps)]

            scene_center = car1_lane_wp.transform.location
            clear_vehicles_near(world, scene_center, radius_m=80.0)

            try:
                car1_bp.set_attribute('role_name', 'hero')
            except Exception:
                pass
            car1 = try_spawn_vehicle(world, car1_bp, car1_wp, attempts=10)
            if not car1:
                print('Failed to spawn Car1 after retries')
                return 1
            actors_to_destroy.append(car1)

            clear_vehicles_near(world, car2_wp.transform.location, radius_m=50.0)
            car2 = try_spawn_vehicle(world, car2_bp, car2_wp, attempts=16)
            if not car2:
                print('Failed to spawn Car2 after retries')
                car2 = None
            else:
                actors_to_destroy.append(car2)

            clear_vehicles_near(world, car3_wp.transform.location, radius_m=80.0)
            car3 = try_spawn_vehicle(world, car3_bp, car3_wp, attempts=16)
            if not car3:
                alt_car3_base = advance_waypoint(opp_adjacent_wp, 90.0, True)
                clear_vehicles_near(world, alt_car3_base.transform.location, radius_m=80.0)
                car3 = try_spawn_vehicle(world, car3_bp, alt_car3_base, attempts=16)
            if not car3:
                print('Failed to spawn Car3 after retries')
                return 1
            actors_to_destroy.append(car3)

        # If using external config but car1 failed to spawn, exit gracefully
        if (config and 'vehicles' in config) and (car1 is None):
            print('car1 not spawned; exiting')
            return 1

        # Construction/blockage on right lane for original scenario only
        if not (config and 'vehicles' in config):
            block_props = spawn_construction_block(world, bp_lib, same_dir_right_wp, start_m=5.0, end_m=35.0, step_m=5.0)
            actors_to_destroy.extend(block_props)

        # Optional: spawn background traffic far from scene
        spawn_points = world.get_map().get_spawn_points()
        if car1 is not None:
            scene_loc = car1.get_transform().location
        else:
            scene_loc = spawn_points[0].location
        far_points = [sp for sp in spawn_points if sp.location.distance(scene_loc) > 120.0]
        random.shuffle(far_points)
        npc_actors: List[carla.Actor] = []
        for sp in far_points[:max(0, bg_vehicles)]:
            bp = random.choice(vehicle_bps)
            v = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                npc_actors.append(v)
        actors_to_destroy.extend(npc_actors)

        # Attach 4 RGB cameras to car1 (only if car1 exists)
        cam_bufs = Cam4Buffers(cam_w, cam_h)
        cams = []
        if car1 is not None:
            cam_bp = bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(cam_w))
            cam_bp.set_attribute('image_size_y', str(cam_h))
            cam_bp.set_attribute('fov', str(fov))
            # FL, FR, RL, RR (corners)
            cam_tfs = [
                (carla.Transform(carla.Location(x=1.6,  y=-0.6, z=1.4), carla.Rotation(yaw=0)), 'FL'),
                (carla.Transform(carla.Location(x=1.6,  y= 0.6, z=1.4), carla.Rotation(yaw=0)), 'FR'),
                (carla.Transform(carla.Location(x=-1.6, y=-0.6, z=1.4), carla.Rotation(yaw=180)), 'RL'),
                (carla.Transform(carla.Location(x=-1.6, y= 0.6, z=1.4), carla.Rotation(yaw=180)), 'RR'),
            ]
            def mk_cb(name):
                def _cb(image: "carla.Image"):
                    rgb = image_to_rgb_np(image)
                    if name == 'FL':
                        cam_bufs.set_fl(rgb)
                    elif name == 'FR':
                        cam_bufs.set_fr(rgb)
                    elif name == 'RL':
                        cam_bufs.set_rl(rgb)
                    elif name == 'RR':
                        cam_bufs.set_rr(rgb)
                return _cb
            for tfm, name in cam_tfs:
                s = world.spawn_actor(cam_bp, tfm, attach_to=car1)
                s.listen(mk_cb(name))
                sensors_to_destroy.append(s)
                cams.append((s, name))

        # Controllers for our three cars (only used in original scenario)
        car1_controller = SimpleLaneFollower(car1, PIDConfig()) if car1 else None
        car3_controller = SimpleLaneFollower(car3, PIDConfig()) if car3 else None

        # State machine for car1
        STATE_APPROACH = 0
        STATE_WAIT = 1
        STATE_PASS = 2
        STATE_RETURN = 3
        STATE_DONE = 4

        state = STATE_APPROACH
        t_elapsed = 0.0
        dt = world.get_settings().fixed_delta_seconds or 1/30.0

        # Waypoint streams for car1 passing maneuver (original scenario only)
        path_main_forward = [advance_waypoint(car1_wp, d, True) for d in range(0, 200, 5)] if (not (config and 'vehicles' in config) and 'car1_wp' in locals()) else []
        # Create an offset path to the left (into opposing lane) for ~35m
        def offset_location(base_wp: carla.Waypoint, offset_m: float) -> carla.Location:
            tr = base_wp.transform
            yaw = yaw_to_rad(tr.rotation.yaw)
            # left vector (yaw + 90deg)
            dx = math.cos(yaw + math.pi/2)
            dy = math.sin(yaw + math.pi/2)
            return carla.Location(x=tr.location.x + offset_m * dx,
                                   y=tr.location.y + offset_m * dy,
                                   z=tr.location.z)

        # Measure approximate lane width; fallback to 3.5m
        lane_w = (car1_lane_wp.lane_width if (not (config and 'vehicles' in config) and 'car1_lane_wp' in locals() and car1_lane_wp.lane_width > 0.1) else 3.5)
        pass_offset = lane_w  # move roughly one lane left

        # Build pass segment locations around the stopped car2
        if not (config and 'vehicles' in config):
            car2_loc = car2.get_transform().location if car2 else (car1_wp.transform.location if 'car1_wp' in locals() else car1.get_transform().location)
        # pick points around 10m before and 15m after car2
        pre_pass_wp = advance_waypoint(car1_lane_wp, 10.0, True) if (not (config and 'vehicles' in config) and 'car1_lane_wp' in locals()) else None
        post_pass_wp = advance_waypoint(car1_lane_wp, 40.0, True) if (not (config and 'vehicles' in config) and 'car1_lane_wp' in locals()) else None

        # Precompute lane polylines for BEV
        lane_pts = [advance_waypoint(car1_lane_wp, d, True).transform for d in range(-60, 61, 3)] if (not (config and 'vehicles' in config) and 'car1_lane_wp' in locals()) else []
        opp_pts = [advance_waypoint(opp_adjacent_wp, d, True).transform for d in range(-60, 61, 3)] if (not (config and 'vehicles' in config) and 'opp_adjacent_wp' in locals()) else []
        bev = BirdsEye(bev_width_m, bev_height_m, bev_ppm)
        bev.draw_lane_lines(np.zeros((1,1,3),dtype=np.uint8), lane_pts, opp_pts)

        if view:
            cv2.namedWindow('Front-Left', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Front-Right', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Rear-Left', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Rear-Right', cv2.WINDOW_NORMAL)
            cv2.namedWindow("Bird's-Eye", cv2.WINDOW_NORMAL)

        # Simulation loop
        while t_elapsed < duration_s and state != STATE_DONE:
            world.tick()
            t_elapsed += dt

            if not (config and 'vehicles' in config):
                # Original scripted behaviors
                if car2:
                    if t_elapsed < 2.0:
                        car2.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))
                    else:
                        car2.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if car3_controller and 'car3_wp' in locals():
                    car3_target_wp = advance_waypoint(car3_wp, max(0.0, 80.0 - 8.0 * t_elapsed), True)
                    ctrl3 = car3_controller.control_towards(car3_target_wp.transform.location, 8.0)
                    car3.apply_control(ctrl3)

            # Center for BEV and optional behaviors
            if car1 is not None:
                car1_tf = car1.get_transform()
                car1_loc = car1_tf.location
            elif car2 is not None:
                car1_loc = car2.get_transform().location
                car1_tf = car2.get_transform()
            elif car3 is not None:
                car1_loc = car3.get_transform().location
                car1_tf = car3.get_transform()
            else:
                car1_loc = world.get_map().get_spawn_points()[0].location
                car1_tf = carla.Transform(car1_loc)

            dist_to_car2 = car1_loc.distance(car2.get_transform().location) if car2 else 999.0
            dist_to_car3 = car1_loc.distance(car3.get_transform().location) if car3 else 999.0

            if not (config and 'vehicles' in config) and state == STATE_APPROACH:
                # Approach at ~6 m/s until near car2
                target_wp = advance_waypoint(car1_wp, min(200.0, 6.0 * t_elapsed), True)
                ctrl1 = car1_controller.control_towards(target_wp.transform.location, 6.0)
                car1.apply_control(ctrl1)
                if dist_to_car2 < 18.0:
                    state = STATE_WAIT
            elif not (config and 'vehicles' in config) and state == STATE_WAIT:
                # Stop and wait for oncoming car to pass (within 25m)
                car1.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                # Once oncoming has passed (distance increasing and lateral passed), begin pass
                # Simple heuristic: if oncoming car is now behind us in heading frame
                v_to_c3 = car3.get_transform().location - car1_loc
                heading = yaw_to_rad(car1_tf.rotation.yaw)
                forward = carla.Vector3D(x=math.cos(heading), y=math.sin(heading))
                ahead_metric = v_to_c3.x * forward.x + v_to_c3.y * forward.y
                if ahead_metric < -5.0 and dist_to_car3 > 20.0:
                    state = STATE_PASS
            elif not (config and 'vehicles' in config) and state == STATE_PASS:
                # Move left into opposing lane to pass stopped car
                # Target a point offset to left near pre_pass and then near post_pass
                pre_loc = offset_location(pre_pass_wp, pass_offset)
                post_loc = offset_location(post_pass_wp, pass_offset)
                # If not yet left of lane center sufficiently, aim pre_loc; else aim post_loc
                lateral_vec = pre_loc - car1_loc
                if abs(lateral_vec.x) + abs(lateral_vec.y) > 2.0:
                    tgt = pre_loc
                else:
                    tgt = post_loc
                ctrl1 = car1_controller.control_towards(tgt, 6.0)
                car1.apply_control(ctrl1)
                # Once we have passed car2 sufficiently, go to return state
                if (not car2) or (car1_loc.distance(car2_loc) > 25.0 and car1_loc.distance(post_loc) < 8.0):
                    state = STATE_RETURN
            elif not (config and 'vehicles' in config) and state == STATE_RETURN:
                # Return to original lane center a bit ahead
                return_wp = advance_waypoint(car1_lane_wp, 55.0, True)
                ctrl1 = car1_controller.control_towards(return_wp.transform.location, 6.0)
                car1.apply_control(ctrl1)
                if car1_loc.distance(return_wp.transform.location) < 3.5:
                    state = STATE_DONE

            # Render viewer
            if view:
                with cam_bufs.lock:
                    cv2.imshow('Front-Left', cv2.cvtColor(cam_bufs.fl, cv2.COLOR_RGB2BGR))
                    cv2.imshow('Front-Right', cv2.cvtColor(cam_bufs.fr, cv2.COLOR_RGB2BGR))
                    cv2.imshow('Rear-Left', cv2.cvtColor(cam_bufs.rl, cv2.COLOR_RGB2BGR))
                    cv2.imshow('Rear-Right', cv2.cvtColor(cam_bufs.rr, cv2.COLOR_RGB2BGR))
                actors_for_bev = [a for a in [car1, car2, car3] if a is not None]
                bev_img = bev.render(world, car1_loc, lane_pts, opp_pts, actors_for_bev)
                cv2.imshow("Bird's-Eye", bev_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

        print(f"Scenario finished in state={state} at t={t_elapsed:.1f}s")
        return 0

    finally:
        try:
            # Stop traffic manager sync
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        # Cleanup
        safe_destroy(actors_to_destroy)
        try:
            for s in sensors_to_destroy:
                try:
                    s.stop()
                except Exception:
                    pass
        finally:
            safe_destroy(sensors_to_destroy)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Overtake scenario with oncoming traffic (deterministic) + BEV + 4 cams or JSON-configured spawns')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--town', default='Town03', help='CARLA town to load')
    p.add_argument('--duration', type=float, default=35.0, help='Episode duration in seconds')
    p.add_argument('--bg-vehicles', type=int, default=8, help='Background NPC vehicles (spawned far from scene)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for deterministic selection')
    p.add_argument('--cam-width', type=int, default=640)
    p.add_argument('--cam-height', type=int, default=360)
    p.add_argument('--fov', type=int, default=90)
    p.add_argument('--no-view', dest='view', action='store_false', help='Disable OpenCV viewer windows')
    p.set_defaults(view=True)
    p.add_argument('--bev-width-m', type=float, default=120.0, help='Birds-eye width (meters)')
    p.add_argument('--bev-height-m', type=float, default=80.0, help='Birds-eye height (meters)')
    p.add_argument('--bev-ppm', type=float, default=4.0, help='Birds-eye pixels-per-meter')
    # Config flag
    p.add_argument('--config', type=str, help='Path to JSON config with vehicles list (spawn_point_index+offsets or transform)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Load JSON config if provided
    cfg = None
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Failed to load config {args.config}: {e}")
            cfg = None
    if cfg is None:
        print('Error: --config is required and must be a valid JSON file with vehicles initialization.')
        return 2
    return run_scenario(args.host, args.port, args.town, args.duration, args.bg_vehicles, args.seed,
                        args.cam_width, args.cam_height, args.fov, args.view, args.bev_width_m, args.bev_height_m, args.bev_ppm,
                        config=cfg)


if __name__ == '__main__':
    sys.exit(main())
