#!/usr/bin/env python3
"""
Spawn 3 stationary vehicles at specific spawn points in Town03 and keep running until Ctrl+C:
- Car1 at spawn index 55
- Car2 at 10 meters ahead of index 55 (along the lane)
- Car3 at spawn index 200

Labels with the indices are drawn above each vehicle so you can verify positions in the map.

Usage:
  python src/test_spawn_three.py
Optional args:
  --host 127.0.0.1 --port 2000 --town Town03 --idx1 55 --idx3 200 --ahead-m 10.0 --bp1 vehicle.tesla.model3 --bp2 vehicle.lincoln.mkz_2017 --bp3 vehicle.audi.tt
"""

import argparse
import sys
import time
import math

try:
    import carla
except Exception as e:
    print('Failed to import carla module. Ensure CARLA Python API is available on PYTHONPATH.')
    print('Error:', e)
    sys.exit(1)


def advance_waypoint(wp: 'carla.Waypoint', distance: float) -> 'carla.Waypoint':
    """Advance a waypoint forward along its lane by distance meters."""
    remaining = max(0.0, float(distance))
    current = wp
    step = 2.0
    while remaining > 0.0:
        nexts = current.next(min(step, remaining))
        if not nexts:
            break
        current = nexts[0]
        remaining -= step
    return current


def try_spawn_exact_then_nudge(world: 'carla.World', bp: 'carla.ActorBlueprint', base_tf: 'carla.Transform') -> 'carla.Vehicle | None':
    """Try to spawn at the exact transform (slightly lifted), then nudge forward/backward along lane a bit."""
    # Slight lift to avoid ground clipping
    tf = carla.Transform(
        carla.Location(x=base_tf.location.x, y=base_tf.location.y, z=base_tf.location.z + 0.1),
        base_tf.rotation,
    )
    veh = world.try_spawn_actor(bp, tf)
    if veh:
        return veh
    # Nudge +/- along lane by a few meters
    amap = world.get_map()
    base_wp = amap.get_waypoint(base_tf.location)
    for d in (2.0, -2.0, 4.0, -4.0, 6.0, -6.0):
        wp = advance_waypoint(base_wp, abs(d)) if d >= 0 else advance_waypoint(base_wp, 0.0)  # use previous by negative isn't available; fallback
        # For negative offsets, go to previous
        if d < 0:
            prevs = base_wp.previous(abs(d))
            if prevs:
                wp = prevs[0]
        t2 = wp.transform
        t2.location.z += 0.1
        veh = world.try_spawn_actor(bp, t2)
        if veh:
            return veh
    return None


def main():
    ap = argparse.ArgumentParser(description='Spawn 3 vehicles at indices 55, 200, and 10m ahead of 55 (Town03).')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--idx1', type=int, default=55, help='Spawn index for car1')
    ap.add_argument('--idx3', type=int, default=200, help='Spawn index for car3')
    ap.add_argument('--ahead-m', type=float, default=10.0, help='Meters ahead of idx1 for car2')
    ap.add_argument('--bp1', default='vehicle.tesla.model3')
    ap.add_argument('--bp2', default='vehicle.lincoln.mkz_2017')
    ap.add_argument('--bp3', default='vehicle.audi.tt')
    ap.add_argument('--label-height', type=float, default=2.5)
    ap.add_argument('--label-life', type=float, default=60.0)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()

    bp_lib = world.get_blueprint_library()
    def get_bp(bpid: str) -> 'carla.ActorBlueprint':
        try:
            return bp_lib.find(bpid)
        except Exception:
            vlist = list(bp_lib.filter('vehicle.*'))
            vlist.sort(key=lambda b: b.id)
            return vlist[0]

    bp1 = get_bp(args.bp1)
    bp2 = get_bp(args.bp2)
    bp3 = get_bp(args.bp3)

    sps = world.get_map().get_spawn_points()
    if not sps:
        print('No spawn points available in the map.')
        return 1

    def safe_index(idx: int) -> int:
        return max(0, min(idx, len(sps) - 1))

    idx1 = safe_index(args.idx1)
    idx3 = safe_index(args.idx3)

    # Base transforms
    tf1 = sps[idx1]
    # car2: 10 m ahead along lane from tf1
    wp1 = world.get_map().get_waypoint(tf1.location)
    wp2 = advance_waypoint(wp1, float(args.ahead_m))
    tf2 = wp2.transform
    tf3 = sps[idx3]

    spawned = []
    labels = []

    # Spawn car1
    car1 = try_spawn_exact_then_nudge(world, bp1, tf1)
    if car1:
        try:
            car1.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass
        spawned.append(car1)
        labels.append((car1, f"# {idx1}"))
    else:
        print(f"Failed to spawn car1 at index {idx1}")

    # Spawn car2
    car2 = try_spawn_exact_then_nudge(world, bp2, tf2)
    if car2:
        try:
            car2.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass
        spawned.append(car2)
        labels.append((car2, f"# {idx1}+{args.ahead_m}m"))
    else:
        print(f"Failed to spawn car2 ~{args.ahead_m}m ahead of index {idx1}")

    # Spawn car3
    car3 = try_spawn_exact_then_nudge(world, bp3, tf3)
    if car3:
        try:
            car3.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass
        spawned.append(car3)
        labels.append((car3, f"# {idx3}"))
    else:
        print(f"Failed to spawn car3 at index {idx3}")

    if not spawned:
        print('No vehicles were spawned; exiting.')
        return 1

    print(f"Spawned {len(spawned)} vehicles. Press Ctrl+C to clean up.")

    try:
        # Periodically refresh labels so they stay visible while you fly
        last = 0.0
        while True:
            now = time.time()
            if now - last > max(5.0, min(args.label_life * 0.5, 30.0)):
                last = now
                for veh, text in labels:
                    if not veh.is_alive:
                        continue
                    loc = veh.get_transform().location + carla.Location(z=args.label_height)
                    world.debug.draw_string(loc, text, life_time=args.label_life, persistent_lines=True, color=carla.Color(r=255, g=150, b=50))
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for a in spawned:
            try:
                if hasattr(a, 'is_alive') and not a.is_alive:
                    continue
                a.destroy()
            except Exception:
                pass
        print('Cleaned up spawned vehicles.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
