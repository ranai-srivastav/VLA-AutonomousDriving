#!/usr/bin/env python3
"""
Spawn a stationary vehicle at every Nth spawn point and label it with its index, so you can
fly the spectator to evaluate good locations.

Examples:
  python src/label_spawns.py --town Town03 --step 5
  python src/label_spawns.py --town Town03 --step 10 --bp vehicle.tesla.model3

Notes:
- Uses async mode (default world settings). Does not freeze the sim.
- Draws persistent labels above each spawned car with the spawn index.
- Cleanly destroys the spawned vehicles on Ctrl+C.
"""

import argparse
import sys
import time
import json

try:
    import carla
except Exception as e:
    print('Failed to import carla module. Ensure CARLA Python API is available on PYTHONPATH.')
    print('Error:', e)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description='Label spawn points by spawning vehicles and drawing index text')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--town', default='Town03')
    p.add_argument('--step', type=int, default=5, help='Spawn a vehicle at every Nth spawn point (e.g., 5)')
    p.add_argument('--spawns', type=str, help='Path to dump JSON (from --dump-spawns) to use spawn transforms from file')
    p.add_argument('--bp', default='vehicle.tesla.model3', help='Vehicle blueprint id to spawn')
    p.add_argument('--z-offset', type=float, default=0.1, help='Lift spawn Z slightly to avoid ground clipping')
    p.add_argument('--label-height', type=float, default=2.5, help='Label height above vehicle roof')
    p.add_argument('--label-life', type=float, default=120.0, help='Seconds labels persist (redrawn periodically)')
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()

    bp_lib = world.get_blueprint_library()
    try:
        veh_bp = bp_lib.find(args.bp)
    except Exception:
        vlist = list(bp_lib.filter('vehicle.*'))
        vlist.sort(key=lambda b: b.id)
        veh_bp = vlist[0]
        print(f"Blueprint {args.bp} not found, using {veh_bp.id}")

    # Load spawn points from JSON file if provided, else from map
    if args.spawns:
        try:
            with open(args.spawns, 'r') as f:
                data = json.load(f)
            sps = data.get('spawn_points', [])
            spawn_points = []
            for sp in sps:
                loc = sp.get('location', {})
                rot = sp.get('rotation', {})
                spawn_points.append(carla.Transform(
                    carla.Location(x=float(loc.get('x', 0.0)), y=float(loc.get('y', 0.0)), z=float(loc.get('z', 0.0))),
                    carla.Rotation(pitch=float(rot.get('pitch', 0.0)), yaw=float(rot.get('yaw', 0.0)), roll=float(rot.get('roll', 0.0)))
                ))
            print(f"Loaded {len(spawn_points)} spawn points from {args.spawns}")
        except Exception as e:
            print(f"Failed to read spawns from {args.spawns}: {e}; falling back to map spawns")
            spawn_points = world.get_map().get_spawn_points()
    else:
        spawn_points = world.get_map().get_spawn_points()
    actors_to_destroy = []
    labeled: list[tuple[carla.Vehicle, int]] = []

    # Try to spawn every Nth vehicle
    for idx in range(0, len(spawn_points), max(1, args.step)):
        sp = spawn_points[idx]
        tr = carla.Transform(sp.location + carla.Location(z=args.z_offset), sp.rotation)
        # Set role name for clarity (optional)
        try:
            veh_bp.set_attribute('role_name', f'label_{idx}')
        except Exception:
            pass
        veh = world.try_spawn_actor(veh_bp, tr)
        if not veh:
            print(f"[skip] Failed to spawn at index {idx}")
            continue
        # Stop vehicle
        try:
            veh.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass
        actors_to_destroy.append(veh)
        labeled.append((veh, idx))

        # Draw a floating label with the spawn index
        loc = veh.get_transform().location + carla.Location(z=args.label_height)
        world.debug.draw_string(loc, f"#{idx}", life_time=args.label_life, persistent_lines=True, color=carla.Color(r=255, g=100, b=50))

    print(f"Spawned {len(actors_to_destroy)} labeled vehicles. Fly around to inspect. Press Ctrl+C to clean up.")

    try:
        # Keep refreshing labels occasionally so they remain visible long enough
        refresh_dt = max(5.0, min(args.label_life * 0.5, 30.0))
        while True:
            time.sleep(refresh_dt)
            for veh, idx in labeled:
                if not veh.is_alive:
                    continue
                loc = veh.get_transform().location + carla.Location(z=args.label_height)
                world.debug.draw_string(loc, f"#{idx}", life_time=args.label_life, persistent_lines=True, color=carla.Color(r=255, g=100, b=50))
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        for a in actors_to_destroy:
            try:
                if hasattr(a, 'is_alive') and not a.is_alive:
                    continue
                a.destroy()
            except Exception:
                pass
        print('Cleaned up spawned vehicles.')


if __name__ == '__main__':
    sys.exit(main())
