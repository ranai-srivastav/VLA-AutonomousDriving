#!/usr/bin/env python3
"""
CARLA utilities:
- dump-spawns: write all map spawn points to JSON
- print-spectator: print current spectator transform and exit

Usage examples:
  python src/carla_utils.py dump-spawns --town Town03 --out spawns_Town03.json
  python src/carla_utils.py print-spectator --town Town03
"""

import argparse
import sys

try:
    import carla
except Exception as e:
    print('Failed to import carla module. Ensure CARLA Python API is available on PYTHONPATH.')
    print('Error:', e)
    sys.exit(1)


def cmd_dump_spawns(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()

    sps = world.get_map().get_spawn_points()
    dump = []
    for idx, sp in enumerate(sps):
        t = sp
        dump.append({
            'index': idx,
            'location': {'x': t.location.x, 'y': t.location.y, 'z': t.location.z},
            'rotation': {'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll},
        })
    import json
    with open(args.out, 'w') as f:
        json.dump({'town': world.get_map().name, 'spawn_points': dump}, f, indent=2)
    print(f"Wrote spawn points to {args.out} ({len(sps)} points)")
    return 0


esspectator_help = 'Print spectator transform and exit (does not change world settings).'

def cmd_print_spectator(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    spec_tf = world.get_spectator().get_transform()
    print({
        'location': {'x': spec_tf.location.x, 'y': spec_tf.location.y, 'z': spec_tf.location.z},
        'rotation': {'pitch': spec_tf.rotation.pitch, 'yaw': spec_tf.rotation.yaw, 'roll': spec_tf.rotation.roll},
    })
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='CARLA utilities')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--town', default='Town03')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_dump = sub.add_parser('dump-spawns', help='Dump all spawn points to a JSON file')
    p_dump.add_argument('--out', required=True, help='Output JSON path')
    p_dump.set_defaults(func=cmd_dump_spawns)

    p_spec = sub.add_parser('print-spectator', help=esspectator_help)
    p_spec.set_defaults(func=cmd_print_spectator)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
