
#!/usr/bin/env python3
"""Simple CARLA client that connects to a server on localhost:2000,
spawns a random vehicle, enables autopilot for a short time, then
cleans up the actor.
"""

import random
import time
import sys

try:
	import carla
except Exception as e:
	print('Failed to import carla module. Is the CARLA Python API installed?')
	print('Error:', e)
	sys.exit(1)


def main(host='127.0.0.1', port=2000, run_seconds=60):
	client = carla.Client(host, port)
	client.set_timeout(10.0)

	print(f'Connecting to CARLA server at {host}:{port}...')
	try:
		world = client.load_world('Town02')
	except Exception as e:
		print('Could not get world from server:', e)
		return 1

	blueprint_library = world.get_blueprint_library()
	vehicle_bp = None
	# pick a vehicle blueprint (prefer cars)
	vehicles = blueprint_library.filter('vehicle.*')
	if not vehicles:
		print('No vehicle blueprints available.')
		return 1
	vehicle_bp = random.choice(vehicles)

	spawn_points = world.get_map().get_spawn_points()
	if not spawn_points:
		print('No spawn points available in the map.')
		return 1

	spawn_point = random.choice(spawn_points)

	vehicle = None
	try:
		print('Spawning vehicle (%s) at %s' % (vehicle_bp.id, spawn_point.location))
		vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
		if vehicle is None:
			print('Failed to spawn vehicle (spawn point collision?).')
			return 1

		print(f'Created vehicle id={vehicle.id}. Enabling autopilot for {run_seconds} seconds...')
		vehicle.set_autopilot(True)

		# Let the vehicle drive for a short duration
		for i in range(run_seconds):
			print(f'  running... {i+1}/{run_seconds}')
			time.sleep(1)

		print('Done. Cleaning up...')
	except KeyboardInterrupt:
		print('\nInterrupted by user. Cleaning up...')
	except Exception as e:
		print('Exception while running client:', e)
	finally:
		# Destroy actors we created
		if vehicle is not None:
			try:
				vehicle.destroy()
				print('Destroyed vehicle')
			except Exception as e:
				print('Could not destroy vehicle:', e)

	return 0


if __name__ == '__main__':
	sys.exit(main())
