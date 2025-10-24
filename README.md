# VLAs for Autonomous Driving (VLAD)

## Steps to run CARLA==0.9.16
We run CARLA in server mode. CARLA then connects to your script on port 2000 via the Python API.
To run this repository open 2 terminals and navigate to the root of this repo on your system

1. Run `./carla.sh`
2. Run `python3 src/CarlaClientTest.py`

## CARLA FAQ

1. If you get errors about `ALSA`
    - Pass the -nosound option to your command as `./CarlaUE4.sh -nosound` for CARLA<0.9.16

## Drive with WASD + Sensors

This repo also includes a manual driving client that attaches two RGB cameras (front/back) and a top-mounted LiDAR, and shows all three sensor views while you drive with WASD.

Setup (first time):

```bash
pip3 install -r requirements.txt
```

Run in two terminals from the repo root:

```bash
# Terminal 1: start CARLA 0.9.16 server
./carla.sh

# Terminal 2: run the WASD client
python3 src/manual_wasd_sensors.py --town Town02 --fps 30
```

Controls:
- W/S: throttle/brake
- A/D: steer left/right
- R: toggle reverse
- Space: handbrake
- Q or ESC: quit

Notes:
- Ensure CARLA’s Python API for your Python version is on PYTHONPATH. Launching via `./carla.sh` typically sets this up. If not, point PYTHONPATH to CARLA’s .egg/.whl for 0.9.16.
- If you prefer windowless CARLA, run the server with `-RenderOffScreen`.

## Deterministic Overtake Scenario + BEV + 4 cams

Run a scripted scenario where the agent (Car1) overtakes a stopped car (Car2) while an oncoming car (Car3) passes in the opposite lane. The script spawns the same way every time and renders four car-mounted cameras plus a bird's-eye view using OpenCV.

```bash
# Terminal 1: start CARLA server
./carla.sh

# Terminal 2: run the scenario
python3 src/scenario_overtake_oncoming.py --town Town02 --duration 35 --bg-vehicles 8 --seed 42 \
    --cam-width 640 --cam-height 360 --fov 90 --bev-width-m 120 --bev-height-m 80 --bev-ppm 4
```

Flags:
- --duration: episode seconds (default 35)
- --bg-vehicles: background NPC cars spawned far from the scene (default 8)
- --seed: deterministic seed (default 42)
- --cam-width/--cam-height/--fov: the four RGB camera resolutions and FOV
- --no-view: disable OpenCV windows (headless)
- --bev-width-m/--bev-height-m/--bev-ppm: birds-eye canvas size and pixels-per-meter

Windows:
- Front-Left, Front-Right, Rear-Left, Rear-Right (RGB views)
- Bird's-Eye (road and vehicles; double yellow center drawn between lanes)

# VLAs for Autonomous Driving (VLAD)

## Steps to run CARLA==0.9.16
We run CARLA in server mode. CARLA then connects to your script on port 2000 via the Python API.
To run this repository open 2 terminals and navigate to the root of this repo on your system

1. Run `./carla.sh`
2. Run `python3 src/CarlaClientTest.py`

## CARLA FAQ

1. If you get errors about `ALSA`
    - Pass the -nosound option to your command as `./CarlaUE4.sh -nosound` for CARLA<0.9.16

