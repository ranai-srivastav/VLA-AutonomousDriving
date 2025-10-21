# VLAs for Autonomous Driving (VLAD)

## Steps to run CARLA==0.9.16
We run CARLA in server mode. CARLA then connects to your script on port 2000 via the Python API.
To run this repository open 2 terminals and navigate to the root of this repo on your system

1. Run `./carla.sh`
2. Run `python3 src/CarlaClientTest.py`

## CARLA FAQ

1. If you get errors about `ALSA`
    - Pass the -nosound option to your command as `./CarlaUE4.sh -nosound` for CARLA<0.9.16

