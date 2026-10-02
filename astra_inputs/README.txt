StreamPETR sensor integration inputs

Video used for inference:
20260817_133718_300f_4096x3072.mp4

Video resolution:
4096x3072

Sensor files:
astra_inputs/Location_meters.csv
astra_inputs/Orientation.csv

Synchronization:
sensor_time = video_time + 21.999308

Derived from:
23.370930 - 1.371622 = 21.999308 seconds

Use horizontal position from Location_meters.csv.
Keep vertical position fixed at z = 0.
Use quaternion orientation from Orientation.csv.
