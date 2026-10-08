#!/usr/bin/env python3
"""Create a synthetic Strava-streams event for testing the Lambda handler.

    python scripts/make_sample_event.py > sample_streams.json
    cd deploy && python handler.py ../sample_streams.json      # needs TensorFlow

The output has exactly the shape of
GET /activities/{id}/streams?keys=heartrate,altitude,latlng,velocity_smooth
(a 60-minute run with a hard block in the middle). The GPS track is a made-up
loop - no real person's data.
"""
import json
import math
import random

random.seed(7)
N = 3600                                   # one sample per second
hr, alt, latlng, vel = [], [], [], []
for i in range(N):
    hard = 1500 <= i < 2500
    hr.append(round(166 + random.gauss(0, 2) if hard else 132 + random.gauss(0, 2)))
    vel.append(round((3.6 if hard else 2.8) + random.gauss(0, 0.1), 2))
    ang = 2 * math.pi * i / N
    latlng.append([round(0.0 + 0.006 * math.sin(ang), 6), round(0.0 + 0.006 * math.cos(ang), 6)])
    alt.append(round(20 + 5 * math.sin(ang * 3), 1))

print(json.dumps({
    "heartrate": {"data": hr},
    "altitude": {"data": alt},
    "latlng": {"data": latlng},
    "velocity_smooth": {"data": vel},
}))
