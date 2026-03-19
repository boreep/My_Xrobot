#!/usr/bin/env bash

ros2 launch hand_control_cpp right_driver.launch.py  &
sleep 1

ros2 topic pub --once /right_arm/gripper_cmd my_interfaces/msg/HeaderFloat32 "{data: 0.0}"&

ros2 launch l515_pcl_processor l515_launch.py  &

wait``