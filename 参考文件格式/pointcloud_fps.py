#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from numba import jit

# Numba加速的FPS采样
@jit(nopython=True)
def farthest_point_sample(xyz, n_samples):
    N = xyz.shape[0]
    centroids = np.zeros(n_samples, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid)**2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return centroids

class PointCloudFilter:
    def __init__(self):
        rospy.init_node('pointcloud_filter_node', anonymous=True)
        self.sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.callback)
        self.pub_sampled = rospy.Publisher('/sampled_points', PointCloud2, queue_size=1)
        self.work_space = np.array([
            [-0.15, 0.35],
            [-0.25, 0.25],
            [0, 0.95]
        ])

    def callback(self, msg):
        try:
            # 高效读取点云数据
            points = np.array([
                p for p in pc2.read_points(
                    msg, 
                    field_names=("x","y","z","rgb"),
                    skip_nans=True
                )
            ], dtype=np.float32)
            
            # 空间剪裁
            mask = (
                (points[:,0] > self.work_space[0][0]) & 
                (points[:,0] < self.work_space[0][1]) &
                (points[:,1] > self.work_space[1][0]) & 
                (points[:,1] < self.work_space[1][1]) &
                (points[:,2] > self.work_space[2][0]) & 
                (points[:,2] < self.work_space[2][1])
            )
            filtered = points[mask]
            
            # 预处理降采样
            if len(filtered) > 2048:
                filtered = filtered[np.random.choice(len(filtered), 2048, replace=False)]
            
            # 采样逻辑
            if len(filtered) >= 1024:
                idx = farthest_point_sample(filtered[:,:3], 1024)
                sampled = filtered[idx]
            else:
                sampled = np.tile(filtered, (1024//len(filtered)+1,1))[:1024]
            
            # 发布点云
            fields = [
		PointField('x', 0, PointField.FLOAT32, 1),
		PointField('y', 4, PointField.FLOAT32, 1),
		PointField('z', 8, PointField.FLOAT32, 1),
		PointField('rgb', 12, PointField.FLOAT32, 1)
            ]
            self.pub_sampled.publish(pc2.create_cloud(msg.header, fields, sampled))
            
        except Exception as e:
            rospy.logerr(f"处理异常: {str(e)}")

if __name__ == '__main__':
    PointCloudFilter()
    rospy.spin()
