from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz
from ischedule import schedule, run_loop
import numpy as np
import placo
from placo_utils.tf import tf

"""
Visualizes a 6-axis robot
"""

robot = placo.RobotWrapper("assets/right_rm65f/right_rm65.urdf")


robot.set_velocity_limit("right_joint_1", 0.5)
robot.set_velocity_limit("right_joint_2", 0.6)
robot.set_velocity_limit("right_joint_3", 0.8)
robot.set_velocity_limit("right_joint_4", 1.0)
robot.set_velocity_limit("right_joint_5", 1.0)
robot.set_velocity_limit("right_joint_6", 1.5)



solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)


effector_task = solver.add_frame_task("right_ee_link", np.eye(4))
effector_task.configure("right_ee_link", "soft", 10.0, 0.2)

# Enabling self collisions avoidance
avoid_self_collisions = solver.add_avoid_self_collisions_constraint()
avoid_self_collisions.configure("avoid_self_collisions", "hard")

# The constraint starts existing when contacts are 3cm away, and keeps a 10cm margin
avoid_self_collisions.self_collisions_margin = 0.05 # [m] 
avoid_self_collisions.self_collisions_trigger = 0.15 # [m]


# Enable velocity limits
solver.enable_velocity_limits(True)

viz = robot_viz(robot)


t = 0
dt = 0.005
solver.dt = dt
last_targets = []
last_positions = []
current_position = robot.get_T_world_frame("right_ee_link")[0:3, 3]
current_qpos = robot.state.q
last_target_t = 0

print(current_position)
print(current_qpos)



@schedule(interval=dt)
def loop():
    global t, last_targets, last_target_t , last_positions
    t += dt

    # Updating the effector task (drawing an infinite sign ∞)
    target = [0.55, np.cos(t) * 0.5, 0.4 + np.sin(2 * t) * 0.25]
    # print(tf.translation_matrix(target))
    effector_task.T_world_frame = tf.translation_matrix(target)
    

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "right_ee_link")
    frame_viz("target", effector_task.T_world_frame, opacity=0.25 , scale=2.0)
    

    # Drawing the last 50 targets (adding one point every 100ms)
    if t - last_target_t > 0.1:
        last_target_t = t
        last_targets.append(target)
        last_targets = last_targets[-50:]
        points_viz("targets", np.array(last_targets), color=0xaaff00)
        
        T_world_ee = robot.get_T_world_frame("right_ee_link")
        current_position= T_world_ee[0:3, 3]
        last_positions.append(current_position)
        last_positions = last_positions[-50:]
        points_viz("positions", np.array(last_positions), color=0x5C2FC2)
        
    
        
run_loop()

