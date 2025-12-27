# XRobot Teleoperation Toolkit

XRobot Teleoperation Toolkit 是一个用于机器人遥操作控制的Python库，支持多种机器人硬件平台，提供基于逆运动学(IK)的遥操作控制功能。该项目支持VR/AR控制器输入，可以实现直观的机器人控制。

## 功能特性

- **多机器人支持**: 支持RM65双臂机器人、ARX R5、Galaxea R1 Lite等多种机器人平台
- **VR/AR控制**: 支持使用VR/AR控制器进行直观的机器人遥操作
- **逆运动学求解**: 基于Placo库的实时IK求解，支持位置和姿态控制
- **双臂协调**: 支持双臂协同操作，适用于复杂任务
- **自碰撞避免**: 内置自碰撞避免约束，确保机器人安全运动
- **数据记录**: 提供数据记录功能，便于后续分析和回放
- **可视化**: 集成MeshCat可视化，实时显示机器人状态

## 目录结构

## my_utils模块详解

`my_utils`模块包含了项目特定的机器人控制实现，主要包括以下组件：

### 1. ros2_rm65.py

RM65机械臂的ROS2接口实现，提供以下功能：

- **RM65Controller类**: 
  - 支持左右臂控制 (`right_arm` / `left_arm`)
  - 通过ROS2消息发布关节位置指令 (`Jointpos`, `Movej`)
  - 订阅关节状态反馈 (`JointState`)
  - 控制灵巧手 (`HeaderFloat32`)
  - 提供初始化位置设置
  - 支持跟随模式 (follow mode)

- **预定义常量**:
  - `LEFT_INITIAL_JOINT_DEG`: 左臂初始关节角度
  - `RIGHT_INITIAL_JOINT_DEG`: 右臂初始关节角度
  - `ARM_VELOCITY_LIMITS`: 关节速度限制

### 2. gripper2hand.py

ROHand灵巧手控制器，提供以下功能：

- **Gripper2HandController类**:
  - 通过ROS2接口控制灵巧手
  - 支持角度控制模式 (`Handangle`)
  - 发布速度和力矩指令 (`Handspeed`, `Handforce`)
  - 订阅手部状态反馈 (`Handstatus`)
  - 提供`gripper2handangle`方法将0-1范围的触发器值映射到手部角度

- **安全机制**:
  - NaN数据检查
  - 角度限制验证
  - 初始化参数设置（力矩、速度）

### 3. base_robot_teleop_controller.py

硬件机器人控制器的抽象基类，封装了通用功能：

- **多线程架构**:
  - IK线程: 运行逆运动学求解
  - 控制线程: 发送指令到硬件
  - 数据记录线程: 管理数据记录

- **抽象方法**:
  - `_robot_setup()`: 硬件接口初始化
  - `_update_robot_state()`: 更新机器人状态
  - `_send_command()`: 发送控制指令
  - `_get_robot_state_for_logging()`: 获取日志数据
  - `_shutdown_robot()`: 机器人关闭

### 4. allrobot_teleop_controller.py

全机器人控制器，整合双臂和灵巧手控制：

- **双臂支持**:
  - 同时控制左右臂
  - 为每臂创建独立的RM65Controller实例
  - 使用MultiThreadedExecutor管理ROS2通信

- **线程管理**:
  - 独立的ROS2通信线程
  - IK求解线程
  - 硬件控制线程
  - 数据记录线程

- **关节切片管理**:
  - 为左右臂分别管理关节切片
  - 同步Placo机器人模型与实际硬件状态

- **等待硬件连接**:
  - `wait_for_hardware()`方法等待机械臂数据连接

## 依赖

- Python 3.8+
- ROS2 (用于硬件接口)
- Placo (逆运动学求解)
- NumPy
- MeshCat (可视化)
- Tyro (命令行参数解析)
- rclpy (ROS2 Python客户端库)

## 使用方法

### 仿真测试

运行仿真环境下的遥操作控制:

```bash
cd test
python test_teleop.py
```


## 配置参数

- `robot_urdf_path`: 机器人URDF模型文件路径
- `scale_factor`: 控制缩放因子，影响操作幅度
- `control_rate_hz`: 控制频率
- `self_collision_avoidance_enabled`: 是否启用自碰撞避免
- `dt`: 时间步长

## 示例配置

项目包含多种机器人配置示例，位于`assets/`目录下:

- `all_robot/`: 双臂机器人URDF模型
- `right_rm65f/`: RM65右臂模型
- `rm65_f/`: RM65相关配置文件

## 主要类说明

- `BaseTeleopController`: 遥操作控制器抽象基类，定义了IK求解、运动学约束等核心功能
- `RobotTeleopController`: 机器人硬件控制器基类，实现硬件接口和线程管理
- `PlacoTeleopController`: 仿真环境控制器，基于Placo进行IK求解
- `AllRobotTeleopController`: 全机器人控制器，支持双臂协调控制

## 许可证

[请在此处添加许可证信息]