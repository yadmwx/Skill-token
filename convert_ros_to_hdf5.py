#!/usr/bin/env python3
"""
将 ROS 数据（RealSense + 睿尔曼机械臂）转换为 HDF5 格式，然后可以转换为 RLDS

使用方法:
    # 从 ROS bag 文件
    python convert_ros_to_hdf5.py --ros_bag_path /path/to/rosbag.bag --output_path data/real_robot.hdf5 --language_instruction "Pick up the red block"
    
    # 从实时 ROS 话题（需要先启动 ROS）
    python convert_ros_to_hdf5.py --ros_topics --output_path data/real_robot.hdf5 --language_instruction "Pick up the red block" --duration 60
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
from PIL import Image
import h5py

try:
    import rospy
    from sensor_msgs.msg import Image as ROSImage, CompressedImage
    from geometry_msgs.msg import Pose, PoseStamped, Twist
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    import rosbag
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("警告: ROS 相关库未安装")
    print("安装方法: sudo apt-get install ros-noetic-cv-bridge ros-noetic-sensor-msgs")
    print("或: pip install rospkg")

# 图像尺寸（VLA 标准）
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


class ROSDataCollector:
    """从 ROS 收集机器人数据"""
    
    def __init__(self, use_ros_topics: bool = False, ros_bag_path: Optional[str] = None):
        self.use_ros_topics = use_ros_topics
        self.ros_bag_path = ros_bag_path
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        
        # ROS 话题名称（根据你的实际设置修改）
        self.image_topic = "/camera/color/image_raw"  # RealSense RGB 图像
        self.depth_topic = "/camera/depth/image_rect_raw"  # RealSense 深度图像（可选）
        self.wrist_image_topic = "/camera/wrist/image_raw"  # 手腕相机（如果有）
        self.joint_state_topic = "/joint_states"  # 关节状态
        self.ee_pose_topic = "/ee_pose"  # 末端执行器位姿
        self.action_topic = "/robot_action"  # 动作命令
        
    def collect_from_bag(self, language_instruction: str) -> List[Dict[str, Any]]:
        """从 ROS bag 文件收集数据"""
        if not ROS_AVAILABLE:
            raise ImportError("ROS 库未安装，无法读取 bag 文件")
            
        if not os.path.exists(self.ros_bag_path):
            raise FileNotFoundError(f"ROS bag 文件不存在: {self.ros_bag_path}")
        
        print(f"正在读取 ROS bag 文件: {self.ros_bag_path}")
        bag = rosbag.Bag(self.ros_bag_path, 'r')
        
        # 获取所有话题
        topics = bag.get_type_and_topic_info()[1].keys()
        print(f"Bag 文件中的话题: {list(topics)}")
        
        trajectory = []
        data_buffer = {}  # 按时间戳缓冲数据
        
        # 按时间戳排序的消息
        messages = []
        for topic, msg, t in bag.read_messages():
            messages.append((t.to_sec(), topic, msg))
        messages.sort(key=lambda x: x[0])
        
        prev_ee_pose = None
        prev_joint_state = None
        
        for timestamp, topic, msg in messages:
            step_data = {}
            
            # 处理图像
            if topic == self.image_topic or 'image' in topic.lower() or 'camera' in topic.lower():
                try:
                    if hasattr(msg, 'encoding'):
                        if msg.encoding == 'rgb8':
                            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                        else:
                            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    elif hasattr(msg, 'format'):
                        # CompressedImage
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    else:
                        continue
                    
                    # Resize 到标准尺寸
                    cv_image = cv2.resize(cv_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    step_data['image'] = cv_image
                    step_data['timestamp'] = timestamp
                except Exception as e:
                    print(f"处理图像错误 ({topic}): {e}")
                    continue
            
            # 处理关节状态
            elif topic == self.joint_state_topic or 'joint' in topic.lower():
                try:
                    if hasattr(msg, 'position'):
                        joint_positions = np.array(msg.position[:7])  # 7自由度
                    elif hasattr(msg, 'data'):
                        joint_positions = np.array(msg.data[:7])
                    else:
                        continue
                    
                    step_data['joint_state'] = joint_positions
                    step_data['timestamp'] = timestamp
                    prev_joint_state = joint_positions
                except Exception as e:
                    print(f"处理关节状态错误 ({topic}): {e}")
                    continue
            
            # 处理末端执行器位姿
            elif topic == self.ee_pose_topic or 'ee_pose' in topic.lower() or 'end_effector' in topic.lower():
                try:
                    if hasattr(msg, 'pose'):  # PoseStamped
                        pose = msg.pose
                    elif hasattr(msg, 'position'):  # Pose
                        pose = msg
                    else:
                        continue
                    
                    ee_pos = [pose.position.x, pose.position.y, pose.position.z]
                    ee_ori = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    
                    # 转换为 roll, pitch, yaw
                    try:
                        from scipy.spatial.transform import Rotation
                        r = Rotation.from_quat(ee_ori)
                        euler = r.as_euler('xyz')
                        ee_pose = ee_pos + list(euler)
                    except:
                        # 如果没有 scipy，使用四元数
                        ee_pose = ee_pos + ee_ori
                    
                    step_data['ee_pose'] = np.array(ee_pose)
                    step_data['timestamp'] = timestamp
                    prev_ee_pose = ee_pose
                except Exception as e:
                    print(f"处理末端执行器位姿错误 ({topic}): {e}")
                    continue
            
            # 处理动作（相对增量）
            elif topic == self.action_topic or 'action' in topic.lower():
                try:
                    if hasattr(msg, 'data'):  # Float64MultiArray
                        action = np.array(msg.data[:7])  # 7维动作
                    elif hasattr(msg, 'linear'):  # Twist
                        action = np.array([
                            msg.linear.x, msg.linear.y, msg.linear.z,
                            msg.angular.x, msg.angular.y, msg.angular.z, 0.0
                        ])
                    else:
                        continue
                    
                    step_data['action'] = action
                    step_data['timestamp'] = timestamp
                except Exception as e:
                    print(f"处理动作错误 ({topic}): {e}")
                    continue
            
            # 合并同一时间戳的数据（10ms 容差）
            if 'timestamp' in step_data:
                existing_step = None
                for step in trajectory:
                    if abs(step.get('timestamp', 0) - step_data['timestamp']) < 0.01:
                        existing_step = step
                        break
                
                if existing_step:
                    existing_step.update(step_data)
                else:
                    step_data['language_instruction'] = language_instruction
                    trajectory.append(step_data)
        
        bag.close()
        
        # 按时间戳排序
        trajectory.sort(key=lambda x: x.get('timestamp', 0))
        
        # 计算动作增量（如果动作是绝对位置）
        if prev_ee_pose is not None:
            for i, step in enumerate(trajectory):
                if 'action' not in step and 'ee_pose' in step:
                    if i > 0 and 'ee_pose' in trajectory[i-1]:
                        # 计算相对增量
                        prev_pose = trajectory[i-1]['ee_pose']
                        curr_pose = step['ee_pose']
                        action = np.array(curr_pose[:6]) - np.array(prev_pose[:6])  # 位置和姿态增量
                        # 添加夹爪动作（如果有）
                        if 'joint_state' in step and len(step['joint_state']) > 0:
                            gripper_action = step['joint_state'][-1] - trajectory[i-1].get('joint_state', np.zeros(7))[-1]
                        else:
                            gripper_action = 0.0
                        action = np.append(action, gripper_action)
                        step['action'] = action[:7]  # 只取前7维
        
        # 确保每个步骤都有必要的字段
        for step in trajectory:
            if 'action' not in step:
                step['action'] = np.zeros(7, dtype=np.float32)
            if 'image' not in step:
                # 创建占位图像
                step['image'] = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            if 'joint_state' not in step:
                step['joint_state'] = np.zeros(7, dtype=np.float32)
            if 'ee_pose' not in step:
                step['ee_pose'] = np.zeros(6, dtype=np.float32)
        
        print(f"从 bag 文件收集了 {len(trajectory)} 个步骤")
        return trajectory
    
    def collect_from_topics(self, language_instruction: str, duration: float = 60.0) -> List[Dict[str, Any]]:
        """从实时 ROS 话题收集数据"""
        if not ROS_AVAILABLE:
            raise ImportError("ROS 库未安装，无法订阅话题")
        
        rospy.init_node('data_collector', anonymous=True)
        
        trajectory = []
        data_lock = {}
        
        def image_callback(msg, data_key='image'):
            try:
                if hasattr(msg, 'encoding'):
                    if msg.encoding == 'rgb8':
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                    else:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                else:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
                cv_image = cv2.resize(cv_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data_lock[data_key] = cv_image
                data_lock[f'{data_key}_time'] = rospy.Time.now().to_sec()
            except Exception as e:
                print(f"处理图像错误: {e}")
        
        def joint_state_callback(msg):
            try:
                if hasattr(msg, 'position'):
                    joint_positions = np.array(msg.position[:7])
                elif hasattr(msg, 'data'):
                    joint_positions = np.array(msg.data[:7])
                else:
                    return
                data_lock['joint_state'] = joint_positions
                data_lock['joint_state_time'] = rospy.Time.now().to_sec()
            except Exception as e:
                print(f"处理关节状态错误: {e}")
        
        def action_callback(msg):
            try:
                if hasattr(msg, 'data'):
                    action = np.array(msg.data[:7])
                elif hasattr(msg, 'linear'):
                    action = np.array([
                        msg.linear.x, msg.linear.y, msg.linear.z,
                        msg.angular.x, msg.angular.y, msg.angular.z, 0.0
                    ])
                else:
                    return
                data_lock['action'] = action
                data_lock['action_time'] = rospy.Time.now().to_sec()
            except Exception as e:
                print(f"处理动作错误: {e}")
        
        # 订阅话题
        rospy.Subscriber(self.image_topic, ROSImage, lambda msg: image_callback(msg, 'image'))
        if self.wrist_image_topic:
            rospy.Subscriber(self.wrist_image_topic, ROSImage, lambda msg: image_callback(msg, 'wrist_image'))
        rospy.Subscriber(self.joint_state_topic, JointState, joint_state_callback)
        rospy.Subscriber(self.action_topic, Float64MultiArray, action_callback)
        
        print(f"开始收集数据，持续 {duration} 秒...")
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(30)  # 30 Hz
        
        while not rospy.is_shutdown() and (rospy.Time.now().to_sec() - start_time) < duration:
            current_time = rospy.Time.now().to_sec()
            step = {
                'timestamp': current_time,
                'language_instruction': language_instruction,
            }
            
            # 复制当前数据（100ms 容差）
            if 'image' in data_lock and abs(current_time - data_lock.get('image_time', 0)) < 0.1:
                step['image'] = data_lock['image'].copy()
            if 'wrist_image' in data_lock and abs(current_time - data_lock.get('wrist_image_time', 0)) < 0.1:
                step['wrist_image'] = data_lock['wrist_image'].copy()
            if 'joint_state' in data_lock and abs(current_time - data_lock.get('joint_state_time', 0)) < 0.1:
                step['joint_state'] = data_lock['joint_state'].copy()
            if 'action' in data_lock and abs(current_time - data_lock.get('action_time', 0)) < 0.1:
                step['action'] = data_lock['action'].copy()
            
            if len(step) > 2:  # 除了 timestamp 和 language_instruction 还有其他数据
                trajectory.append(step)
            
            rate.sleep()
        
        print(f"收集了 {len(trajectory)} 个步骤")
        return trajectory


def save_to_hdf5(trajectory: List[Dict[str, Any]], output_path: Path, episode_id: int = 0):
    """将轨迹保存为 HDF5 格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件存在，追加模式；否则创建新文件
    if output_path.exists():
        f = h5py.File(output_path, 'a')
    else:
        f = h5py.File(output_path, 'w')
        grp = f.create_group("data")
        grp.attrs["num_demos"] = 0
        grp.attrs["total"] = 0
    
    grp = f["data"]
    
    # 准备数据
    images = []
    actions = []
    joint_states = []
    ee_poses = []
    language_instructions = []
    
    for step in trajectory:
        if 'image' in step:
            images.append(step['image'])
        else:
            images.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
        
        if 'action' in step:
            action = step['action']
            if len(action) < 7:
                action = np.pad(action, (0, 7 - len(action)), 'constant')
            actions.append(action[:7])
        else:
            actions.append(np.zeros(7, dtype=np.float32))
        
        if 'joint_state' in step:
            joint_state = step['joint_state']
            if len(joint_state) < 7:
                joint_state = np.pad(joint_state, (0, 7 - len(joint_state)), 'constant')
            joint_states.append(joint_state[:7])
        else:
            joint_states.append(np.zeros(7, dtype=np.float32))
        
        if 'ee_pose' in step:
            ee_pose = step['ee_pose']
            if len(ee_pose) < 6:
                ee_pose = np.pad(ee_pose, (0, 6 - len(ee_pose)), 'constant')
            ee_poses.append(ee_pose[:6])
        else:
            ee_poses.append(np.zeros(6, dtype=np.float32))
        
        language_instructions.append(step.get('language_instruction', '').encode('utf-8'))
    
    # 创建 episode 组
    ep_data_grp = grp.create_group(f"demo_{episode_id}")
    
    obs_grp = ep_data_grp.create_group("obs")
    obs_grp.create_dataset("agentview_rgb", data=np.stack(images, axis=0))
    obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
    obs_grp.create_dataset("ee_states", data=np.stack(ee_poses, axis=0))
    
    ep_data_grp.create_dataset("actions", data=np.stack(actions, axis=0))
    ep_data_grp.create_dataset("language_instruction", data=language_instructions)
    ep_data_grp.attrs["num_samples"] = len(trajectory)
    ep_data_grp.attrs["language_instruction"] = trajectory[0].get('language_instruction', '')
    
    # 更新统计信息
    grp.attrs["num_demos"] = len([k for k in grp.keys() if k.startswith("demo_")])
    grp.attrs["total"] = sum(grp[f"demo_{i}"].attrs["num_samples"] 
                             for i in range(grp.attrs["num_demos"]))
    
    f.close()
    print(f"已保存轨迹到: {output_path} (episode {episode_id}, {len(trajectory)} 步)")


def main():
    parser = argparse.ArgumentParser(description="将 ROS 数据转换为 HDF5 格式")
    parser.add_argument("--ros_bag_path", type=str, help="ROS bag 文件路径")
    parser.add_argument("--ros_topics", action="store_true", help="从实时 ROS 话题收集数据")
    parser.add_argument("--output_path", type=str, required=True, help="输出 HDF5 文件路径")
    parser.add_argument("--language_instruction", type=str, required=True, help="语言指令")
    parser.add_argument("--duration", type=float, default=60.0, help="收集数据持续时间（秒）")
    parser.add_argument("--image_topic", type=str, default="/camera/color/image_raw", help="图像话题")
    parser.add_argument("--joint_state_topic", type=str, default="/joint_states", help="关节状态话题")
    parser.add_argument("--action_topic", type=str, default="/robot_action", help="动作话题")
    parser.add_argument("--episode_id", type=int, default=0, help="Episode ID（用于追加多个轨迹）")
    
    args = parser.parse_args()
    
    # 创建收集器
    collector = ROSDataCollector(
        use_ros_topics=args.ros_topics,
        ros_bag_path=args.ros_bag_path
    )
    
    # 设置话题名称
    collector.image_topic = args.image_topic
    collector.joint_state_topic = args.joint_state_topic
    collector.action_topic = args.action_topic
    
    # 收集数据
    if args.ros_bag_path:
        trajectory = collector.collect_from_bag(args.language_instruction)
    elif args.ros_topics:
        trajectory = collector.collect_from_topics(args.language_instruction, args.duration)
    else:
        raise ValueError("必须指定 --ros_bag_path 或 --ros_topics")
    
    # 保存为 HDF5
    save_to_hdf5(trajectory, Path(args.output_path), args.episode_id)
    
    print("转换完成！")
    print(f"下一步: 使用 convert_hdf5_to_rlds.py 将 HDF5 转换为 RLDS 格式")


if __name__ == "__main__":
    main()
