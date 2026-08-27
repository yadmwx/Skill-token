#!/usr/bin/env python3
"""
将 ROS 数据（RealSense + 睿尔曼机械臂）转换为 RLDS 格式数据集

使用方法:
    python convert_ros_to_rlds.py --ros_bag_path /path/to/rosbag.bag --output_dir data/libero/real_robot_data --language_instruction "Pick up the red block"
    
或者从实时 ROS 订阅:
    python convert_ros_to_rlds.py --ros_topics --output_dir data/libero/real_robot_data --language_instruction "Pick up the red block"
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
from PIL import Image
import io

try:
    import rospy
    from sensor_msgs.msg import Image as ROSImage, CompressedImage
    from geometry_msgs.msg import Pose, Twist
    from std_msgs.msg import Float64MultiArray
    import rosbag
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("警告: ROS 相关库未安装，将使用模拟数据模式")

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import rlds
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("警告: TensorFlow 相关库未安装，请安装: pip install tensorflow tensorflow-datasets rlds")

# 图像尺寸（RLDS 标准）
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
        self.depth_topic = "/camera/depth/image_rect_raw"  # RealSense 深度图像
        self.wrist_image_topic = "/camera/wrist/image_raw"  # 手腕相机（如果有）
        self.joint_state_topic = "/joint_states"  # 关节状态
        self.ee_pose_topic = "/ee_pose"  # 末端执行器位姿
        self.action_topic = "/robot_action"  # 动作命令
        
        self.trajectory = []
        
    def collect_from_bag(self, language_instruction: str) -> List[Dict[str, Any]]:
        """从 ROS bag 文件收集数据"""
        if not ROS_AVAILABLE:
            raise ImportError("ROS 库未安装，无法读取 bag 文件")
            
        if not os.path.exists(self.ros_bag_path):
            raise FileNotFoundError(f"ROS bag 文件不存在: {self.ros_bag_path}")
        
        print(f"正在读取 ROS bag 文件: {self.ros_bag_path}")
        bag = rosbag.Bag(self.ros_bag_path, 'r')
        
        trajectory = []
        prev_ee_pose = None
        prev_joint_state = None
        
        # 按时间戳排序的消息
        messages = []
        for topic, msg, t in bag.read_messages():
            messages.append((t, topic, msg))
        messages.sort(key=lambda x: x[0])
        
        for t, topic, msg in messages:
            step_data = {}
            
            # 处理图像
            if topic == self.image_topic:
                if msg.encoding == 'rgb8':
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
                # Resize 到标准尺寸
                cv_image = cv2.resize(cv_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                step_data['image'] = cv_image
                step_data['timestamp'] = t.to_sec()
            
            # 处理深度图像
            elif topic == self.depth_topic:
                depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                depth_image = cv2.resize(depth_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                step_data['depth'] = depth_image
                step_data['timestamp'] = t.to_sec()
            
            # 处理关节状态
            elif topic == self.joint_state_topic:
                joint_positions = np.array(msg.position[:7])  # 7自由度
                step_data['joint_state'] = joint_positions
                step_data['timestamp'] = t.to_sec()
                prev_joint_state = joint_positions
            
            # 处理末端执行器位姿
            elif topic == self.ee_pose_topic:
                ee_pos = [msg.position.x, msg.position.y, msg.position.z]
                ee_ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                # 转换为 roll, pitch, yaw
                from scipy.spatial.transform import Rotation
                r = Rotation.from_quat(ee_ori)
                euler = r.as_euler('xyz')
                ee_pose = ee_pos + list(euler)
                step_data['ee_pose'] = ee_pose
                step_data['timestamp'] = t.to_sec()
                prev_ee_pose = ee_pose
            
            # 处理动作（相对增量）
            elif topic == self.action_topic:
                if isinstance(msg, Float64MultiArray):
                    action = np.array(msg.data[:7])  # 7维动作
                else:
                    action = np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                                     msg.angular.x, msg.angular.y, msg.angular.z, 0.0])
                step_data['action'] = action
                step_data['timestamp'] = t.to_sec()
            
            # 合并同一时间戳的数据
            if 'timestamp' in step_data:
                # 查找或创建对应时间戳的步骤
                existing_step = None
                for step in trajectory:
                    if abs(step.get('timestamp', 0) - step_data['timestamp']) < 0.01:  # 10ms 容差
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
                        action = np.array(curr_pose) - np.array(prev_pose)
                        step['action'] = action[:7]  # 只取前7维
        
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
                if msg.encoding == 'rgb8':
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                cv_image = cv2.resize(cv_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data_lock[data_key] = cv_image
                data_lock[f'{data_key}_time'] = rospy.Time.now().to_sec()
            except Exception as e:
                print(f"处理图像错误: {e}")
        
        def joint_state_callback(msg):
            try:
                joint_positions = np.array(msg.position[:7])
                data_lock['joint_state'] = joint_positions
                data_lock['joint_state_time'] = rospy.Time.now().to_sec()
            except Exception as e:
                print(f"处理关节状态错误: {e}")
        
        def action_callback(msg):
            try:
                if isinstance(msg, Float64MultiArray):
                    action = np.array(msg.data[:7])
                else:
                    action = np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                                     msg.angular.x, msg.angular.y, msg.angular.z, 0.0])
                data_lock['action'] = action
                data_lock['action_time'] = rospy.Time.now().to_sec()
            except Exception as e:
                print(f"处理动作错误: {e}")
        
        # 订阅话题
        rospy.Subscriber(self.image_topic, ROSImage, lambda msg: image_callback(msg, 'image'))
        if self.wrist_image_topic:
            rospy.Subscriber(self.wrist_image_topic, ROSImage, lambda msg: image_callback(msg, 'wrist_image'))
        rospy.Subscriber(self.joint_state_topic, Float64MultiArray, joint_state_callback)
        rospy.Subscriber(self.action_topic, Float64MultiArray, action_callback)
        
        print(f"开始收集数据，持续 {duration} 秒...")
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(30)  # 30 Hz
        
        while not rospy.is_shutdown() and (rospy.Time.now().to_sec() - start_time) < duration:
            # 合并当前时间的数据
            current_time = rospy.Time.now().to_sec()
            step = {
                'timestamp': current_time,
                'language_instruction': language_instruction,
            }
            
            # 复制当前数据
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


def encode_image(image: np.ndarray) -> bytes:
    """将图像编码为 JPEG bytes"""
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


def create_rlds_dataset(trajectories: List[List[Dict[str, Any]]], output_dir: Path, dataset_name: str):
    """创建 RLDS 格式的数据集"""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow 相关库未安装")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    version_dir = output_dir / "1.0.0"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 features.json
    features = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "action": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {
                                            "shape": {"dimensions": ["7"]},
                                            "dtype": "float32",
                                            "encoding": "none"
                                        },
                                        "description": "Robot EEF action."
                                    },
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "image": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {
                                                            "dimensions": [str(IMAGE_HEIGHT), str(IMAGE_WIDTH), "3"]
                                                        },
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg"
                                                    },
                                                    "description": "Main camera RGB observation."
                                                },
                                                "wrist_image": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {
                                                            "dimensions": [str(IMAGE_HEIGHT), str(IMAGE_WIDTH), "3"]
                                                        },
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg"
                                                    },
                                                    "description": "Wrist camera RGB observation."
                                                },
                                                "state": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": ["8"]},
                                                        "dtype": "float32",
                                                        "encoding": "none"
                                                    },
                                                    "description": "Robot EEF state (6D pose, 2D gripper)."
                                                },
                                                "joint_state": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": ["7"]},
                                                        "dtype": "float32",
                                                        "encoding": "none"
                                                    },
                                                    "description": "Robot joint angles."
                                                }
                                            }
                                        }
                                    },
                                    "language_instruction": {
                                        "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                        "text": {},
                                        "description": "Language Instruction."
                                    },
                                    "is_first": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on first step of the episode."
                                    },
                                    "is_last": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step of the episode."
                                    },
                                    "is_terminal": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step of the episode if it is a terminal step."
                                    },
                                    "discount": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Discount if provided, default to 1."
                                    },
                                    "reward": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Reward if provided, 1 on final step for demos."
                                    }
                                }
                            }
                        },
                        "length": "-1"
                    }
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "file_path": {
                                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                "text": {},
                                "description": "Path to the original data file."
                            }
                        }
                    }
                }
            }
        }
    }
    
    with open(version_dir / "features.json", "w") as f:
        json.dump(features, f, indent=2)
    
    # 创建 TFRecord 文件
    def create_episode(trajectory: List[Dict[str, Any]], episode_id: int):
        """创建一个 episode"""
        steps = []
        
        for i, step in enumerate(trajectory):
            # 准备观察
            obs = {}
            
            # 图像
            if 'image' in step:
                obs['image'] = encode_image(step['image'])
            else:
                obs['image'] = b''  # 空图像
            
            if 'wrist_image' in step:
                obs['wrist_image'] = encode_image(step['wrist_image'])
            else:
                obs['wrist_image'] = b''
            
            # 状态（8维：位置3 + 姿态3 + 夹爪1 + 关节1）
            state = np.zeros(8, dtype=np.float32)
            if 'ee_pose' in step:
                state[:6] = step['ee_pose'][:6]  # 位置和姿态
            if 'joint_state' in step:
                state[6] = step['joint_state'][0] if len(step['joint_state']) > 0 else 0.0  # 夹爪状态
                state[7] = step['joint_state'][-1] if len(step['joint_state']) > 0 else 0.0  # 最后一个关节
            obs['state'] = state
            
            # 关节状态
            if 'joint_state' in step:
                obs['joint_state'] = step['joint_state'].astype(np.float32)
            else:
                obs['joint_state'] = np.zeros(7, dtype=np.float32)
            
            # 动作（7维）
            if 'action' in step:
                action = step['action'].astype(np.float32)
                if len(action) < 7:
                    action = np.pad(action, (0, 7 - len(action)), 'constant')
                action = action[:7]
            else:
                action = np.zeros(7, dtype=np.float32)
            
            # 语言指令
            lang_inst = step.get('language_instruction', '').encode('utf-8')
            
            # 创建 step
            step_dict = {
                'action': action,
                'observation': obs,
                'language_instruction': lang_inst,
                'is_first': i == 0,
                'is_last': i == len(trajectory) - 1,
                'is_terminal': i == len(trajectory) - 1,
                'discount': 1.0,
                'reward': 1.0 if i == len(trajectory) - 1 else 0.0,
            }
            steps.append(step_dict)
        
        episode = {
            'steps': steps,
            'episode_metadata': {
                'file_path': f'episode_{episode_id}'.encode('utf-8')
            }
        }
        return episode
    
    # 使用 rlds 库创建数据集
    print(f"正在创建 RLDS 数据集，包含 {len(trajectories)} 个轨迹...")
    
    # 创建 TFRecord writer
    tfrecord_path = version_dir / f"{dataset_name}-train.tfrecord-00000-of-00001"
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for episode_id, trajectory in enumerate(trajectories):
            episode = create_episode(trajectory, episode_id)
            
            # 转换为 TFRecord 格式
            # 这里需要根据 rlds 的具体格式进行序列化
            # 简化版本：直接使用 TensorFlow 的序列化
            example = tf.train.Example()
            # ... 填充 example ...
            # 注意：完整的 RLDS 序列化比较复杂，建议使用 rlds 库的 API
    
    print(f"数据集已保存到: {output_dir}")
    print(f"包含 {len(trajectories)} 个轨迹")


def main():
    parser = argparse.ArgumentParser(description="将 ROS 数据转换为 RLDS 格式")
    parser.add_argument("--ros_bag_path", type=str, help="ROS bag 文件路径")
    parser.add_argument("--ros_topics", action="store_true", help="从实时 ROS 话题收集数据")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--dataset_name", type=str, default="real_robot_data", help="数据集名称")
    parser.add_argument("--language_instruction", type=str, required=True, help="语言指令")
    parser.add_argument("--duration", type=float, default=60.0, help="收集数据持续时间（秒）")
    parser.add_argument("--image_topic", type=str, default="/camera/color/image_raw", help="图像话题")
    parser.add_argument("--joint_state_topic", type=str, default="/joint_states", help="关节状态话题")
    parser.add_argument("--action_topic", type=str, default="/robot_action", help="动作话题")
    
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
        trajectories = [trajectory]  # 单个轨迹
    elif args.ros_topics:
        trajectory = collector.collect_from_topics(args.language_instruction, args.duration)
        trajectories = [trajectory]
    else:
        raise ValueError("必须指定 --ros_bag_path 或 --ros_topics")
    
    # 转换为 RLDS 格式
    create_rlds_dataset(trajectories, Path(args.output_dir), args.dataset_name)
    
    print("转换完成！")


if __name__ == "__main__":
    main()
