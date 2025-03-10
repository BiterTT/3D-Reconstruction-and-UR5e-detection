#! /usr/bin/env python3
from trajectory_msgs.msg import *
from control_msgs.msg import *
from sensor_msgs.msg import  *
import rospy
import actionlib
import cv2
import numpy as np
from cv_bridge import CvBridge

import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


initial_shoulder_lift_joint =  -1.5707 
initial_shoulder_pan_joint = 0 
initial_wrist_1_joint = 1.5707 
initial_wrist_2_joint = 1.5707
initial_wrist_3_joint = 0
home_joint_positions = [-1.5707, -1.5707, 0, 1.5707, 1.5707, 0]
class UR5Grasper:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("ur5_grasping_control", anonymous=True)
        # 初始化 moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        #self.arm = moveit_commander.MoveGroupCommander("ur5_arm_controller")
        self.arm_client = actionlib.SimpleActionClient(
            "/ur5/ur5_arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for arm controller...")
        self.arm_client.wait_for_server()

        rospy.loginfo("Arm controller connected.")
        #self.gripper_pub = rospy.Publisher("/gripper/command", String, queue_size=10)
        rospy.sleep(1)

    def move_to_home(self):
        
        self.arm.set_joint_value_target(home_joint_positions)
        self.arm.go(wait=True)
    def move_to_joint_positions(self, joint_positions):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joint_positions

        point.time_from_start = rospy.Duration(2.0)  # 设定2秒内完成

        goal.trajectory.points.append(point)
        self.arm_client.send_goal(goal)

        self.arm_client.wait_for_result()

class UR5Gripper:
    def __init__(self):
        #rospy.init_node("ur5_gripper_control", anonymous=True)
        self.gripper_client = actionlib.SimpleActionClient(
            "/ur5/gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
            
        rospy.loginfo("Waiting for gripper controller...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper controller connected.")

        rospy.sleep(1)
    def control_gripper(self, open_gripper=True):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["robotiq_85_left_knuckle_joint"]  # 夹爪的关节名称

        gripper_point = JointTrajectoryPoint()

        gripper_point.positions = [0.2 if open_gripper else 0.4]

        gripper_point.time_from_start = rospy.Duration(1.0)

        goal.trajectory.points.append(gripper_point)

        self.gripper_client.send_goal(goal)

        self.gripper_client.wait_for_result()

class RealsenseObjectDetector:
    def __init__(self):
        #rospy.init_node("realsense_rgb_detector", anonymous=True)
        self.bridge = CvBridge()
        
        # 订阅 深度图像
        self.depth_sub = rospy.Subscriber("/d435/depth/image_raw", Image, self.depth_callback)
        # 订阅 相机内参
        self.camera_info_sub = rospy.Subscriber("/d435/color/camera_info", CameraInfo, self.camera_info_callback)

        # 订阅 RGB 图像
        self.image_sub = rospy.Subscriber("/d435/color/image_raw", Image, self.image_callback)
        self.depth_image = None
        self.fx = self.fy = self.ppx = self.ppy = None  # 相机内参

    def camera_info_callback(self, msg):
        """ 仅加载一次相机内参 """
        if self.fx is None:  # 仅在未加载时赋值
            self.fx = msg.K[0]  # 焦距 fx
            self.fy = msg.K[4]  # 焦距 fy
            self.ppx = msg.K[2]  # 主点 x
            self.ppy = msg.K[5]  # 主点 y
            rospy.loginfo("Camera Intrinsics Loaded!")


    def depth_callback(self, msg):
        """ 读取深度图数据 """
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # 深度单位：mm

    def image_callback(self, msg):
        """ 处理 RGB 图像，检测红色物体，并计算 3D 坐标 """
        if self.depth_image is None or self.fx is None:
            return  # 确保深度数据和内参已就绪

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")#将 ROS Image 转换为 OpenCV 格式（BGR 格式）。

        # 颜色检测（红色物体）
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # 过滤噪声
                x, y, w, h = cv2.boundingRect(cnt)
                center_x, center_y = x + w // 2, y + h // 2

                # 获取深度值（mm）
                depth = self.depth_image[center_y, center_x]
                if depth == 0 or depth > 200:  # 2m 以上的深度被忽略
                    continue  # 忽略无效深度值

                # 计算 3D 坐标
                world_x, world_y, world_z = self.pixel_to_world(center_x, center_y, depth)

                rospy.loginfo(f"Object Position: x={world_x:.3f}, y={world_y:.3f}, z={world_z:.3f}")

                # 画框和中心点
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(cv_image, (center_x, center_y), 5, (255, 0, 0), -1)

        cv2.imshow("Object Detection", cv_image)
        cv2.waitKey(1)
    
    def pixel_to_world(self, u, v, depth):
        """ 使用相机内参，将像素 (u, v) 转换为 3D 坐标 """
        x = (u - self.ppx) * depth / self.fx
        y = (v - self.ppy) * depth / self.fy
        z = depth / 1000.0  # mm 转换为 m
        return x, y, z

        

if __name__ == "__main__":
    controller = UR5Grasper()
    gripper = UR5Gripper()
    
    # 让机械臂移动到初始位置
    controller.move_to_joint_positions([0, -1.57, 1.57, -1.57, -1.57, 0])
    detector = RealsenseObjectDetector()
    
    # #rospy.loginfo("Closing gripper...")
    # # gripper.control_gripper(open_gripper=False)  # 夹爪闭合
    # rospy.sleep(2)
    # rospy.loginfo("Opening gripper...")
    # gripper.control_gripper(open_gripper=True)   # 夹爪张开

    
    rospy.spin()
