#!/usr/bin/env python3

import rospy
import sys
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import actionlib

from trajectory_msgs.msg import *
from control_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from tf.transformations import quaternion_from_euler
from copy import deepcopy


home_joint_positions = [0, -1.57, 1.57, -1.57, -1.57, 0]

class UR5GraspSystem:
    def __init__(self):
        rospy.init_node("ur5_grasp_system")

        # MoveIt setup
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.arm = MoveGroupCommander("ur5_arm")
        self.arm.set_planning_time(5.0)
        self.arm.set_num_planning_attempts(10)
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.01)

        # Gripper setup
        self.gripper_client = actionlib.SimpleActionClient("/ur5/gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for gripper controller...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper controller connected.")

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Camera setup
        self.bridge = CvBridge()
        self.depth_image = None
        self.fx = self.fy = self.cx = self.cy = None

        self.target_pose = None
        self.processing = False

        rospy.Subscriber("/d435/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/d435/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/d435/depth/image_raw", Image, self.depth_callback)

        self.move_to_home()

    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx, self.fy, self.cx, self.cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
            rospy.loginfo("Camera intrinsics loaded.")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def image_callback(self, msg):
        if self.processing or self.target_pose is not None or self.depth_image is None or self.fx is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            depth_window = self.depth_image[cy-2:cy+3, cx-2:cx+3].astype(np.float32)
            valid_window = depth_window[(depth_window > 100) & (depth_window < 1500)]
            # depth = self.depth_image[cy, cx] / 1000.0
            if len(valid_window) < 5:
                continue
            depth = np.median(valid_window) / 1000.0
            if depth <= 0.1 or depth > 0.5:
                continue
            cam_x = (cx - self.cx) * depth / self.fx
            cam_y = (cy - self.cy) * depth / self.fy
            cam_z = depth

            rospy.loginfo(f"[CAM COORD] cam_x: {cam_x:.4f}, cam_y: {cam_y:.4f}, cam_z: {cam_z:.4f}")
            object_pose = self.transform_pose_to_base(cam_x, cam_y, cam_z)
            if object_pose:
                self.target_pose = object_pose
                break

    def smooth_target_with_stability(self, cam_x, cam_y, cam_z):
        if not hasattr(self, 'coord_buffer'):
            self.coord_buffer = []
        self.coord_buffer.append([cam_x, cam_y, cam_z])
        if len(self.coord_buffer) > 5:
            self.coord_buffer.pop(0)

        # 判断位置是否稳定
        coords = np.array(self.coord_buffer)
        std_dev = np.std(coords, axis=0)
        if all(std_dev < [0.005, 0.005, 0.01]):  # XY方向标准差小于5mm
            mean_coord = np.mean(coords, axis=0)
            self.coord_buffer.clear()  # 清空缓冲，防止后续重复抓
            return mean_coord[0], mean_coord[1], mean_coord[2], True  # ✅ 代表“稳定了”
        else:
            return cam_x, cam_y, cam_z, False  



    def transform_pose_to_base(self, x, y, z):
        pose_cam = tf2_geometry_msgs.PoseStamped()
        pose_cam.header.frame_id = "d435_color_optical_frame"
        pose_cam.header.stamp = rospy.Time.now()
        pose_cam.pose.position.x = x
        pose_cam.pose.position.y = y
        pose_cam.pose.position.z = z
        pose_cam.pose.orientation.w = 1.0
        try:
            trans = self.tf_buffer.lookup_transform("base_link", "d435_color_optical_frame", rospy.Time(0), rospy.Duration(2.0))
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, trans)
            return pose_base.pose
        except Exception as e:
            rospy.logwarn("TF error: " + str(e))
            return None

    def execute_grasp(self, pose):
        #pose.position.x,pose.position.y,pose.position.z = self.smooth_target(pose.position.x,pose.position.y,pose.position.z)
        rospy.loginfo(f" pose: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
        q = quaternion_from_euler(np.pi/2, np.pi/2, np.pi/2)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        if pose.position.z < 0.08 or pose.position.z > 0.45:
            pose.position.z = 0.3  # 强制修正抓取高度

        approach_pose = deepcopy(pose)
        approach_pose.position.z += 0.15
        self.arm.set_pose_target(approach_pose)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        
        rospy.sleep(1)

        self.control_gripper(open_gripper=False)
        rospy.sleep(1)

        lift_pose = deepcopy(pose)
        lift_pose.position.z += 0.35
        self.arm.set_pose_target(lift_pose)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        self.move_to_home()


        # pose.position.y -= 0.1
        # rospy.loginfo(f"estimate pose: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
        # #pose.position.z -= 0.1
        # approach_pose = Pose()
        # approach_pose.position.x = pose.position.x
        # approach_pose.position.y = pose.position.y
        # approach_pose.position.z = pose.position.z + 0.5
        # approach_pose.orientation = pose.orientation
        # rospy.loginfo(f"Approach pose: x={approach_pose.position.x:.3f}, y={approach_pose.position.y:.3f}, z={approach_pose.position.z:.3f}")

        # self.arm.set_pose_target(approach_pose)
        # if not self.arm.go(wait=True):
        #     rospy.logwarn("Failed to move to approach pose.")
            
        # self.arm.stop()
        # self.arm.clear_pose_targets()
        # rospy.sleep(0.5)

        # self.arm.set_pose_target(pose)
        # if not self.arm.go(wait=True):
        #     rospy.logwarn("Failed to move to grasp pose.")
            
        # self.arm.stop()
        # self.arm.clear_pose_targets()
        # rospy.sleep(0.5)

        # rospy.sleep(1.0)
        # self.control_gripper(open_gripper=False)
        # rospy.sleep(1.0)
        # self.lift_arm()

        # self.move_to_home()
        # rospy.sleep(2.0)

    def control_gripper(self, open_gripper=True):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["robotiq_85_left_knuckle_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.0 if open_gripper else 0.1]
        point.time_from_start = rospy.Duration(1.0)
        goal.trajectory.points.append(point)
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result()

    def lift_arm(self):
        pose = self.arm.get_current_pose().pose
        pose.position.z += 0.5
        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

    def move_to_home(self):
        self.arm.set_joint_value_target(home_joint_positions)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(5)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.target_pose and not self.processing:
                self.processing = True
                self.execute_grasp(self.target_pose)
                self.target_pose = None
                self.processing = False
            rate.sleep()

if __name__ == "__main__":
    try:
        grasp_system = UR5GraspSystem()
        grasp_system.run()
    except rospy.ROSInterruptException:
        pass
