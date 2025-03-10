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
from sensor_msgs.msg import  *
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge

import moveit_commander
from moveit_commander import MoveGroupCommander

from tf.transformations import quaternion_from_euler

home_joint_positions = [0, -1.57, 1.57, -1.57, -1.57, 0]

class GraspNode:
    def __init__(self):
        rospy.init_node("ur5_grasp_moveit_node")
        
        #moveit initialization
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm = MoveGroupCommander("ur5_arm")

        #planning parameters setting
        self.arm.set_planning_time(5.0)
        self.arm.set_num_planning_attempts(10)

        self.arm.set_goal_position_tolerance(0.05)
        self.arm.set_goal_orientation_tolerance(0.05)

        #gripper initial
        self.gripper_client = actionlib.SimpleActionClient(
            "/ur5/gripper_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
            
        rospy.loginfo("Waiting for gripper controller...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper controller connected.")
        rospy.sleep(1)

        self.move_to_home()

        self.bridge = CvBridge()

        # camera
        self.camera_info_sub = rospy.Subscriber("/d435/color/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber("/d435/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/d435/depth/image_raw", Image, self.depth_callback)

        #tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.fx = self.fy = self.cx = self.cy = None
        
    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            rospy.loginfo("Camera intrinsics loaded.")
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def image_callback(self, msg):
        
        if self.depth_image is None or self.fx is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #rospy.loginfo(f"[DEBUG] Found {len(contours)} contours")
        for cnt in contours:
            
            area = cv2.contourArea(cnt)
            if area > 100:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    continue

                depth = self.depth_image[center_y, center_x] / 1000.0  #mm-----m
                rospy.loginfo(f"[DEBUG] Detected center ({center_x},{center_y}), depth={depth:.3f} m")
                # if depth <= 0.1 or depth > 2.0:
                #     continue

                cam_x = (center_x - self.cx) * depth / self.fx
                cam_y = (center_y - self.cy) * depth / self.fy
                cam_z = depth

                object_pose = self.transform_pose_to_base(cam_x, cam_y, cam_z)
                if object_pose:
                    self.move_to_grasp(object_pose)
                    break
        
        #cv2.imshow("Detection", cv_image)
        cv2.waitKey(1)
    
    def transform_pose_to_base(self, x, y, z):
        pose_cam = tf2_geometry_msgs.PoseStamped()
        pose_cam.header.frame_id = "d435_color_optical_frame"
        pose_cam.pose.position.x = x
        pose_cam.pose.position.y = y
        pose_cam.pose.position.z = z
        pose_cam.pose.orientation.w = 1.0

        try:
            self.tf_buffer.lookup_transform("base_link", "d435_color_optical_frame", rospy.Time(0), rospy.Duration(1.0))

            transform = self.tf_buffer.lookup_transform("base_link", "d435_color_optical_frame", rospy.Time(0), rospy.Duration(1.0))
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
            return pose_base.pose
        except Exception as e:
            rospy.logwarn("TF transform failed: " + str(e))
            return None

    def move_to_grasp(self, pose):
        MIN_SAFE_Z = 0.05
        if pose.position.z < MIN_SAFE_Z:
            pose.position.z = MIN_SAFE_Z

        rospy.loginfo(f"[DEBUG] Executing move_to_grasp: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")

        q = quaternion_from_euler(np.pi/2, np.pi/2, np.pi/2)
        
        #pose.position.y -= 0.05
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        approach_pose = Pose()
        approach_pose.position.x = pose.position.x
        approach_pose.position.y = pose.position.y
        approach_pose.position.z = pose.position.z + 0.3  # 抬高一些
        approach_pose.orientation = pose.orientation

        self.arm.set_pose_target(approach_pose)
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        if not success:
            rospy.logwarn("Approach move failed.")
            return

        pose.position.z = pose.position.z + 0.2  # 抬高一些
        self.arm.set_pose_target(pose)
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(1.0)
        if not success:
            rospy.logwarn("Grasp move failed.")
            return

        rospy.sleep(1.0)
        self.control_gripper(open_gripper=False)
        rospy.sleep(1.0)
        self.lift_arm()
            
    def control_gripper(self, open_gripper=True):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["robotiq_85_left_knuckle_joint"]  # 夹爪的关节名称

        gripper_point = JointTrajectoryPoint()

        gripper_point.positions = [0.0 if open_gripper else 0.2]

        gripper_point.time_from_start = rospy.Duration(1.0)

        goal.trajectory.points.append(gripper_point)

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
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        if success:
            rospy.loginfo("Moved to home position.")
        else:
            rospy.logwarn("Failed to move to home position.")
        rospy.sleep(2)



if __name__ == "__main__":
    try:
        GraspNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass








