import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, LaserScan
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from packaging.version import parse

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
    
    

if parse(cv2.__version__) >= parse('4.7.0'):
    def local_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
        marker = np.array([[-marker_size /2, marker_size / 2, 0],
                           [marker_size /2, marker_size / 2, 0],
                           [marker_size /2, -marker_size / 2, 0],
                           [-marker_size /2, -marker_size / 2, 0]],
                           dtype = np.float32)
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(marker, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash


class ArucoTarget(Node):
    _DICTS = {
        "4x4_100" : cv2.aruco.DICT_4X4_100,
        "4x4_1000" : cv2.aruco.DICT_4X4_1000,
        "4x4_250" : cv2.aruco.DICT_4X4_250,
        "4x4_50" : cv2.aruco.DICT_4X4_50,
        "5x5_100" : cv2.aruco.DICT_5X5_100,
        "5x5_1000" : cv2.aruco.DICT_5X5_1000,
        "5x5_250" : cv2.aruco.DICT_5X5_250,
        "5x5_50" : cv2.aruco.DICT_5X5_50,
        "6x6_100" : cv2.aruco.DICT_6X6_100,
        "6x6_1000" : cv2.aruco.DICT_6X6_1000,
        "6x6_250" : cv2.aruco.DICT_6X6_250,
        "6x6_50" : cv2.aruco.DICT_6X6_50,
        "7x7_100" : cv2.aruco.DICT_7X7_100,
        "7x7_1000" : cv2.aruco.DICT_7X7_1000,
        "7x7_250": cv2.aruco.DICT_7X7_250,
        "7x7_50": cv2.aruco.DICT_7X7_50,
        "apriltag_16h5" : cv2.aruco.DICT_APRILTAG_16H5,
        "apriltag_25h9" : cv2.aruco.DICT_APRILTAG_25H9,
        "apriltag_36h10" : cv2.aruco.DICT_APRILTAG_36H10,
        "apriltag_36h11" : cv2.aruco.DICT_APRILTAG_36H11,
        "aruco_original" : cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    def __init__(self, tag_set="apriltag_36h10", target_width=0.30):
        super().__init__('aruco_targetv2')
        self.get_logger().info(f'{self.get_name()} created')

        self.declare_parameter('image', "/mycamera/image_raw")
        self.declare_parameter('info', "/mycamera/camera_info")

        self._image_topic = self.get_parameter('image').get_parameter_value().string_value
        self._info_topic = self.get_parameter('info').get_parameter_value().string_value

        self.create_subscription(Image, self._image_topic, self._image_callback, 1)
        self.create_subscription(CameraInfo, self._info_topic, self._info_callback, 1)
        
        # --- FIX: INITIALIZE VARIABLES FIRST ---
        # We define these BEFORE creating subscriptions so they exist when callbacks fire
        self.target_locked = False     
        self.obstacle_detected = False 
        #self.avoid_turn = 0.0 # Use to store +0.5 (Left) or -0.5 (Right)
        
        self.avoid_bias = 0.0
        
        # ---------------------------------------
        
        # --- NEW: LiDAR Subscription ---
        # We subscribe to the scan topic to detect obstacles
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 1)
        self._safety_stop = False  # Flag to trigger emergency stop
        self._lidar_threshold = 0.8 # Meters. Stop if anything is closer than this.
        # -------------------------------
        
        # Create publisher for velocity commands
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._bridge = CvBridge()

        dict = ArucoTarget._DICTS.get(tag_set.lower(), None)
        if dict is None:
            self.get_logger().error(f'ARUCO tag set {tag_set} not found')
        else:
            if parse(cv2.__version__) < parse('4.7.0'):
                self._aruco_dict = cv2.aruco.Dictionary_get(dict)
                self._aruco_param = cv2.aruco.DetectorParameters_create()
            else:
                self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict)
                self._aruco_param = cv2.aruco.DetectorParameters()
                self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_param)
            self._target_width = target_width
            self._image = None
            self._cameraMatrix = None
            self.get_logger().info(f"using dictionary {tag_set}")
            
        # --- NEW: LiDAR Callback ---
    def _scan_callback(self, msg):
        ranges = msg.ranges
        valid_entries = [(r, i) for i, r in enumerate(ranges) if np.isfinite(r) and r > 0.25]
        
        if len(valid_entries) > 0:
            closest_dist, closest_index = min(valid_entries, key=lambda x: x[0])
            
            # 1. Define the "Center" width (e.g., middle 10% of the scan)
            total_indices = len(ranges)
            mid_point = total_indices / 2
            center_width = total_indices * 0.1 # 10% wide cone
            
            # Calculate the start and end of the center cone
            center_start = mid_point - (center_width / 2)
            center_end = mid_point + (center_width / 2)

            safety_limit = 0.3 
            
            if closest_dist < safety_limit:
                self.obstacle_detected = True
                
                # 2. Check: Is it Dead Center?
                if center_start < closest_index < center_end:
                    # Object is directly in front -> STOP
                    #self.avoid_turn = 0.0
                    self.avoid_bias = 0.0
                    self.stop_completely = True # New flag to signal a full stop
                    self.get_logger().warn(f"BLOCKED FRONT ({closest_dist:.2f}m) - STOPPING")
                
                # 3. Check Left vs Right (for glancing blows)
                elif closest_index < mid_point:
                    # Object on Left -> Turn Right (Negative)
                    #self.avoid_turn = 0.05
                    self.avoid_bias = -0.3
                    self.stop_completely = False
                    self.get_logger().warn("Avoiding Right Object")
                else:
                    # Object on Right -> Turn Left (Positive)
                    #self.avoid_turn = -0.05
                    self.avoid_bias = 0.3
                    self.stop_completely = False
                    self.get_logger().warn("Avoiding Left Object")
            else:
                self.obstacle_detected = False
                self.stop_completely = False
                self.avoid_bias = 0.0
    # ---------------------------
            
            

    def _info_callback(self, msg):
        if msg.distortion_model != "plumb_bob":
            self.get_logger().error(f"We can only deal with plumb_bob distortion {msg.distortion_model}")
        self._distortion = np.reshape(msg.d, (1,5))
        self._cameraMatrix = np.reshape(msg.k, (3,3))

    def _image_callback(self, msg):
        self._image = self._bridge.imgmsg_to_cv2(msg, "bgr8") 
        twist = Twist()

        # 1. EMERGENCY STOP (Dead Center Block)
        if self.obstacle_detected and self.stop_completely:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self._cmd_pub.publish(twist)
            cv2.imshow('window', self._image)
            cv2.waitKey(3)
            return

        # 2. ArUco Detection
        grey = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        
        if parse(cv2.__version__) < parse('4.7.0'):
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(grey, self._aruco_dict)
        else:
            corners, ids, rejectedImgPoints = self._aruco_detector.detectMarkers(grey)
        
        # Draw the square around the marker (this is safe even if ids is None)
        frame = cv2.aruco.drawDetectedMarkers(self._image, corners, ids)
        
        # 3. DECISION LOGIC
        if ids is None:
            # --- CASE A: NO TARGET SEEN ---
            if self.obstacle_detected:
                # Blind Avoidance
                twist.linear.x = 0.05
                twist.angular.z = -1.0 * self.avoid_bias 
                self.get_logger().info("No target found #1")
            else:
                # No Target, No Obstacle -> Stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info("No target found #2")
                
            # Show image immediately and return
            cv2.imshow('window', frame)
            cv2.waitKey(3)
            self._cmd_pub.publish(twist)
            return  # Stop here, do not try to access rvec/tvec
                
        else:
            # --- CASE B: TARGET FOUND ---
            if parse(cv2.__version__) < parse('4.7.0'):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self._target_width, self._cameraMatrix, self._distortion)
            else:
                rvec, tvec, _ = local_estimatePoseSingleMarkers(corners, self._target_width, self._cameraMatrix, self._distortion)
        
            # Tracking Logic
            t = tvec[0].flatten() 
            z_dist = float(t[2])
            x_offset = float(t[0]) 
            real_angle = math.atan2(x_offset, z_dist)   
        
            self.get_logger().info(f"Target: {z_dist:.2f}m away, Offset: {x_offset:.2f}m")

            steer_error = real_angle + self.avoid_bias
            
            k_p = 1.5 
            twist.angular.z = -k_p * steer_error

            if z_dist > 1.0:
               twist.linear.x = 0.2
            else:
               twist.linear.x = 0.0

            # Draw axes for visualization (ONLY safely done here inside the else block)
            for r, t in zip(rvec, tvec):
                if parse(cv2.__version__) < parse('4.7.0'):
                    frame = cv2.aruco.drawAxis(frame, self._cameraMatrix, self._distortion, r, t, self._target_width)
                else:
                    frame = cv2.drawFrameAxes(frame, self._cameraMatrix, self._distortion, r, t, self._target_width)
  
            cv2.imshow('window', frame)
            cv2.waitKey(3)
            self._cmd_pub.publish(twist)
        # --------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ArucoTarget()
#    node = ArucoTarget(tag_set="4x4_50", target_width=0.048)
    try:
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
