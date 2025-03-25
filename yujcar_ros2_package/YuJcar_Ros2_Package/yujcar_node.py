import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ai_msgs.msg import PerceptionTargets
from origincar_msg.msg import Sign
import cv2
import numpy as np
import threading
from pyzbar.pyzbar import decode
import time

class AdaptivePID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class YuJcarNode(Node):
    def __init__(self):
        super().__init__('yujcar_node')
        self.get_logger().info("Start!")
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, 'image_raw1', self.image_callback, 10)
        self.detection_sub = self.create_subscription(PerceptionTargets, '/hobot_dnn_detection', self.detection_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sign_switch_pub = self.create_publisher(Sign, '/sign_switch', 10)
        self.sign_sub = self.create_subscription(Int32, '/sign4return', self.sign_callback, 10)
        self.twist = Twist()
        self.processing_lock = threading.Lock()
        self._shutdown_requested = False
        self.allow_follow_line = True
        self.obstacle_detected = False
        self.phase = 1
        self.pid = AdaptivePID(0.5, 0.01, 0.2)
        self.last_collision_time = time.time()

    def follow_line_phase_one(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 50, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        h, w, d = cv_image.shape
        rois = [
            (int(h * 0.5), int(h * 0.55)),
            (int(h * 0.55), int(h * 0.6)),
            (int(h * 0.6), int(h * 0.65)),
            (int(h * 0.65), int(h * 0.7)),
            (int(h * 0.7), int(h * 0.75)),
        ]
        cx_list = []
        weights = []
        for i, (start, end) in enumerate(rois):
            mask_roi = mask[start:end, 0:w]
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cx_list.append(cx)
                        weights.append(i + 1)
                        cv2.circle(cv_image, (cx, int((start + end) / 2)), 10, (0, 0, 255), -1)
        if cx_list:
            avg_cx = int(np.average(cx_list, weights=weights))
            err = avg_cx - w / 2
            dt = 0.1
            output = self.pid.update(err, dt)
            self.twist.angular.z = -output / 480
            self.cmd_vel_pub.publish(self.twist)
            time.sleep(0.1)
            self.twist.linear.x = 0.5
            self.cmd_vel_pub.publish(self.twist)

    def follow_line_phase_two(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 50, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        h, w, d = cv_image.shape
        rois = [
            (int(h * 0.5), int(h * 0.55)),
            (int(h * 0.55), int(h * 0.6)),
            (int(h * 0.6), int(h * 0.65)),
            (int(h * 0.65), int(h * 0.7)),
            (int(h * 0.7), int(h * 0.75)),
        ]
        cx_list = []
        weights = []
        for i, (start, end) in enumerate(rois):
            mask_roi = mask[start:end, 0:w]
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cx_list.append(cx)
                        weights.append(i + 1)
                        cv2.circle(cv_image, (cx, int((start + end) / 2)), 10, (0, 0, 255), -1)
        if cx_list:
            avg_cx = int(np.average(cx_list, weights=weights))
            err = avg_cx - w / 2
            dt = 0.1
            output = self.pid.update(err, dt)
            self.twist.angular.z = -output / 240
            self.cmd_vel_pub.publish(self.twist)
            time.sleep(0.1)
            self.twist.linear.x = 0.45
            self.cmd_vel_pub.publish(self.twist)

    def detection_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_collision_time < 2.0:
            return

        if not self.allow_follow_line:
            return

        for target in msg.targets:
            if target.rois[0].type == 'barrier':
                xmin = target.rois[0].rect.x_offset
                ymin = target.rois[0].rect.y_offset
                xmax = target.rois[0].rect.width
                ymax = target.rois[0].rect.height
                box_area = xmax * ymax

                if box_area >= 23352 and 138 <= xmin <= 328:
                    self.stop_robot()
                    time.sleep(0.5)
                    self.obstacle_direction = 'left' if xmin < 236 else 'right'
                    self.last_collision_time = current_time

                    if self.obstacle_direction == 'left':
                        self.twist.angular.z = 1.0
                        self.twist.linear.x = 0.6
                        self.cmd_vel_pub.publish(self.twist)
                        time.sleep(2)
                        self.stop_robot()
                        time.sleep(0.5)
                        self.twist.angular.z = -1.0
                        self.twist.linear.x = -0.6
                        self.cmd_vel_pub.publish(self.twist)
                        time.sleep(2)
                    else:
                        self.twist.angular.z = -0.8
                        self.twist.linear.x = 0.6
                        self.cmd_vel_pub.publish(self.twist)
                        time.sleep(2)
                        self.stop_robot()
                        time.sleep(0.5)
                        self.twist.angular.z = 0.8
                        self.twist.linear.x = -0.6
                        self.cmd_vel_pub.publish(self.twist)
                        time.sleep(2)

                    self.stop_robot()
                    self.allow_follow_line = True
                    return

    def process_qr_code(self, cv_image):
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray_image)
        if decoded_objects:
            for obj in decoded_objects:
                qr_data = obj.data.decode("utf-8")
                self.get_logger().info(f"Detected QR Code: {qr_data}")
                self.publish_signal(qr_data)
                self.stop_robot()
                self.allow_follow_line = False
                self.obstacle_detected = False
                break

    def publish_signal(self, qr_data):
        msg = Sign()
        if qr_data == "ClockWise":
            msg.sign_data = 3
        elif qr_data == "AntiClockWise":
            msg.sign_data = 4
        else:
            msg.sign_data = 2
        self.sign_switch_pub.publish(msg)
        self.get_logger().info(f'Published signal: {msg.sign_data}')

    def image_callback(self, msg):
        if self._shutdown_requested:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_qr_code(cv_image)
        if self.allow_follow_line:
            if self.phase == 1:
                self.follow_line_phase_one(cv_image)
            else:
                self.follow_line_phase_two(cv_image)

    def sign_callback(self, msg):
        if msg.data == -1:
            self.get_logger().info("Received shutdown signal")
            self.stop_robot()
            self._shutdown_requested = True
        elif msg.data == 6:
            self.get_logger().info("Received signal to enable phase 2 line following")
            self.phase = 2
            self.allow_follow_line = True
            self.obstacle_detected = True

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = YuJcarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()