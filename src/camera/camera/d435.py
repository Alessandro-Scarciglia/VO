# Import modules
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge

# Import messages
from sensor_msgs.msg import Image


class IntelD435(Node):
    def __init__(self):
        super().__init__('d435_camera')
        self.get_logger().info('Intel RealSense D435 camera started.')

        # Attributes
        self.pipeline = rs.pipeline()
        self.pipeline.start()
        self.bridge = CvBridge()

        # Publisher
        self.frame_publisher = self.create_publisher(Image, 'frame', 10)
        self.timer = self.create_timer(0.1, self.color_frame_publisher)
    
    # Collect and publish frame
    def color_frame_publisher(self):
        
        # Get frame
        frames_rs = self.pipeline.wait_for_frames()
        color_frame_rs = frames_rs.get_color_frame()
        color_frame_np = np.asanyarray(color_frame_rs.get_data())
        color_frame_msg = self.bridge.cv2_to_imgmsg(color_frame_np, 'rgb8')

        # Create the message
        color_frame_msg = self.bridge.cv2_to_imgmsg(color_frame_np, 'rgb8')
        color_frame_msg.header.stamp = self.get_clock().now().to_msg()
        color_frame_msg.header.frame_id = 'camera_frame'

        # Publish the message
        self.frame_publisher.publish(color_frame_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IntelD435()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node manually interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

