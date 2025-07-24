# Import modules
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

# Import messages
from sensor_msgs.msg import Image


class VirtualCamera(Node):
    def __init__(self):
        super().__init__("virtual_camera")
        self.get_logger().info("Virtual camera started.")

        # Attributes
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('dataset_path', '/home/visione/Projects/VO/src/dataset/sat_light_360')
        self.dataset_path = self.get_parameter('dataset_path').get_parameter_value().string_value

        self.declare_parameter('image_format', '.png')
        self.image_format = self.get_parameter('image_format').get_parameter_value().string_value

        # Load images paths
        samples_paths = os.listdir(self.dataset_path)
        self.num_samples = len(samples_paths)
        self.sample_idx = 0

        # Publisher
        self.frame_publisher = self.create_publisher(Image, 'frame', 10)
        self.timer = self.create_timer(0.1, self.publish_frame)

    # Collect and publish frame
    def publish_frame(self):
        
        # If the sample idx is greater than the dataset size, restart.
        if self.sample_idx >= self.num_samples:
            self.sample_idx = 0

        # Publish the idx-th frame of the target dataset
        frame_cv = cv2.imread(os.path.join(self.dataset_path, str(self.sample_idx).zfill(3) + self.image_format))
        frame_np = np.asanyarray(frame_cv)
        frame_msg = self.bridge.cv2_to_imgmsg(frame_np, 'bgr8')
        frame_msg.header.stamp = self.get_clock().now().to_msg()
        frame_msg.header.frame_id = 'virtual_frame'

        # Increase the sample idx
        self.frame_publisher.publish(frame_msg)
        self.sample_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = VirtualCamera()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        rclpy.get_logger().info("Node manually interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
