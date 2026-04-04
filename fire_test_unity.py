import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO # YOLOv8 또는 v11

class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')
        self.subscription = self.create_subscription(
            Image, '/cctv/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.model = YOLO('/home/junsu/engineering_contest/Engineering_industry_contest_2026_computer_visoin/best.pt') # 학습시킨 화재 감지 모델 경로

    def listener_callback(self, data):
        # 1. ROS 이미지를 OpenCV로 변환
        current_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        # ----------------------------------------------------
        # 추가: 상하 반전 해결 (0은 상하 반전, 1은 좌우 반전)
        current_frame = cv2.flip(current_frame, 0) 
        # ----------------------------------------------------

        # 2. YOLO 추론
        results = self.model(current_frame)
        
        # 3. 결과 시각화
        annotated_frame = results[0].plot()
        cv2.imshow("Unity CCTV - YOLO Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# ----------------------------
# 이 두 줄이 반드시 있어야 합니다!
# ----------------------------
if __name__ == '__main__':
    main()