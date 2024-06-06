import argparse
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime

lane_colors = [(68,65,249),(44,114,243),(30,150,248),(74,132,249),(79,199,249),(109,190,144),(142, 144, 77),(161, 125, 39)]
log_space = np.logspace(0,2, 50, base=1/10, endpoint=True)

class LSTR():
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession("lstr_360x640.onnx",  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.rgb_input_name = model_inputs[0].name
        self.mask_input_name = model_inputs[1].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.mask_tensor = np.zeros((1, 1, self.input_height, self.input_width), dtype=np.float32)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1).T

    def detect(self, frame):
        img_height, img_width = frame.shape[:2]
        img = cv2.resize(frame, (self.input_width, self.input_height))

        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std

        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)

        #推理
        outputs = self.session.run(self.output_names, {self.rgb_input_name: input_tensor, self.mask_input_name: self.mask_tensor})

        # Get the output logits and curves
        pred_logits = outputs[0]
        pred_curves = outputs[1]

        # Filter good lanes based on the probability
        prob = self.softmax(pred_logits)
        good_detections = np.where(np.argmax(prob, axis=-1) == 1)
        pred_logits = pred_logits[good_detections]
        pred_curves = pred_curves[good_detections]

        lanes = []
        for lane_data in pred_curves:
            bounds = lane_data[:2]
            k_2, f_2, m_2, n_1, b_2, b_3 = lane_data[2:]

            # Calculate the points for the lane
            # Note: the logspace is used for a visual effect, np.linspace would also work as in the original repository
            y_norm = bounds[0] + log_space * (bounds[1] - bounds[0])
            x_norm = (k_2 / (y_norm - f_2) ** 2 + m_2 / (y_norm - f_2) + n_1 + b_2 * y_norm - b_3)
            lane_points = np.vstack((x_norm * img_width, y_norm * img_height)).astype(int)

            lanes.append(lane_points)

        return lanes, good_detections[1]
    
    def draw_lanes(self, input_img, detected_lanes, good_lanes):
        # Write the detected line points in the image
        visualization_img = input_img.copy()

        # Draw a mask for the current lane
        right_lane = np.where(good_lanes == 0)[0]
        left_lane = np.where(good_lanes == 5)[0]

        if (len(left_lane) and len(right_lane)):
            lane_segment_img = visualization_img.copy()

            points = np.vstack((detected_lanes[left_lane[0]].T, np.flipud(detected_lanes[right_lane[0]].T)))
            cv2.fillConvexPoly(lane_segment_img, points, color=(0, 191, 255))
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

        for lane_num, lane_points in zip(good_lanes, detected_lanes):
            if lane_num in [0, 5]:
                for lane_point in lane_points.T:
                    cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

        return visualization_img
    
    def count_difference(self,input_img, detected_lanes, good_lanes):
        visualization_img = input_img.copy()
        
        left_lane = np.where(good_lanes == 5)[0]
        right_lane = np.where(good_lanes == 0)[0]
        
        if len(left_lane) == 0 or len(right_lane) == 0:
            print("Line drop out")
            return visualization_img
        
        left_lane_lowest_point = detected_lanes[right_lane[0]].T[0]
        right_lane_lowest_point = detected_lanes[left_lane[0]].T[0]
        mid_lane_lowest_point =  ((left_lane_lowest_point[0] + right_lane_lowest_point[0]) // 2, (left_lane_lowest_point[1] + right_lane_lowest_point[1]) // 2)

        # cv2.circle(visualization_img, (left_lane_lowest_point[0], left_lane_lowest_point[1]), 3, (0, 255, 0), -1)
        # cv2.circle(visualization_img, (right_lane_lowest_point[0], right_lane_lowest_point[1]), 3, (0, 0, 255), -1)

        left_lane_points = detected_lanes[left_lane[0]].T
        right_lane_points = detected_lanes[right_lane[0]].T
        mid_points = []
        start_line = 0
        end_line = 10
        line_num = start_line
        x_diff_sum = 0
        for left_point, right_point in zip(left_lane_points, right_lane_points):
            if start_line<=line_num<=end_line:
                line_num += 1
                mid_point = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
                x_diff = mid_point[0] - mid_lane_lowest_point[0]
                x_diff_sum += x_diff
                mid_points.append(mid_point)
                cv2.circle(visualization_img, mid_point, 3, (0, 255, 0), -1)
        x_diff_avg = x_diff_sum / (end_line - start_line)
        cv2.putText(visualization_img, f"X diff: {x_diff_avg:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        """
        #拟合曲线
        # 将中点坐标分成 x 和 y 两个数组
        mid_points = np.array(mid_points)
        mid_x = mid_points[:, 0]
        mid_y = mid_points[:, 1]

        # 使用多项式拟合中点曲线
        poly_coeff = np.polyfit(mid_y, mid_x, 2)
        poly_func = np.poly1d(poly_coeff)

        # 生成拟合曲线的 y 值
        fit_y = np.linspace(mid_y.min(), mid_y.max(), num=100)
        fit_x = poly_func(fit_y)

        # 将拟合曲线绘制到图像上
        for x, y in zip(fit_x.astype(int), fit_y.astype(int)):
            cv2.circle(visualization_img, (x, y), 3, (0, 0, 255), -1)
        """

        return visualization_img
                    



if __name__ == '__main__':
    net = LSTR()
    cap = cv2.VideoCapture('videos/test2.mp4')
    while True:
        start = datetime.now()
        ret, frame = cap.read()
        #print(frame.shape)
        frame = cv2.resize(frame, (640,480))
        if not ret:
            print("Can't receive frame")
        detected_lanes, lane_ids = net.detect(frame)
        dstimg = net.draw_lanes(frame, detected_lanes, lane_ids)
        dstimg = net.count_difference(dstimg, detected_lanes, lane_ids)
        end = datetime.now()
        time = end - start
        time = int(1 / time.total_seconds())
        cv2.putText(dstimg, f"{time} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("image", dstimg)
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()