import math
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    NUM_SAMPLE_FRAMES,
    PERSON_CONF_THRESHOLD,
    HUE_HIST_BINS
)

class VideoFeatureExtractor:
    """
    Trích xuất tự động 8 nhóm đặc trưng từ video (theo tài liệu 'Đặc trưng nhận diện video'):
    1. Số lượng chủ thể và Tương tác (Person Count & Interaction IoU)
    2. Quy mô và cự ly ống kính (Camera Scale / Shot Type)
    3. Tư thế cơ thể qua góc khớp xương (Body Pose & Joint Angles)
    4. Phạm vi dịch chuyển trọng tâm (Motion Scope - In-place vs Translational)
    5. Vận tốc và biên độ cử động tứ chi (Limb Velocity & Motion Amplitude)
    6. Sắc độ màu môi trường chủ đạo (Dominant Hue & Environment Palette)
    7. Độ ổn định phông nền (Background Stability)
    8. Độ tương phản người và phông nền (Foreground vs Background Contrast)
    """

    def __init__(self, model_path=YOLO_MODEL_PATH):
        print(f"[Init] Đang tải mô hình YOLO Pose từ: {model_path} ...")
        self.model = YOLO(model_path)

    @staticmethod
    def calculate_angle(pointA, pointB, pointC):
        """
        Tính góc tại đỉnh pointB tạo bởi vector BA và BC (độ: 0° đến 180°).
        """
        if pointA is None or pointB is None or pointC is None:
            return 165.0  # Mặc định tư thế đứng thẳng tự nhiên

        ba = np.array(pointA) - np.array(pointB)
        bc = np.array(pointC) - np.array(pointB)

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 165.0

        cosine_val = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_val = np.clip(cosine_val, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_val)))

    @staticmethod
    def calculate_iou(boxA, boxB):
        """Tính chỉ số Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        union_area = areaA + areaB - inter_area

        if union_area < 1e-6:
            return 0.0
        return float(inter_area / union_area)

    def sample_frames(self, video_path):
        """Đọc và lấy mẫu đều NUM_SAMPLE_FRAMES khung hình từ video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            return None, None

        sample_count = min(NUM_SAMPLE_FRAMES, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frames = []
        frame_timestamps = []
        current_idx = 0
        target_idx_set = set(sample_indices)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_idx_set:
                frames.append(frame)
                frame_timestamps.append(current_idx / fps)
            current_idx += 1

        cap.release()

        metadata = {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "width": frames[0].shape[1] if frames else 0,
            "height": frames[0].shape[0] if frames else 0
        }
        return frames, frame_timestamps, metadata

    def extract(self, video_path):
        """
        Hàm chính: Trích xuất toàn bộ 7 nhóm đặc trưng từ một file video.
        """
        sampled = self.sample_frames(video_path)
        if sampled[0] is None or len(sampled[0]) == 0:
            return None

        frames, timestamps, metadata = sampled
        frame_count = len(frames)
        frame_w, frame_h = metadata["width"], metadata["height"]
        frame_area = frame_w * frame_h if (frame_w > 0 and frame_h > 0) else 1.0

        # Chạy YOLO Pose theo batch để tối ưu tốc độ
        results = self.model(frames, verbose=False)

        person_counts = []
        max_ious = []
        scale_ratios = []
        aspect_ratios = []
        hip_angles = []
        knee_angles = []
        centroids = []
        limb_positions = []  # Lưu tọa độ 4 khớp: [lw, rw, la, ra]
        hue_histograms = []
        contrast_scores = []
        bg_saturations = []
        bg_values = []
        background_masks = []

        for i, (frame, res) in enumerate(zip(frames, results)):
            # 1. Phát hiện người & Bounding boxes
            boxes = res.boxes
            person_boxes = []
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    if cls_id == 0 and conf >= PERSON_CONF_THRESHOLD:
                        coords = b.xyxy[0].cpu().numpy().tolist()
                        person_boxes.append(coords)

            count = len(person_boxes)
            person_counts.append(count)

            # Tính IoU tương tác giữa 2 người nếu có từ 2 người trở lên
            if count >= 2:
                iou = self.calculate_iou(person_boxes[0], person_boxes[1])
                max_ious.append(iou)
            else:
                max_ious.append(0.0)

            # Phân tích chủ thể chính (người có BBox lớn nhất)
            main_box = None
            if count > 0:
                # Chọn box có diện tích lớn nhất
                person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                main_box = person_boxes[0]

                bw = main_box[2] - main_box[0]
                bh = main_box[3] - main_box[1]
                scale = (bw * bh) / frame_area
                scale_ratios.append(scale)
                aspect_ratios.append(bh / max(bw, 1.0))
                centroids.append(((main_box[0] + main_box[2]) / 2.0, (main_box[1] + main_box[3]) / 2.0))
            else:
                scale_ratios.append(0.3)
                aspect_ratios.append(1.5)

            # 2. Keypoints & Góc khớp
            keypoints = res.keypoints
            current_limbs = [None, None, None, None]  # [L_wrist, R_wrist, L_ankle, R_ankle]

            if keypoints is not None and len(keypoints) > 0 and count > 0:
                kp_data = keypoints.data[0].cpu().numpy()  # [17, 3] (x, y, conf)

                # Các điểm mấu chốt quan trọng:
                # 5: L_shoulder, 6: R_shoulder
                # 9: L_wrist, 10: R_wrist
                # 11: L_hip, 12: R_hip
                # 13: L_knee, 14: R_knee
                # 15: L_ankle, 16: R_ankle

                def get_pt(idx):
                    if idx < len(kp_data) and kp_data[idx][2] >= 0.25:
                        return (float(kp_data[idx][0]), float(kp_data[idx][1]))
                    return None

                l_sh, r_sh = get_pt(5), get_pt(6)
                l_hip, r_hip = get_pt(11), get_pt(12)
                l_knee, r_knee = get_pt(13), get_pt(14)
                l_ank, r_ank = get_pt(15), get_pt(16)

                current_limbs = [get_pt(9), get_pt(10), l_ank, r_ank]

                # Tính góc hông (Vai - Hông - Gối)
                angle_h_left = self.calculate_angle(l_sh, l_hip, l_knee) if (l_sh and l_hip and l_knee) else None
                angle_h_right = self.calculate_angle(r_sh, r_hip, r_knee) if (r_sh and r_hip and r_knee) else None
                valid_hip = [a for a in [angle_h_left, angle_h_right] if a is not None]
                hip_angles.append(float(np.mean(valid_hip)) if valid_hip else 165.0)

                # Tính góc gối (Hông - Gối - Cổ chân)
                angle_k_left = self.calculate_angle(l_hip, l_knee, l_ank) if (l_hip and l_knee and l_ank) else None
                angle_k_right = self.calculate_angle(r_hip, r_knee, r_ank) if (r_hip and r_knee and r_ank) else None
                valid_knee = [a for a in [angle_k_left, angle_k_right] if a is not None]
                knee_angles.append(float(np.mean(valid_knee)) if valid_knee else 170.0)
            else:
                hip_angles.append(165.0)
                knee_angles.append(170.0)

            limb_positions.append(current_limbs)

            # 3. Phân tích màu sắc bối cảnh (HSV) & Tương phản
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [HUE_HIST_BINS], [0, 180])
            hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1).flatten().tolist()
            hue_histograms.append(hist)

            # Mặt nạ nền (loại trừ vùng có người)
            bg_mask = np.ones((frame_h, frame_w), dtype=np.uint8) * 255
            fg_color = [128, 128, 128]
            if main_box is not None:
                x1, y1 = max(0, int(main_box[0])), max(0, int(main_box[1]))
                x2, y2 = min(frame_w, int(main_box[2])), min(frame_h, int(main_box[3]))
                if x2 > x1 and y2 > y1:
                    bg_mask[y1:y2, x1:x2] = 0
                    fg_crop = frame[y1:y2, x1:x2]
                    fg_color = np.mean(fg_crop, axis=(0, 1))

            background_masks.append((frame, bg_mask))
            bg_pixels = frame[bg_mask == 255]
            if len(bg_pixels) > 0:
                bg_color = np.mean(bg_pixels, axis=0)
                contrast = float(np.linalg.norm(np.array(fg_color) - np.array(bg_color)))
                contrast_scores.append(contrast)

                # Lấy độ bão hòa (S) và độ sáng (V) của phông nền
                bg_hsv = hsv[bg_mask == 255]
                bg_saturations.append(float(np.mean(bg_hsv[:, 1])))
                bg_values.append(float(np.mean(bg_hsv[:, 2])))
            else:
                contrast_scores.append(50.0)
                bg_saturations.append(100.0)
                bg_values.append(100.0)

        # 4. Tính toán các đặc trưng thời gian (Temporal Differences)
        limb_velocities = []
        bg_diff_scores = []

        for i in range(frame_count - 1):
            dt = max(timestamps[i + 1] - timestamps[i], 1e-4)

            # Vận tốc cử động tứ chi
            disp_list = []
            for k in range(4):
                p1 = limb_positions[i][k]
                p2 = limb_positions[i + 1][k]
                if p1 is not None and p2 is not None:
                    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    disp_list.append(dist / dt)
            if disp_list:
                limb_velocities.append(float(np.mean(disp_list)))

            # Độ biến động phông nền (Background Difference)
            f1, m1 = background_masks[i]
            f2, m2 = background_masks[i + 1]
            diff = cv2.absdiff(f1, f2)
            common_bg = cv2.bitwise_and(m1, m2)
            bg_diff = diff[common_bg == 255]
            if len(bg_diff) > 0:
                bg_diff_scores.append(float(np.mean(bg_diff)))

        # 5. Tổng hợp các giá trị đại diện (Aggregations)
        avg_person_count = max(1, int(round(np.mean(person_counts))))
        avg_iou = float(np.max(max_ious)) if max_ious else 0.0
        avg_scale = float(np.mean(scale_ratios)) if scale_ratios else 0.35
        avg_aspect_ratio = float(np.mean(aspect_ratios)) if aspect_ratios else 1.5
        avg_hip_angle = float(np.mean(hip_angles)) if hip_angles else 165.0
        avg_knee_angle = float(np.mean(knee_angles)) if knee_angles else 170.0

        # Độ lệch chuẩn trọng tâm
        if len(centroids) >= 2:
            centroids_arr = np.array(centroids)
            centroid_std = float(np.sqrt(np.mean(np.var(centroids_arr, axis=0))))
        else:
            centroid_std = 0.0

        avg_limb_velocity = float(np.mean(limb_velocities)) if limb_velocities else 20.0
        avg_bg_diff = float(np.mean(bg_diff_scores)) if bg_diff_scores else 5.0
        avg_contrast = float(np.mean(contrast_scores)) if contrast_scores else 50.0
        avg_bg_sat = float(np.mean(bg_saturations)) if bg_saturations else 100.0
        avg_bg_val = float(np.mean(bg_values)) if bg_values else 100.0

        # Histogram màu trung bình
        avg_hue_hist = np.mean(hue_histograms, axis=0).tolist()
        dominant_hue = float(np.argmax(avg_hue_hist) * (180.0 / HUE_HIST_BINS))

        # 6. Gán nhãn ngữ nghĩa (Semantic Labels) chuẩn theo tài liệu
        # Đặc trưng 1: Số lượng chủ thể và Tương tác
        if avg_person_count == 1:
            interaction_label = "Solo"
        elif avg_iou > 0.08:
            interaction_label = "Interacting"
        else:
            interaction_label = "Group"

        # Đặc trưng 2: Quy mô và cự ly ống kính
        if avg_scale > 0.45:
            scale_label = "Close-up"
        elif avg_scale >= 0.18:
            scale_label = "Medium"
        else:
            scale_label = "Long-shot"

        # Đặc trưng 3: Tư thế cơ thể qua góc khớp xương
        if avg_aspect_ratio < 0.90:
            pose_label = "Horizontal / Lying"
        elif avg_hip_angle < 125.0:
            pose_label = "Sitting / Bending"
        else:
            pose_label = "Standing"

        # Đặc trưng 4: Phạm vi dịch chuyển trọng tâm
        motion_scope_label = "In-place" if centroid_std < 25.0 else "Translational"

        # Đặc trưng 5: Vận tốc và biên độ cử động tứ chi
        if avg_limb_velocity < 50.0:
            amplitude_label = "Fine / Local"
        elif avg_limb_velocity <= 140.0:
            amplitude_label = "Moderate"
        else:
            amplitude_label = "High-energy"

        # Đặc trưng 6: Sắc độ màu môi trường chủ đạo
        if avg_bg_sat < 40.0 and avg_bg_val > 110.0:
            environment_label = "Ice / Snow"
        elif 35.0 <= dominant_hue <= 85.0:
            environment_label = "Green / Turf Field"
        elif 90.0 <= dominant_hue <= 130.0:
            environment_label = "Aquatic / Water"
        else:
            environment_label = "Indoor / Warm Tone"

        # Đặc trưng 7: Độ ổn định phông nền
        bg_stability_label = "Static Background" if avg_bg_diff < 15.0 else "Dynamic Background"

        # Đặc trưng 8: Độ tương phản người và phông nền
        contrast_label = "High-contrast" if avg_contrast >= 55.0 else "Low-contrast"

        # 7. Xây dựng Vector đặc trưng tổng hợp chuẩn hóa (26 chiều)
        feature_vector = [
            float(np.clip(avg_person_count / 4.0, 0.0, 1.0)),
            float(np.clip(avg_iou, 0.0, 1.0)),
            float(np.clip(avg_scale, 0.0, 1.0)),
            float(np.clip(avg_aspect_ratio / 3.0, 0.0, 1.0)),
            float(np.clip(avg_hip_angle / 180.0, 0.0, 1.0)),
            float(np.clip(avg_knee_angle / 180.0, 0.0, 1.0)),
            float(np.clip(centroid_std / 100.0, 0.0, 1.0)),
            float(np.clip(avg_limb_velocity / 300.0, 0.0, 1.0)),
            float(np.clip(avg_bg_diff / 50.0, 0.0, 1.0)),
            float(np.clip(avg_contrast / 120.0, 0.0, 1.0))
        ] + avg_hue_hist

        return {
            "fps": metadata["fps"],
            "total_frames": metadata["total_frames"],
            "duration_sec": metadata["duration_sec"],
            "person_count": avg_person_count,
            "interaction_iou": round(avg_iou, 4),
            "interaction_label": interaction_label,
            "scale_ratio": round(avg_scale, 4),
            "scale_label": scale_label,
            "bbox_aspect_ratio": round(avg_aspect_ratio, 4),
            "hip_joint_angle": round(avg_hip_angle, 2),
            "knee_joint_angle": round(avg_knee_angle, 2),
            "pose_label": pose_label,
            "centroid_std": round(centroid_std, 2),
            "motion_scope_label": motion_scope_label,
            "limb_velocity": round(avg_limb_velocity, 2),
            "amplitude_label": amplitude_label,
            "dominant_hue": round(dominant_hue, 2),
            "hue_histogram": [round(x, 4) for x in avg_hue_hist],
            "environment_label": environment_label,
            "bg_diff_score": round(avg_bg_diff, 2),
            "bg_stability_label": bg_stability_label,
            "contrast_score": round(avg_contrast, 2),
            "contrast_label": contrast_label,
            "feature_vector": [round(x, 5) for x in feature_vector]
        }
