import os
import sys
import json
import sqlite3
import argparse
import numpy as np

from config import SQLITE_DB_PATH
from feature_extractor import VideoFeatureExtractor

def cosine_similarity(v1, v2):
    """Tính độ tương đồng Cosine giữa 2 vector đặc trưng"""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def search_similar_video(query_video_path, db_path=SQLITE_DB_PATH, top_k=5, filter_category=None):
    """
    Tìm kiếm video giống nhất với video đầu vào:
    1. Trích xuất đặc trưng của video truy vấn (query video).
    2. Truy vấn dữ liệu đặc trưng từ SQLite.
    3. Tính độ tương đồng Cosine.
    4. Trả về Top-K video tương đồng nhất kèm bảng so sánh.
    """
    if not os.path.exists(query_video_path):
        print(f"[Lỗi] File video truy vấn không tồn tại: {query_video_path}")
        return None

    if not os.path.exists(db_path):
        print(f"[Lỗi] File CSDL SQLite chưa được tạo: {db_path}")
        print("Vui lòng chạy 'python extract_to_sqlite.py' trước để trích xuất CSDL.")
        return None

    # Bước 1: Trích xuất đặc trưng của video truy vấn
    print(f"\n[1/3] Đang trích xuất 8 đặc trưng từ video truy vấn: {os.path.basename(query_video_path)} ...")
    extractor = VideoFeatureExtractor()
    query_feat = extractor.extract(query_video_path)
    if query_feat is None:
        print("[Lỗi] Không thể đọc hoặc trích xuất đặc trưng từ video đầu vào!")
        return None

    query_vec = query_feat["feature_vector"]

    # Bước 2: Tải dữ liệu vector từ SQLite
    print(f"[2/3] Đang tải đặc trưng các video từ CSDL SQLite ({db_path}) ...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    sql = """
        SELECT video_id, file_name, file_path, action_category,
               person_count, interaction_label, scale_label, scale_ratio,
               bbox_aspect_ratio, hip_joint_angle, knee_joint_angle, pose_label,
               centroid_std, motion_scope_label,
               limb_velocity, amplitude_label,
               dominant_hue, environment_label,
               bg_diff_score, bg_stability_label,
               contrast_score, contrast_label,
               feature_vector_json
        FROM video_features
    """
    params = []
    if filter_category:
        sql += " WHERE action_category = ?"
        params.append(filter_category)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[Thông báo] CSDL SQLite hiện chưa có video nào được lưu!")
        return None

    print(f"[3/3] Đang so khớp với {len(rows)} video trong CSDL ...")

    # Bước 3: Tính toán độ tương đồng Cosine
    results = []
    for r in rows:
        vid_id, file_name, file_path, category = r[0], r[1], r[2], r[3]
        vec_json = r[22]
        try:
            db_vec = json.loads(vec_json)
            score = cosine_similarity(query_vec, db_vec)

            # Bỏ qua chính video đó nếu so khớp trong cùng tập dữ liệu
            is_self = os.path.normcase(os.path.abspath(query_video_path)) == os.path.normcase(os.path.abspath(file_path))

            results.append({
                "video_id": vid_id,
                "file_name": file_name,
                "file_path": file_path,
                "action_category": category,
                "similarity_score": round(score * 100, 2),
                "is_self": is_self,
                "details": {
                    "person_count": r[4],
                    "interaction_label": r[5],
                    "scale_label": r[6],
                    "scale_ratio": r[7],
                    "bbox_aspect_ratio": r[8],
                    "hip_joint_angle": r[9],
                    "knee_joint_angle": r[10],
                    "pose_label": r[11],
                    "centroid_std": r[12],
                    "motion_scope_label": r[13],
                    "limb_velocity": r[14],
                    "amplitude_label": r[15],
                    "dominant_hue": r[16],
                    "environment_label": r[17],
                    "bg_diff_score": r[18],
                    "bg_stability_label": r[19],
                    "contrast_score": r[20],
                    "contrast_label": r[21]
                }
            })
        except Exception:
            continue

    # Sắp xếp theo điểm tương đồng giảm dần
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Hiển thị kết quả
    top_results = results[:top_k]

    print("\n" + "=" * 80)
    print(" KẾT QUẢ TÌM KIẾM VIDEO TƯƠNG ĐỒNG NHẤT (CONTENT-BASED VIDEO RETRIEVAL)")
    print("=" * 80)
    print(f"▶ Video đầu vào (Query): {os.path.basename(query_video_path)}")
    print(f"  • Nhãn dự đoán: {query_feat['pose_label']} | {query_feat['motion_scope_label']} | {query_feat['scale_label']} | {query_feat['interaction_label']} | {query_feat['environment_label']} | {query_feat['contrast_label']}")

    print("\n[TOP KẾT QUẢ TƯƠNG ĐỒNG NHẤT]")
    print(f"{'Hạng':<5} | {'Điểm (%)':<10} | {'Tên Video':<35} | {'Danh mục':<20}")
    print("-" * 80)
    for idx, item in enumerate(top_results, start=1):
        flag = " (Chính nó)" if item["is_self"] else ""
        print(f"#{idx:<4} | {item['similarity_score']:>6.2f}%    | {item['file_name']:<35} | {item['action_category'] + flag:<20}")

    # Bảng so sánh chi tiết giữa Query và Top 1
    if top_results:
        best = top_results[0]
        # Nếu top 1 là chính nó và có video top 2, hiển thị so sánh với video khác
        if best["is_self"] and len(top_results) > 1:
            best = top_results[1]

        print("\n" + "-" * 85)
        print(f" SO SÁNH 8 ĐẶC TRƯNG CHI TIẾT: Query Video VS Video Giống Nhất ({best['file_name']})")
        print("-" * 85)
        print(f"{'Đặc trưng':<34} | {'Video Query':<25} | {'Video Top 1':<25}")
        print("-" * 85)
        print(f"{'1. Số lượng chủ thể & Tương tác':<34} | {str(query_feat['person_count']) + ' (' + str(query_feat['interaction_label']) + ')':<25} | {str(best['details']['person_count']) + ' (' + str(best['details']['interaction_label']) + ')':<25}")
        print(f"{'2. Quy mô cự ly ống kính':<34} | {str(query_feat['scale_label']) + ' (' + str(query_feat['scale_ratio']) + ')':<25} | {str(best['details']['scale_label']) + ' (' + str(best['details']['scale_ratio']) + ')':<25}")
        print(f"{'3. Tư thế cơ thể (Pose)':<34} | {query_feat['pose_label']:<25} | {str(best['details']['pose_label']):<25}")
        print(f"{'   - Góc khớp hông (Hip)':<34} | {str(query_feat['hip_joint_angle']) + '°':<25} | {str(best['details']['hip_joint_angle']) + '°':<25}")
        print(f"{'   - Góc khớp gối (Knee)':<34} | {str(query_feat['knee_joint_angle']) + '°':<25} | {str(best['details']['knee_joint_angle']) + '°':<25}")
        print(f"{'   - Tỷ lệ khung bao H/W':<34} | {str(query_feat['bbox_aspect_ratio']):<25} | {str(best['details']['bbox_aspect_ratio']):<25}")
        print(f"{'4. Dịch chuyển trọng tâm':<34} | {query_feat['motion_scope_label']:<25} | {str(best['details']['motion_scope_label']):<25}")
        print(f"{'   - Độ phân tán tâm (std)':<34} | {str(query_feat['centroid_std']):<25} | {str(best['details']['centroid_std']):<25}")
        print(f"{'5. Vận tốc cử động tứ chi':<34} | {query_feat['amplitude_label']:<25} | {str(best['details']['amplitude_label']):<25}")
        print(f"{'   - Tốc độ chi (px/s)':<34} | {str(query_feat['limb_velocity']):<25} | {str(best['details']['limb_velocity']):<25}")
        print(f"{'6. Sắc độ màu môi trường':<34} | {query_feat['environment_label']:<25} | {str(best['details']['environment_label']):<25}")
        print(f"{'   - Sắc độ cực đại (Hue)':<34} | {str(query_feat['dominant_hue']) + '°':<25} | {str(best['details']['dominant_hue']) + '°':<25}")
        print(f"{'7. Độ ổn định phông nền':<34} | {query_feat['bg_stability_label']:<25} | {str(best['details']['bg_stability_label']):<25}")
        print(f"{'   - Điểm sai khác nền':<34} | {str(query_feat['bg_diff_score']):<25} | {str(best['details']['bg_diff_score']):<25}")
        print(f"{'8. Độ tương phản người - nền':<34} | {query_feat['contrast_label']:<25} | {str(best['details']['contrast_label']):<25}")
        print(f"{'   - Điểm tương phản':<34} | {str(query_feat['contrast_score']):<25} | {str(best['details']['contrast_score']):<25}")
        print("=" * 85)

    return top_results

def main():
    parser = argparse.ArgumentParser(description="Tìm kiếm video tương đồng nhất từ SQLite")
    parser.add_argument("--query", type=str, required=True, help="Đường dẫn đến file video cần tìm kiếm")
    parser.add_argument("--db", type=str, default=SQLITE_DB_PATH, help="Đường dẫn file SQLite database")
    parser.add_argument("--top_k", type=int, default=5, help="Số lượng video tương đồng hiển thị")
    args = parser.parse_args()

    search_similar_video(args.query, db_path=args.db, top_k=args.top_k)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Nhập đường dẫn video cần test tại đây khi chạy trực tiếp:
        test_video_path = r"D:\csdl_dpt\videos\query\CricketShot1\v_CricketShot_g01_c01.avi"
        if not os.path.exists(test_video_path):
            # Fallback lấy file video mẫu trong database nếu test_video_path không tồn tại
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos", "database")
            for root, _, files in os.walk(sample_dir):
                for f in files:
                    if f.endswith((".avi", ".mp4")):
                        test_video_path = os.path.join(root, f)
                        break
                if os.path.exists(test_video_path):
                    break
        search_similar_video(test_video_path, top_k=5)