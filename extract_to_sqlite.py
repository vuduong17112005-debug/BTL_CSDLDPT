import os
import sys
import json
import sqlite3
import argparse
from tqdm import tqdm

from config import (
    DATABASE_DIR,
    SQLITE_DB_PATH,
    SUPPORTED_EXTENSIONS
)
from feature_extractor import VideoFeatureExtractor

def init_database(db_path=SQLITE_DB_PATH, reset=False):
    """
    Khởi tạo bảng video_features và các chỉ mục trong SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if reset:
        print(f"[DB] Đang xóa bảng cũ (reset) tại: {db_path} ...")
        cursor.execute("DROP TABLE IF EXISTS video_features")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_features (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            action_category TEXT NOT NULL,
            duration_sec REAL,
            fps REAL,
            total_frames INTEGER,
            
            -- 1. Số lượng chủ thể và Tương tác (Person Count & Interaction)
            person_count INTEGER DEFAULT 1,
            interaction_iou REAL DEFAULT 0.0,
            interaction_label TEXT DEFAULT 'Solo',
            
            -- 2. Quy mô và cự ly ống kính (Camera Scale / Shot Type)
            scale_ratio REAL DEFAULT 0.35,
            scale_label TEXT DEFAULT 'Medium',
            
            -- 3. Tư thế cơ thể qua góc khớp xương (Body Pose & Joint Angles)
            bbox_aspect_ratio REAL DEFAULT 1.5,
            hip_joint_angle REAL DEFAULT 165.0,
            knee_joint_angle REAL DEFAULT 170.0,
            pose_label TEXT DEFAULT 'Standing',
            
            -- 4. Phạm vi dịch chuyển trọng tâm (Motion Scope)
            centroid_std REAL DEFAULT 0.0,
            motion_scope_label TEXT DEFAULT 'In-place',
            
            -- 5. Vận tốc và biên độ cử động tứ chi (Limb Velocity & Amplitude)
            limb_velocity REAL DEFAULT 0.0,
            amplitude_label TEXT DEFAULT 'Moderate',
            
            -- 6. Sắc độ màu môi trường chủ đạo (Dominant Hue & Environment)
            dominant_hue REAL DEFAULT 0.0,
            hue_histogram_json TEXT,
            environment_label TEXT DEFAULT 'Indoor / Warm Tone',
            
            -- 7. Độ ổn định phông nền (Background Stability)
            bg_diff_score REAL DEFAULT 0.0,
            bg_stability_label TEXT DEFAULT 'Static Background',
            
            -- 8. Độ tương phản người và phông nền (Foreground vs Background Contrast)
            contrast_score REAL DEFAULT 0.0,
            contrast_label TEXT DEFAULT 'High-contrast',
            
            -- Vector đặc trưng tổng hợp chuẩn hóa (26 chiều)
            feature_vector_json TEXT NOT NULL
        )
    """)

    # Tự động cập nhật cột mới nếu bảng đã tồn tại từ trước (Migration)
    cursor.execute("PRAGMA table_info(video_features)")
    existing_cols = [c[1] for c in cursor.fetchall()]

    if "environment_label" not in existing_cols:
        cursor.execute("ALTER TABLE video_features ADD COLUMN environment_label TEXT DEFAULT 'Indoor / Warm Tone'")
        cursor.execute("""
            UPDATE video_features
            SET environment_label = CASE
                WHEN dominant_hue BETWEEN 35.0 AND 85.0 THEN 'Green / Turf Field'
                WHEN dominant_hue BETWEEN 90.0 AND 130.0 THEN 'Aquatic / Water'
                ELSE 'Indoor / Warm Tone'
            END
        """)

    if "contrast_label" not in existing_cols:
        cursor.execute("ALTER TABLE video_features ADD COLUMN contrast_label TEXT DEFAULT 'High-contrast'")
        cursor.execute("""
            UPDATE video_features
            SET contrast_label = CASE
                WHEN contrast_score >= 55.0 THEN 'High-contrast'
                ELSE 'Low-contrast'
            END
        """)

    # Đồng bộ nhãn ngữ nghĩa chuẩn hóa theo tài liệu
    cursor.execute("""
        UPDATE video_features
        SET bg_stability_label = CASE
            WHEN bg_stability_label = 'Static' THEN 'Static Background'
            WHEN bg_stability_label = 'Dynamic' THEN 'Dynamic Background'
            ELSE bg_stability_label
        END,
        pose_label = CASE
            WHEN pose_label = 'Horizontal/Lying' THEN 'Horizontal / Lying'
            WHEN pose_label = 'Sitting/Bending' THEN 'Sitting / Bending'
            ELSE pose_label
        END,
        amplitude_label = CASE
            WHEN amplitude_label = 'Fine/Local' THEN 'Fine / Local'
            ELSE amplitude_label
        END
    """)

    # Tạo chỉ mục (Index) hỗ trợ truy vấn lọc thô tốc độ cao
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_category ON video_features(action_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pose_label ON video_features(pose_label)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_motion_scope ON video_features(motion_scope_label)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scale_label ON video_features(scale_label)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_person_count ON video_features(person_count)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_environment_label ON video_features(environment_label)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_contrast_label ON video_features(contrast_label)")

    conn.commit()
    conn.close()
    print(f"[DB] Khởi tạo/cập nhật cấu trúc bảng SQLite thành công tại: {db_path}")

def get_existing_paths(db_path):
    """Lấy danh sách các đường dẫn video đã có trong CSDL để tránh trích xuất lại"""
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT file_path FROM video_features")
        rows = cursor.fetchall()
        return set(row[0] for row in rows)
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()

def scan_video_files(directory):
    """Quét toàn bộ file video trong thư mục database và các thư mục con"""
    video_list = []
    if not os.path.exists(directory):
        print(f"[Cảnh báo] Thư mục database không tồn tại: {directory}")
        return video_list

    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.abspath(os.path.join(root, file))
                # Tên danh mục hành động lấy từ tên thư mục chứa (vd: ApplyLipstick1 -> ApplyLipstick)
                folder_name = os.path.basename(root)
                category = folder_name.rstrip("0123456789") if folder_name else "Unknown"
                video_list.append((full_path, file, category))

    return video_list

def main():
    parser = argparse.ArgumentParser(description="Trích xuất đặc trưng video vào SQLite")
    parser.add_argument("--reset", action="store_true", help="Xóa bảng cũ và trích xuất lại từ đầu")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng video cần xử lý")
    parser.add_argument("--db", type=str, default=SQLITE_DB_PATH, help="Đường dẫn file database SQLite")
    args = parser.parse_args()

    init_database(args.db, reset=args.reset)
    existing_paths = set() if args.reset else get_existing_paths(args.db)

    print(f"[Quét] Đang quét thư mục: {DATABASE_DIR} ...")
    all_videos = scan_video_files(DATABASE_DIR)
    print(f"[Quét] Tìm thấy tổng cộng {len(all_videos)} file video.")

    # Lọc các video chưa trích xuất
    pending_videos = [v for v in all_videos if v[0] not in existing_paths]
    print(f"[Trạng thái] Đã xử lý trước đó: {len(existing_paths)} video. Còn lại cần xử lý: {len(pending_videos)} video.")

    if args.limit and args.limit > 0:
        pending_videos = pending_videos[:args.limit]
        print(f"[Giới hạn] Chỉ xử lý {len(pending_videos)} video theo tham số --limit.")

    if not pending_videos:
        print("[Hoàn tất] Không có video nào cần xử lý thêm!")
        return

    # Khởi tạo mô hình trích xuất đặc trưng
    extractor = VideoFeatureExtractor()

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    success_count = 0
    error_count = 0

    print(f"\n[Bắt đầu] Bắt đầu trích xuất đặc trưng...")
    with tqdm(total=len(pending_videos), desc="Trích xuất video", unit="vid") as pbar:
        for full_path, file_name, category in pending_videos:
            try:
                feat = extractor.extract(full_path)
                if feat is None:
                    error_count += 1
                    pbar.update(1)
                    continue

                cursor.execute("""
                    INSERT OR REPLACE INTO video_features (
                        file_name, file_path, action_category, duration_sec, fps, total_frames,
                        person_count, interaction_iou, interaction_label,
                        scale_ratio, scale_label,
                        bbox_aspect_ratio, hip_joint_angle, knee_joint_angle, pose_label,
                        centroid_std, motion_scope_label,
                        limb_velocity, amplitude_label,
                        dominant_hue, hue_histogram_json, environment_label,
                        bg_diff_score, bg_stability_label,
                        contrast_score, contrast_label,
                        feature_vector_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_name,
                    full_path,
                    category,
                    feat["duration_sec"],
                    feat["fps"],
                    feat["total_frames"],
                    feat["person_count"],
                    feat["interaction_iou"],
                    feat["interaction_label"],
                    feat["scale_ratio"],
                    feat["scale_label"],
                    feat["bbox_aspect_ratio"],
                    feat["hip_joint_angle"],
                    feat["knee_joint_angle"],
                    feat["pose_label"],
                    feat["centroid_std"],
                    feat["motion_scope_label"],
                    feat["limb_velocity"],
                    feat["amplitude_label"],
                    feat["dominant_hue"],
                    json.dumps(feat["hue_histogram"]),
                    feat["environment_label"],
                    feat["bg_diff_score"],
                    feat["bg_stability_label"],
                    feat["contrast_score"],
                    feat["contrast_label"],
                    json.dumps(feat["feature_vector"])
                ))

                success_count += 1
                if success_count % 10 == 0:
                    conn.commit()

            except Exception as e:
                print(f"\n[Lỗi] Không thể trích xuất {file_name}: {e}")
                error_count += 1

            pbar.update(1)

    conn.commit()
    conn.close()

    print(f"\n==========================================")
    print(f" HOÀN TẤT TRÍCH XUẤT ĐẶC TRƯNG VÀO SQLITE")
    print(f" File CSDL: {args.db}")
    print(f" Thành công: {success_count} video")
    print(f" Thất bại:   {error_count} video")
    print(f"==========================================")

if __name__ == "__main__":
    main()
