import os
import json
import sqlite3
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from config import SQLITE_DB_PATH
from feature_extractor import VideoFeatureExtractor
from search_similar_video import cosine_similarity

st.set_page_config(page_title="Hệ thống Tìm kiếm Video Tương đồng", page_icon="🎬", layout="wide")

VISUAL_CACHE_PATH = "visual_embeddings.npz"

@st.cache_resource
def load_models():
    """Load đồng thời YOLOv8-Pose (của nhóm) và MobileNetV2 (Visual Semantic)"""
    pose_extractor = VideoFeatureExtractor()
    
    # Load MobileNetV2 pre-trained để trích xuất đặc trưng hình ảnh sâu (1280 chiều)
    weights = models.MobileNet_V2_Weights.DEFAULT
    mobilenet = models.mobilenet_v2(weights=weights)
    mobilenet.classifier = torch.nn.Identity()  # Bỏ lớp phân loại cuối, lấy vector 1280-d
    mobilenet.eval()
    
    preprocess = weights.transforms()
    return pose_extractor, mobilenet, preprocess

def extract_visual_embedding(video_path, mobilenet, preprocess, num_frames=6):
    """Lấy mẫu đều num_frames khung hình và trích xuất vector thị giác 1280 chiều"""
    if not video_path or not os.path.exists(video_path):
        return np.zeros(1280, dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros(1280, dtype=np.float32)

    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames_tensor = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = transforms.ToPILImage()(rgb)
            frames_tensor.append(preprocess(pil_img))
    cap.release()

    if not frames_tensor:
        return np.zeros(1280, dtype=np.float32)

    batch = torch.stack(frames_tensor)
    with torch.no_grad():
        feats = mobilenet(batch)  # (N, 1280)
        mean_feat = feats.mean(dim=0).numpy()
        norm = np.linalg.norm(mean_feat)
        if norm > 1e-6:
            mean_feat = mean_feat / norm
    return mean_feat.astype(np.float32)

def resolve_local_path(db_path_str):
    """Chuyển đường dẫn lưu trong SQLite sang đường dẫn thực tế trên máy hiện tại"""
    if os.path.exists(db_path_str):
        return db_path_str
    parts = db_path_str.replace("\\", "/").split("videos/")
    if len(parts) > 1:
        local_p = os.path.join("videos", parts[1])
        if os.path.exists(local_p):
            return local_p
    return db_path_str

@st.cache_resource
def build_or_load_visual_index(_mobilenet, _preprocess, db_path=SQLITE_DB_PATH):
    """Tạo chỉ mục đặc trưng thị giác cho toàn bộ video trong DB (chạy 1 lần duy nhất)"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT video_id, file_name, file_path FROM video_features").fetchall()
    conn.close()

    if os.path.exists(VISUAL_CACHE_PATH):
        data = np.load(VISUAL_CACHE_PATH, allow_pickle=True)
        cached_ids = data["video_ids"].tolist()
        if len(cached_ids) == len(rows):
            return dict(zip(cached_ids, data["embeddings"]))

    # Nếu chưa có cache thì tự động trích xuất (chỉ mất ~15-20 giây lần đầu)
    progress_bar = st.progress(0, text="Đang xây dựng chỉ mục thị giác (Visual Index) lần đầu cho CSDL...")
    video_ids = []
    embeddings = []
    total = len(rows)

    for i, r in enumerate(rows):
        real_p = resolve_local_path(r["file_path"])
        emb = extract_visual_embedding(real_p, _mobilenet, _preprocess, num_frames=5)
        video_ids.append(r["video_id"])
        embeddings.append(emb)
        if i % 10 == 0 or i == total - 1:
            progress_bar.progress((i + 1) / total, text=f"Đang lập chỉ mục thị giác: {i+1}/{total} videos...")

    progress_bar.empty()
    np.savez_compressed(VISUAL_CACHE_PATH, video_ids=np.array(video_ids), embeddings=np.array(embeddings))
    return dict(zip(video_ids, embeddings))

def convert_to_web_mp4(input_path):
    """Chuyển đổi video (.avi/.mov/...) sang .mp4 để trình duyệt phát được mượt mà"""
    if not input_path or not os.path.exists(input_path):
        return None
    if input_path.lower().endswith(".mp4"):
        return input_path

    temp_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_mp4 = os.path.join(temp_dir, f"preview_{base_name}.mp4")
    if os.path.exists(output_mp4):
        return output_mp4

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return output_mp4

def find_top_k_multimodal(q_pose_vec, q_vis_vec, visual_index, db_path=SQLITE_DB_PATH, top_k=5):
    """Kết hợp Đặc trưng Thị giác sâu (65%) + Đặc trưng Pose/Chuyển động SQLite (35%)"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM video_features").fetchall()
    conn.close()

    q_p = np.array(q_pose_vec, dtype=np.float32)
    scored_videos = []

    for row in rows:
        vid_id = row["video_id"]
        db_p = np.array(json.loads(row["feature_vector_json"]), dtype=np.float32)

        # Kiểm tra trùng khớp tuyệt đối
        if np.allclose(q_p, db_p, atol=1e-4):
            final_sim = 1.0
            vis_sim = 1.0
            pose_sim = 1.0
        else:
            # 1. Điểm tương đồng Thị giác sâu (Visual Semantic)
            db_v = visual_index.get(vid_id)
            vis_sim = cosine_similarity(q_vis_vec, db_v) if db_v is not None else 0.0

            # 2. Điểm tương đồng Hình thái & Chuyển động (10 chiều đầu của vector SQLite)
            dist_motion_pose = np.linalg.norm(q_p[:10] - db_p[:10]) / np.sqrt(10)
            pose_sim = 1.0 / (1.0 + 2.0 * dist_motion_pose)

            # 3. Hợp nhất đa phương thức (Multimodal Fusion)
            final_sim = 0.65 * vis_sim + 0.35 * pose_sim

        item = dict(row)
        item["similarity"] = float(final_sim)
        item["vis_sim"] = float(vis_sim)
        item["pose_sim"] = float(pose_sim)
        scored_videos.append(item)

    scored_videos.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_videos[:top_k]

# ================= GIAO DIỆN HỆ THỐNG =================
st.title("🎬 Hệ thống Tìm kiếm Video Hành động Tương đồng")

pose_extractor, mobilenet, preprocess = load_models()
visual_index = build_or_load_visual_index(mobilenet, preprocess, SQLITE_DB_PATH)

uploaded_file = st.file_uploader(
    "Chọn một file video đầu vào (độ dài tùy chọn, định dạng .avi, .mp4, .mov, .mkv):",
    type=["avi", "mp4", "mov", "mkv"]
)

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        query_path = tmp.name

    col_query, col_status = st.columns([1, 1.5])
    with col_query:
        st.subheader("🎥 Video Đầu vào (Query)")
        web_q_path = convert_to_web_mp4(query_path)
        if web_q_path:
            st.video(web_q_path)
        st.caption(f"Tệp tin: `{uploaded_file.name}`")

    with col_status:
        st.subheader("🔍 Kết quả Truy vấn Hệ thống")
        with st.spinner("Đang trích xuất đặc trưng Khung xương (YOLOv8-Pose) & Thị giác (MobileNetV2)..."):
            # 1. Trích xuất đặc trưng Pose/Motion từ module của nhóm
            if hasattr(pose_extractor, "extract_video_features"):
                q_feat = pose_extractor.extract_video_features(query_path)
            else:
                q_feat = pose_extractor.extract(query_path)
            q_pose_vec = q_feat["feature_vector"] if "feature_vector" in q_feat else json.loads(q_feat["feature_vector_json"])

            # 2. Trích xuất đặc trưng thị giác sâu
            q_vis_vec = extract_visual_embedding(query_path, mobilenet, preprocess, num_frames=6)

            # 3. Tìm kiếm Top 5
            top_5 = find_top_k_multimodal(q_pose_vec, q_vis_vec, visual_index, SQLITE_DB_PATH, top_k=5)

        st.success("Đã tìm thấy **5 video giống nhất** (Kết hợp Thị giác + Chuyển động khung xương).")
        summary_df = pd.DataFrame([
            {
                "Thứ hạng": f"Top {i}",
                "Tên file": r["file_name"],
                "Nhãn hành động": r["action_category"],
                "Giống hình ảnh": f"{r['vis_sim']*100:.1f}%",
                "Giống tư thế/động tác": f"{r['pose_sim']*100:.1f}%",
                "Tổng độ tương đồng": f"{r['similarity']*100:.2f}%"
            }
            for i, r in enumerate(top_5, start=1)
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🏆 Danh sách 5 Video Giống Nhất (Xếp thứ tự giảm dần)")

    cols = st.columns(5)
    for idx, (col, item) in enumerate(zip(cols, top_5), start=1):
        with col:
            st.markdown(f"**#{idx} — {item['similarity']*100:.2f}%**")
            local_vid = resolve_local_path(item["file_path"])
            web_res_path = convert_to_web_mp4(local_vid)
            if web_res_path and os.path.exists(web_res_path):
                st.video(web_res_path)
            else:
                st.warning("Không tìm thấy file video gốc trên ổ đĩa")
            st.caption(f"**File:** `{item['file_name']}`\n\n**Nhãn:** `{item['action_category']}`")