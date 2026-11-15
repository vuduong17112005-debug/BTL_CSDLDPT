import os

# Đường dẫn thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn thư mục chứa video cơ sở dữ liệu
DATABASE_DIR = os.path.join(BASE_DIR, "videos", "database")

# Đường dẫn file Cơ sở dữ liệu SQLite
SQLITE_DB_PATH = os.path.join(BASE_DIR, "video_multimedia.db")

# Đường dẫn mô hình YOLO Pose
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")

# Số lượng khung hình lấy mẫu đều từ mỗi video
NUM_SAMPLE_FRAMES = 16

# Ngưỡng tin cậy nhận diện người của YOLO
PERSON_CONF_THRESHOLD = 0.40

# Định dạng các file video hỗ trợ
SUPPORTED_EXTENSIONS = (".avi", ".mp4", ".mkv", ".mov")

# Số lượng bins cho Histogram màu kênh Hue (HSV)
HUE_HIST_BINS = 16
