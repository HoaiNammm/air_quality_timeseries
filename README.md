# Beijing Multi-Site Air Quality Classification (No PM2.5)

Phân tích dữ liệu chất lượng không khí **Beijing Multi-Site Air Quality (12 stations)** để xây dựng mô hình **phân lớp mức độ ô nhiễm** mà **không sử dụng PM2.5**. Project triển khai pipeline đầy đủ từ tải & gộp dữ liệu → làm sạch → tạo nhãn (AQI/levels) → huấn luyện mô hình → đánh giá → sinh báo cáo.

---

## Features

- Load & merge dữ liệu từ **12 trạm**
- Làm sạch dữ liệu:
  - xử lý missing values
  - xử lý outliers cơ bản
  - chuẩn hoá kiểu dữ liệu thời gian
- Feature engineering:
  - trích xuất đặc trưng theo thời gian (hour/day/month/season)
  - lag/rolling features (tuỳ chọn)
- **Không dùng PM2.5** trong tập đặc trưng
- Mô hình phân lớp (gợi ý):
  - Logistic Regression / RandomForest / GradientBoosting / XGBoost (optional)
- Metrics:
  - Accuracy
  - Precision / Recall / F1
  - Confusion Matrix
  - ROC-AUC (nếu binary)
- Visualization:
  - phân phối lớp theo trạm
  - confusion matrix heatmap
  - feature importance / permutation importance
  - (tuỳ chọn) Plotly interactive
- Tự động hoá pipeline notebook bằng **Papermill**

---

## Project Structure

```text
beijing_air_quality_classification/
├── data/
│   ├── raw/
│   │   ├── station_01.csv
│   │   ├── station_02.csv
│   │   └── ... (12 stations)
│   └── processed/
│       ├── merged_cleaned.parquet
│       ├── dataset_features.parquet
│       ├── train.parquet
│       ├── test.parquet
│       └── metrics.json
│
├── notebooks/
│   ├── 01_preprocessing_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification_modelling.ipynb
│   └── runs/
│       ├── 01_preprocessing_and_eda_run.ipynb
│       ├── 02_feature_engineering_run.ipynb
│       └── 03_classification_modelling_run.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modelling.py
│   ├── evaluation.py
│   └── utils.py
│
├── run_papermill.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your_repo_url>
cd beijing_air_quality_classification
pip install -r requirements.txt
```

## Data Preparation

Đặt file gốc vào:
```

```bash
data/raw/PRSA2017_Data_20130301-20170228.zip
```
Hoặc tải dataset Beijing Multi-Site Air Quality Data (UCI) và đặt các file trạm vào:

```bash 
data/raw/
```
Ví dụ

```bash
data/raw/station_01.csv
data/raw/station_02.csv
...
data/raw/station_12.csv
```

File output sẽ được sinh tự động vào:
```bash
data/processed/
```



Run Pipeline (Recommended)
Chạy toàn bộ phân tích chỉ với 1 lệnh:

```bash
python run_papermill.py
```
Kết quả sinh ra:

```bash
data/processed/merged_cleaned.parquet
data/processed/dataset_features.parquet
data/processed/train.parquet
data/processed/test.parquet
data/processed/metrics.json
notebooks/runs/03_classification_modelling_run.ipynb
```

### Changing Parameters
Các tham số có thể chỉnh trong run_papermill.py:

```python
TARGET_MODE = "AQI_LEVEL"     # hoặc "BINARY_THRESHOLD"
EXCLUDE_COLS = ["PM2.5"]      # đảm bảo loại PM2.5
TEST_SPLIT_MODE = "TIME"      # "TIME" để tránh leakage, hoặc "RANDOM"
TEST_SIZE = 0.2
MODEL_NAME = "random_forest"  # "logreg", "rf", "gb", "xgb"(optional)
RANDOM_STATE = 42
```

Hoặc sửa trong cell PARAMETERS của mỗi notebook để chạy với cấu hình khác nhau.

### Visualization & Results

Notebook 03 hiển thị các nội dung sau:

Confusion Matrix

Classification Report (Precision/Recall/F1)

ROC curve (nếu bài toán binary)

Feature importance (tuỳ mô hình)

So sánh hiệu năng giữa các mô hình (nếu bật chế độ compare)

Bạn có thể export notebook chạy ra HTML:

```bash
jupyter nbconvert notebooks/runs/03_classification_modelling_run.ipynb --to html
```

## Ứng dụng thực tế 

Cảnh báo sớm mức độ ô nhiễm theo giờ/ngày cho từng trạm

So sánh đặc trưng ô nhiễm giữa các khu vực (multi-site analysis)

Thiết kế bài giảng ML: phân lớp + đánh giá + chống leakage theo thời gian

Nâng cấp sang chuỗi thời gian (forecast) và bán giám sát (semi-supervised)

### Tech Stack

| Công nghệ | Mục đích |
|----------|----------|
| Python | Ngôn ngữ chính |
| Pandas | Xử lý dữ liệu transaction |
| Scikit-learn | Modelling & metrics |
| Papermill | Chạy pipeline notebook tự động |
| Matplotlib & Seaborn | Visualization biểu đồ tĩnh |
| Plotly | Dashboard / biểu đồ tương tác |
| Jupyter Notebook | Môi trường notebook |

### Author
Project được thực hiện bởi:
Trang Le

📄 License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.