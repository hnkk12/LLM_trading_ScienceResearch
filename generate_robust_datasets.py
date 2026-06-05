import pandas as pd
import numpy as np
import os
from pathlib import Path
import random

# --- CẤU HÌNH ---
INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset_robust"
BLOCK_SIZE = 10  # Số ngày trong 1 khối để xáo trộn (giữ tính liên tục của pattern)
NOISE_LEVEL = 0.001  # 0.1% nhiễu ngẫu nhiên để tạo tính "Synthetic"
RANDOM_SEED = 42

def clean_and_parse(value):
    """Làm sạch dữ liệu số có dấu phẩy hoặc ký tự đặc biệt."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        return float(value.replace(",", "").replace('"', '').replace('%', ''))
    return float(value)

def process_file(file_path, output_path):
    print(f"Processing: {file_path.name}...")
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Lưu lại cột Date gốc để gán lại sau khi xáo trộn (giữ trục thời gian thẳng)
    # Tìm cột Date (không phân biệt hoa thường)
    date_col = next((c for c in df.columns if c.lower() == "date"), "Date")
    original_dates = df[date_col].copy()
    
    # 2. Xử lý các cột giá (Open, High, Low, Price/Close)
    price_cols = [col for col in df.columns if col.lower() in ['price', 'open', 'high', 'low', 'close']]
    for col in price_cols:
        df[col] = df[col].apply(clean_and_parse)

    # 3. Permutation-Scrambled (Block Bootstrapping)
    # Chia dữ liệu thành các khối
    chunks = [df.iloc[i:i + BLOCK_SIZE] for i in range(0, len(df), BLOCK_SIZE)]
    random.shuffle(chunks) # Xáo trộn các khối
    df_scrambled = pd.concat(chunks).reset_index(drop=True)

    # 4. Synthetic Augmentation (Thêm nhiễu Gaussian)
    # Tạo nhiễu nhỏ để giá không y hệt lịch sử
    for col in price_cols:
        noise = np.random.normal(0, NOISE_LEVEL, size=len(df_scrambled))
        df_scrambled[col] = df_scrambled[col] * (1 + noise)

    # 5. Đảm bảo tính logic của nến (High >= Open/Close/Low và Low <= Open/Close/High)
    # Xác định các cột giá chính xác có mặt trong file
    available_price_cols = [c for c in price_cols if c in df_scrambled.columns]
    df_scrambled['High'] = df_scrambled[available_price_cols].max(axis=1)
    df_scrambled['Low'] = df_scrambled[available_price_cols].min(axis=1)

    # 6. Gán lại ngày tháng theo thứ tự chuẩn để không lỗi Backtest
    df_scrambled[date_col] = original_dates.values
    
    # 7. Tính lại Change % dựa trên giá mới (nếu có cột này)
    price_col_name = next((c for c in df_scrambled.columns if c.lower() in ['price', 'close']), None)
    change_col_name = next((c for c in df_scrambled.columns if c.lower() == 'change %'), 'Change %')
    
    if price_col_name:
        pct_change = df_scrambled[price_col_name].pct_change().fillna(0)
        df_scrambled[change_col_name] = pct_change.map(lambda x: f"{x:.2%}")

    # Xuất file
    df_scrambled.to_csv(output_path, index=False)
    print(f" Saved: {output_path}")

def main():
    # Tạo thư mục đầu ra nếu chưa có
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    if not input_path.exists():
        print(f"Error: Thư mục {INPUT_DIR} không tồn tại!")
        return

    # Lấy tất cả file csv trong dataset
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print("Error: Không tìm thấy file CSV nào.")
        return

    print(f"Starting Robust dataset generation for {len(csv_files)} files...")
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    for file_path in csv_files:
        if "_ROBUST" in file_path.name: continue # Tránh xử lý đè lên file đã tạo nếu để cùng folder
        new_filename = file_path.stem + "_ROBUST.csv"
        process_file(file_path, output_path / new_filename)

    print(f"\nComplete! All robust datasets are located at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
