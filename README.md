# DeepSeek Multi-Asset Trading Bot (Local & Live)

Hệ thống giao dịch tự động sử dụng trí tuệ nhân tạo (DeepSeek V3.1) để phân tích thị trường và đưa ra quyết định giao dịch đa tài sản. Bot hỗ trợ cả chế độ Backtest bằng dữ liệu lịch sử cục bộ và giao dịch Paper/Live trên sàn Hyperliquid.

## 🚀 Tính năng nổi bật

- **Event-Driven AI (Mới):** Tối ưu chi phí API bằng cách chỉ gọi AI phân tích khi thị trường có biến động mạnh (>2%), RSI quá mua/bán, hoặc cần quản lý lệnh.
- **Local Dataset Backtesting:** Cho phép chạy backtest nhanh chóng bằng file CSV trong folder `dataset` (Hỗ trợ Chứng khoán, Vàng, Crypto...).
- **Warmup Analysis:** AI được cung cấp 100 nến quá khứ tại mỗi thời điểm để phân tích xu hướng, hỗ trợ/kháng cự mà không nhìn trước tương lai.
- **Multi-Asset & Multi-Timeframe:** Phân tích đồng thời nhiều mã tài sản với sự kết hợp của nến 15m (Execution), 1h (Structure), và 4h (Trend).
- **Telegram Integration:** Gửi báo cáo kết quả Backtest và tín hiệu Entry/Close chi tiết (kèm lý do phân tích) trực tiếp về điện thoại.

## 📁 Cấu trúc thư mục

- `dataset/`: Chứa các file dữ liệu CSV (`AAPL`, `GOLD`, `BTC`...) để Backtest.
- `prompts/`: Chứa `system_prompt.txt` - "linh hồn" của bot, nơi định nghĩa chiến thuật giao dịch.
- `data-backtest/`: Lưu trữ kết quả, log và biểu đồ của các lần chạy backtest.
- `bot.py`: Core xử lý logic, tính toán chỉ báo và gọi API AI.
- `backtest.py`: Harness điều khiển việc giả lập thị trường từ dữ liệu CSV.

## 🛠 Cài đặt & Cấu hình

1. **Cài đặt thư viện:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình file `.env`:**

   ```env
   # API Keys
   OPENROUTER_API_KEY=your_key
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_id

   # Backtest Settings
   BACKTEST_SYMBOLS=AAPL,GOLD  # Tên symbol phải trùng với tiền tố file CSV
   BACKTEST_INTERVAL=1d        # Khung thời gian của dữ liệu (1d, 1h, 15m)
   BACKTEST_START=2008-01-01
   BACKTEST_END=2009-12-31
   BACKTEST_START_CAPITAL=10000
   ```

## 📈 Cách sử dụng

### 1. Chạy Backtest (Dữ liệu Local)

Bot sẽ quét folder `dataset`, nạp dữ liệu và bắt đầu giả lập. AI sẽ chỉ "thức dậy" khi có sự kiện quan trọng để tiết kiệm token.

```bash
python backtest.py
```

### 2. Chạy Giao dịch thực tế (Paper/Live)

Bot sẽ lấy dữ liệu trực tiếp từ sàn Binance và thực hiện lệnh trên Hyperliquid.

```bash
python bot.py
```

### 3. Up lên index.html

Sau khi chạy xong backtest, dùng lệnh python replay/build_replay_site.py --data data-backtest/run-xxx-xxx (ví dụ: 20260520-124919) sẽ có 1 file index.html trong folder replay

## 📊 Chiến thuật của AI

Bot sử dụng mô hình DeepSeek V3.1 với khả năng suy nghĩ (Reasoning). Các thông số kỹ thuật cung cấp cho AI bao gồm:

- **EMA:** 20, 50, 200 (Xác định xu hướng).
- **RSI & MACD:** Xác định động lượng và điểm đảo chiều.
- **ATR:** Tính toán khoảng cách đặt Stop Loss và Take Profit theo biến động thực tế.
- **Trend Strength Score:** Điểm số sức mạnh xu hướng do hệ thống tự tính toán.

## ⚠️ Lưu ý

- Backtest chỉ mang tính chất tham khảo dựa trên dữ liệu lịch sử.
- Luôn kiểm tra kỹ cấu hình `risk_usd` trong Prompt để quản lý vốn an toàn.
- Đảm bảo các file CSV trong `dataset` có định dạng: `Date, Price, Open, High, Low, Vol.`.
