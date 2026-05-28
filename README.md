# LLM & SMC Multi-Asset Trading Bot (Local & Live)

Hệ thống giao dịch tự động tích hợp Trí tuệ nhân tạo (LLM/DeepSeek) và phương pháp Smart Money Concepts (SMC) để phân tích và thực thi lệnh đa tài sản. Hệ thống hỗ trợ song song hai chế độ: mô phỏng giao dịch bằng AI Agent (LLM) và mô phỏng giao dịch bằng bộ quy tắc toán học SMC/Wyckoff đối chứng (Deterministic Baseline).

---

## 🚀 Tính năng nổi bật

*   **Hai chế độ giao dịch đối chứng:**
    *   **AI Agent Mode ([bot.py](file:///D:/NCKH/LLM_trading_ScienceResearch/bot.py)):** Phân tích thị trường bằng mô hình ngôn ngữ lớn (LLM - DeepSeek/Gemini/Llama) với khả năng lập luận phân vùng rủi ro (Risk Zones), hội tụ chỉ báo và quản lý vị thế chủ động.
    *   **Deterministic Baseline ([run_baseline.py](file:///D:/NCKH/LLM_trading_ScienceResearch/scripts/run_baseline.py)):** Áp dụng trực tiếp bộ quy tắc SMC (Order Block, FVG), Wyckoff (Markup/Markdown) và Price Action lướt sóng (Scalping) để vào lệnh Limit và Breakout tự động.
*   **Event-Driven AI Wakeup:** Giảm thiểu tới 80% chi phí gọi API LLM bằng cách chỉ đánh thức AI khi thị trường có biến động mạnh ($\ge 0.8\%$), RSI rơi vào vùng cực hạn, hoặc tài khoản có lệnh mở cần quản lý.
*   **Hạ tầng Backtest lịch sử mạnh mẽ:** Tái sử dụng mô hình khớp lệnh trong nến (intrabar TP/SL), mô phỏng phí sàn (maker/taker) và trượt giá (slippage) thực tế cho cả hai chế độ.
*   **Kiểm định thống kê khoa học:** Tích hợp kiểm định phân phối tỷ suất sinh lời (Student's t-test, Mann-Whitney U) và Bootstrap chênh lệch tỷ lệ Sharpe/Sortino để đánh giá xem AI thực sự vượt trội hơn Baseline truyền thống hay không.
*   **Trực quan hóa đồ thị & Replay:** Streamlit Dashboard giám sát tài khoản trực tiếp và công cụ xuất giao diện HTML động để xem lại (replay) từng lệnh đã khớp trực quan.
*   **Tích hợp Telegram:** Gửi tín hiệu vào lệnh, chốt lời/cắt lỗ kèm phân tích và báo cáo hiệu suất tức thời về điện thoại.

---

## 📁 Cấu trúc dự án

```text
LLM_trading_ScienceResearch/
├── dataset/                  # Chứa dữ liệu lịch sử CSV (AAPL, GOLD, BTC...)
├── prompts/                  # Nơi định nghĩa System Prompt và chỉ thị cho AI Agent
│   └── system_prompt.txt     # Chiến thuật lõi của AI Bot
├── replay/                   # Công cụ phát lại giao dịch trực quan
│   ├── build_replay_site.py  # Tạo tệp index.html để xem lại đồ thị giao dịch
│   └── index.html            # Giao diện hiển thị đồ thị và các điểm vào/thoát lệnh
├── scripts/                  # Thư mục chứa các script vận hành phụ trợ
│   ├── run_baseline.py       # Bot chạy baseline đối chứng (SMC/Wyckoff/Price Action)
│   ├── run_stats_significance.py  # Kiểm định ý nghĩa thống kê giữa AI và Baseline
│   └── recalculate_portfolio.py   # Tính toán lại số dư tài sản từ nhật ký giao dịch
├── bot.py                    # Khởi chạy logic chính, kết nối API sàn/LLM
├── backtest.py               # Trình giả lập dòng thời gian thị trường cho AI Bot
├── dashboard.py              # Streamlit Dashboard giám sát Live/Paper Trading
├── LogicAI.md                # Tài liệu kỹ thuật chi tiết cơ chế hoạt động của AI Bot
├── logicbaseline.md          # Tài liệu kỹ thuật chi tiết cơ chế hoạt động của Baseline Bot
├── requirements.txt          # Các thư viện Python cần cài đặt
└── .env                      # File chứa cấu hình môi trường bảo mật
```

---

## 🛠 Hướng dẫn Cài đặt

1. **Cài đặt thư viện Python:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Thiết lập file cấu hình môi trường:**
   Sao chép file `.env.example` thành `.env` và điền đầy đủ các thông tin:
   ```env
   # API Keys (OpenRouter làm proxy gọi LLM, Telegram thông báo)
   OPENROUTER_API_KEY=sk-or-v1-...
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...

   # Cấu hình Mô hình ngôn ngữ lớn (cho AI Agent Mode)
   TRADEBOT_LLM_MODEL=meta-llama/llama-3.3-70b-instruct
   TRADEBOT_LLM_TEMPERATURE=0.2

   # Cấu hình Backtest
   BACKTEST_SYMBOLS=GOLD       # Ký hiệu tài sản cần chạy (phải khớp tiền tố file CSV)
   BACKTEST_INTERVAL=1d        # Khung thời gian chạy nến (1d, 1h, 15m, 5m...)
   BACKTEST_START=2008-01-01   # Ngày bắt đầu kiểm thử
   BACKTEST_END=2009-12-31     # Ngày kết thúc kiểm thử
   BACKTEST_START_CAPITAL=1000 # Số vốn ban đầu ($)
   ```

---

## 📈 Hướng dẫn Sử dụng

### 1. Kiểm thử AI Agent (AI Backtesting)
Giả lập thị trường qua dữ liệu CSV trong folder `dataset/` và cho AI đưa ra quyết định giao dịch theo từng nến:
```bash
python backtest.py
```
Kết quả thống kê, nhật ký giao dịch và tệp tin cấu hình sẽ tự động lưu vào thư mục `data-backtest/run-YYYYMMDD-HHMMSS/`.

### 2. Kiểm thử Baseline (Baseline Backtesting)
Chạy bộ quy tắc SMC Scalping đối chứng trên cùng khoảng thời gian và dữ liệu lịch sử:
```bash
$env:PYTHONIOENCODING="utf-8"
python scripts/run_baseline.py
```
Kết quả được lưu vào thư mục `data-backtest/` tương tự để thực hiện đối chứng.

### 3. Tạo Replay Đồ thị (Build Trade Replay)
Để xem trực quan biểu đồ nến kèm điểm mua/bán và giá trị tài sản qua từng nến, biên dịch thư mục kết quả backtest thành trang HTML Replay:
```bash
python replay/build_replay_site.py --data data-backtest/run-YOUR_RUN_ID
```
Sau lệnh này, mở tệp `replay/index.html` bằng trình duyệt web để xem đồ thị tương tác.

### 4. Kiểm định Ý nghĩa Thống kê (Statistical Significance Test)
Đánh giá độ tin cậy khoa học xem hiệu suất vượt trội của AI so với Baseline có thực sự mang ý nghĩa thống kê hay chỉ do may mắn ngẫu nhiên:
```bash
python scripts/run_stats_significance.py --agent data-backtest/run-AGENT_ID --baseline data-backtest/run-BASELINE_ID
```
Một báo cáo nghiên cứu chi tiết `statistical_significance_report.md` sẽ được tạo ra tại thư mục `data-backtest/`.

### 5. Giám sát Live & Paper Trading
Khởi chạy bot trong môi trường Live (dữ liệu Binance thực tế, đặt lệnh trên sàn Hyperliquid):
```bash
python bot.py
```
Mở giao diện giám sát Streamlit Dashboard thời gian thực:
```bash
streamlit run dashboard.py
```

---

## 📘 Tài liệu tham khảo sâu
* Xem chi tiết kiến trúc Prompt, Event-Driven Wakeup và Quản trị rủi ro của AI Bot tại: [LogicAI.md](file:///D:/NCKH/LLM_trading_ScienceResearch/LogicAI.md).
* Xem chi tiết cách nhận diện SMC (OB, FVG, BOS, CHoCH), Wyckoff và quy tắc lướt sóng của Baseline Bot tại: [logicbaseline.md](file:///D:/NCKH/LLM_trading_ScienceResearch/logicbaseline.md).

---

## ⚠️ Lưu ý Quan trọng
* Dữ liệu CSV trong thư mục `dataset/` phải có định dạng tiêu chuẩn gồm các cột: `Date, Price, Open, High, Low, Vol., Change %` (phân cách bằng dấu phẩy).
* Hệ thống này được xây dựng cho mục đích nghiên cứu khoa học và kiểm thử lý thuyết. Không khuyến khích nạp tiền thật chạy live khi chưa kiểm thử kỹ lưỡng.
