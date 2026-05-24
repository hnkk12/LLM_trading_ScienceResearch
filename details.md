# Chi tiết Dự án: DeepSeek Backtest & Research System (Daily SMC)

Tài liệu này tập trung vào hệ thống kiểm thử chiến thuật (Backtest) và nghiên cứu các mô hình giao dịch dựa trên AI DeepSeek V3.1 và Smart Money Concepts (SMC) trên các tập dữ liệu lịch sử.

---

## 1. Tổng quan Hệ thống Backtest

Hệ thống được thiết kế để đánh giá hiệu suất của mô hình DeepSeek V3.1 trong việc nhận diện các dấu chân của "Dòng tiền thông minh" (Smart Money) trên các loại tài sản khác nhau (Chứng khoán, Vàng, Crypto).

- **Trình thực thi chính**: `backtest.py`
- **Mục tiêu**: Mô phỏng quá trình giao dịch, đo lường tỷ lệ thắng, lợi nhuận, và quản trị rủi ro mà không cần thực thi lệnh thực tế.
- **Cấu trúc dữ liệu**: Sử dụng các file CSV chuẩn hóa trong thư mục `dataset/`.

---

## 2. Nguồn dữ liệu Nghiên cứu (`dataset/`)

Hệ thống tập trung phân tích trên các tập dữ liệu khung Ngày (Daily) để đảm bảo tính ổn định của các tín hiệu SMC:

- **AAPL (Apple Inc.)**: Dữ liệu chứng khoán các giai đoạn 2008-2009, 2020-2021, 2022-2023.
- **Gold (XAU/USD)**: Dữ liệu thị trường vàng qua các giai đoạn biến động kinh tế mạnh.

**Cấu trúc tệp tin CSV**:

- `Date`: Ngày giao dịch.
- `Open, High, Low, Price (Close)`: Các mức giá nến Ngày.
- `Vol.`: Khối lượng giao dịch (được xử lý định dạng K, M, B).

---

## 3. Logic AI & Quy trình Ra quyết định

Hệ thống không chỉ gửi dữ liệu thô cho AI mà thực hiện một quy trình xử lý tín hiệu phức tạp trước khi yêu cầu DeepSeek đưa ra quyết định:

### A. Tiền xử lý Tín hiệu (Python Side)
Trước khi gọi API, mã nguồn Python thực hiện các tính toán kỹ thuật để làm "đầu vào" cho AI:
*   **SMC Indicator Engine**: Tự động phát hiện các vùng **FVG (Fair Value Gaps)** và **Order Blocks (OB)** từ dữ liệu nến. Các vùng này được gửi cho AI dưới dạng tọa độ giá cụ thể (Top/Bottom).
*   **Trend Scoring**: Tính toán điểm số sức mạnh xu hướng dựa trên độ dốc của EMA200, giá trị ADX (>25 là có xu hướng) và sự hội tụ/phân kỳ của MACD.
*   **Volatility Mapping**: Sử dụng ATR (Average True Range) để đo lường "độ thở" của thị trường, giúp AI xác định khoảng cách đặt SL/TP hợp lý.

### B. Cấu trúc Prompt (The "Brain")
AI (DeepSeek V3.1) được cung cấp một "khung tư duy" thông qua System Prompt và Context:
1.  **Dữ liệu chuỗi thời gian**: AI nhận được chuỗi giá đóng cửa của 10-20 nến gần nhất để nhận diện mô hình giá (Price Action).
2.  **Chỉ dẫn SMC**: AI được yêu cầu ưu tiên các vùng thanh khoản và mất cân bằng giá. Nó phải trả lời được câu hỏi: "Giá đang tìm về vùng FVG nào?" hoặc "Đã có cú quét thanh khoản (Liquidity Sweep) nào xảy ra chưa?".
3.  **Phản hồi có cấu trúc (JSON)**: AI bắt buộc phải trả về dữ liệu định dạng JSON để mã nguồn có thể thực thi lệnh ngay lập tức, bao gồm cả phần `justification` (lý giải bằng văn bản) để người dùng có thể hậu kiểm.

### C. Logic Suy luận của AI
Khi nhận được dữ liệu, DeepSeek thực hiện các bước suy luận:
1.  **Xác định Xu hướng**: So sánh vị thế giá với các đường EMA.
2.  **Tìm kiếm Confluence (Sự hội tụ)**: Ví dụ: Nếu Giá chạm Order Block + RSI quá bán + Có nến rút chân -> AI sẽ tăng điểm tin cậy (`confidence`) để vào lệnh.
3.  **Tính toán Risk/Reward**: AI được hướng dẫn chỉ vào lệnh khi tỷ lệ R:R tối thiểu đạt 1:1.5 hoặc 1:2.

---

## 4. Quản trị Rủi ro trong Mô phỏng


## 4. Quản trị Rủi ro trong Mô phỏng

Hệ thống áp dụng các quy tắc nghiêm ngặt để đảm bảo kết quả Backtest phản ánh đúng thực tế:

- **Mô hình Rủi ro 1%**: Mỗi lệnh giả lập chỉ được phép rủi ro tối đa 1% tổng số dư hiện tại.
- **Position Sizing**: Khối lượng lệnh được tính toán động dựa trên khoảng cách từ điểm vào đến `stop_loss` do AI chỉ định.
- **Phased Trailing Stop**: Mô phỏng việc dời SL về Breakeven hoặc Trailing theo ATR để bảo vệ lợi nhuận trong suốt quá trình chạy dữ liệu.
- **Chi phí Giao dịch**: Khấu trừ phí giao dịch (Taker fee) vào kết quả để đảm bảo tính thực tế.

---

## 5. Chỉ số Đánh giá Hiệu suất

Sau khi kết thúc quá trình chạy dữ liệu trong `dataset/`, hệ thống xuất ra các báo cáo:

- **Equity Curve**: Biểu đồ tăng trưởng vốn theo thời gian.
- **Max Drawdown (MDD)**: Mức sụt giảm vốn lớn nhất từ đỉnh.
- **Sharpe & Sortino Ratio**: Đánh giá lợi nhuận trên đơn vị rủi ro.
- **AI Justification Log**: Lưu trữ toàn bộ lý do vào lệnh của AI (`ai_decisions.csv`) để nghiên cứu lại các lỗi hoặc các lệnh thắng đậm.

---

_Tài liệu này tập trung hoàn toàn vào quy trình nghiên cứu và Backtest trên dataset local._
