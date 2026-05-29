# Chi tiết Dự án: DeepSeek Backtest & Research System (Daily SMC)

Tài liệu này tập trung vào hệ thống kiểm thử chiến thuật (Backtest) và nghiên cứu các mô hình giao dịch dựa trên AI DeepSeek V3.1 và Smart Money Concepts (SMC) trên các tập dữ liệu lịch sử.

---

## 1. Tổng quan Hệ thống Backtest

Hệ thống được thiết kế để đánh giá hiệu suất của mô hình DeepSeek V3.1 trong việc nhận diện các dấu chân của "Dòng tiền thông minh" (Smart Money) trên các loại tài sản khác nhau (Chứng khoán, Vàng, Crypto) bằng cách sử dụng dữ liệu lịch sử địa phương.

- **Trình thực thi chính**: `backtest.py`
- **Mục tiêu**: Mô phỏng quá trình giao dịch, đo lường tỷ lệ thắng, lợi nhuận, và quản trị rủi ro trên dữ liệu quá khứ.
- **Cơ chế nạp dữ liệu**: Tự động quét, làm sạch và gộp nhiều tệp tin CSV cho cùng một mã tài sản để tạo ra chuỗi thời gian liên tục.

---

## 2. Nguồn dữ liệu Nghiên cứu (`dataset/`)

Hệ thống tập trung phân tích trên các tập dữ liệu khung **Ngày (Daily)** để đảm bảo tính ổn định của các tín hiệu SMC và giảm nhiễu thị trường:

- **AAPL (Apple Inc.)**: Dữ liệu chứng khoán qua các giai đoạn 2008-2009, 2020-2021, 2022-2023.
- **Gold (XAU/USD)**: Dữ liệu thị trường vàng (Vàng giao ngay) được chuẩn hóa từ các nguồn dữ liệu tài chính lớn.

**Cấu trúc dữ liệu và Tiền xử lý**:
- **Tự động xử lý định dạng**: Hệ thống tự động xóa bỏ dấu phẩy phân cách hàng nghìn (ví dụ: `1,552.40` -> `1552.40`) để đảm bảo tính toán số học chính xác.
- **Chuẩn hóa khối lượng**: Chuyển đổi các ký hiệu `K` (nghìn), `M` (triệu), `B` (tỷ) thành giá trị số tương ứng.
- **Gộp dữ liệu (Merging)**: Nếu một mã tài sản có nhiều file (ví dụ: AAPL_2008.csv và AAPL_2020.csv), hệ thống sẽ tự động gộp, loại bỏ trùng lặp và sắp xếp theo thời gian.

---

## 3. Logic AI & Quy trình Ra quyết định

Hệ thống sử dụng cơ chế **Event-Driven (Kích hoạt theo sự kiện)** để tối ưu hóa chi phí API và mô phỏng tư duy của một nhà giao dịch chuyên nghiệp:

### A. Cơ chế Kích hoạt AI (Wake-up Logic)
AI không được gọi ở mọi nến để tiết kiệm tài nguyên, thay vào đó DeepSeek chỉ được "đánh thức" khi:
- **Biến động mạnh (Volatility)**: Giá di chuyển > 2% trong một ngày.
- **Vùng cực trị (RSI Extremes)**: RSI < 35 hoặc > 65 (Cơ hội đảo chiều).
- **Cơ hội tích cực (Aggressive Mode)**: Khi có biến động > 0.8% kết hợp với vùng RSI thuận lợi.
- **Quản lý lệnh**: Kiểm tra định kỳ các vị thế đang mở để tối ưu dời Stop Loss.
- **Điểm bắt đầu/kết thúc**: Luôn gọi AI ở nến đầu tiên để thiết lập chiến lược và nến cuối cùng để tất toán.

### B. SMC Indicator Engine
Trước khi gửi dữ liệu cho AI, hệ thống tính toán:
- **FVG (Fair Value Gaps)**: Tìm kiếm các khoảng trống giá chưa được lấp đầy.
- **Order Blocks (OB)**: Xác định các vùng cung/cầu mạnh dựa trên nến nhấn chìm và khối lượng đột biến.
- **Trend Strength**: Điểm số xu hướng tổng hợp từ EMA, MACD và ADX.

### C. Khả năng phục hồi API (API Resilience)
- **Timeout 60s**: Tăng thời gian chờ để AI có đủ tài nguyên suy luận phức tạp.
- **Auto-Retry (3 lần)**: Tự động thử lại khi gặp sự cố mạng hoặc lỗi bắt tay SSL, đảm bảo quá trình backtest không bị gián đoạn giữa chừng.

---

## 4. Quản trị Rủi ro & Thực thi

Hệ thống áp dụng các quy tắc nghiêm ngặt:

- **Mô hình Rủi ro cố định**: Mặc định rủi ro 1% tài khoản trên mỗi lệnh (hoặc theo cấu hình AI).
- **Tính toán khối lượng động**: Khối lượng vị thế (`quantity`) được tính toán dựa trên khoảng cách từ điểm vào đến `stop_loss`.
- **Phí giao dịch thực tế**: Khấu trừ phí Taker/Maker ngay khi khớp lệnh giả lập.
- **Hệ thống Trailing Stop**: AI có khả năng cập nhật `stop_loss` và `phase` của lệnh trong mỗi lần được kích hoạt để khóa lợi nhuận.

---

## 5. Kết quả và Hậu kiểm (`data-backtest/`)

Mỗi lần chạy backtest sẽ tạo ra một thư mục riêng biệt chứa:
- `backtest_results.json`: Toàn bộ chỉ số hiệu suất (Return, Drawdown, Sharpe, Sortino).
- `trade_history.csv`: Chi tiết từng lần vào/ra lệnh, bao gồm cả lý do (`reason`) của AI.
- `ai_decisions.csv`: Nhật ký toàn bộ các quyết định Buy/Sell/Hold của DeepSeek.
- **Session Summary**: Gửi tóm tắt kết quả cuối cùng qua Telegram (nếu cấu hình) để theo dõi từ xa.

---
_Tài liệu cập nhật dựa trên phiên bản logic gộp dữ liệu và xử lý dataset khung Ngày._
