# Logic AI - Trading Bot

Tài liệu này tóm tắt logic hoạt động, dữ liệu đầu vào (Input) và kết quả đầu ra (Output) của hệ thống AI trong bot giao dịch.

## 1. Tổng quan Logic AI

Hệ thống sử dụng các mô hình ngôn ngữ lớn (LLM) để đóng vai trò như một **Chuyên gia giao dịch (Expert Trader)**. Đối với bộ dữ liệu khung ngày (1D), AI tập trung vào việc phân tích xu hướng dài hạn và các chỉ báo động lượng để đưa ra quyết định giao dịch theo phong cách **Swing Trading**.

### Các thành phần chính:

- **Phân tích Xu hướng Ngày (Daily Analysis):**
  - Sử dụng các chỉ báo EMA (20, 50, 200) để xác định cấu trúc thị trường dài hạn.
  - Phân tích RSI và MACD để tìm điểm đảo chiều hoặc tiếp diễn xu hướng trên khung ngày.
- **Vòng lặp phản hồi (Feedback Loop):** AI học từ các kết quả giao dịch gần nhất để điều chỉnh mức độ chấp nhận rủi ro.
- **Quản trị rủi ro:** Thực thi quy tắc rủi ro 1% và tính toán Stop Loss dựa trên biến động thực tế của tài sản.

---

## 2. Dữ liệu Đầu vào (Input)

Dữ liệu được bot thu thập từ các tệp dữ liệu lịch sử (CSV) trong thư mục `dataset/` và định dạng thành một Prompt chi tiết gửi cho AI.

### A. Dữ liệu Thị trường (Market Data):

- **Khung thời gian Backtest:** **1D (Daily)**.
- **Nguồn dữ liệu:** Các tệp CSV lưu trữ dữ liệu lịch sử (ví dụ: `AAPL_D_2022_2023.csv`, `gold_D_2020_2021.csv`).
- **Giá:** Open, High, Low, Close (OHLC).
- **Chỉ báo kỹ thuật (được tính toán từ dữ liệu CSV):**
  - **EMA (20, 50, 200):** Xác định xu hướng và hỗ trợ/kháng cự động.
  - **RSI:** Xác định vùng quá mua/quá bán và sức mạnh động lượng.
  - **MACD:** Xác định sự giao cắt và phân kỳ.
  - **ATR:** Đo lường biến động để đặt Stop Loss/Take Profit.
  - **ADX:** Đo lường sức mạnh của xu hướng.
- **Cấu trúc & Khối lượng:**
  - Swing High/Low trong 10-20 nến gần nhất.
  - Volume Ratio (so sánh khối lượng hiện tại với trung bình).

### B. Trạng thái Tài khoản (Portfolio State):

- Số dư khả dụng (Available Balance).
- Tổng vốn (Total Equity).
- Các vị thế đang mở (Open Positions): Giá vào, khối lượng, PnL hiện tại, Stop Loss/Take Profit hiện tại.
- Tỉ lệ R-multiple: Đo lường hiệu quả lợi nhuận trên rủi ro của lệnh đang chạy.

### C. Chỉ thị Chiến lược (System Prompt):

- Quy tắc quản lý vốn (Risk per trade).
- Phong cách giao dịch (Aggressive, Sniper, hoặc Defensive).
- Điều kiện dừng giao dịch (Crisis management).

---

## 3. Quản lý Vốn và Rủi ro (Money Management)

Hệ thống áp dụng các quy tắc toán học chặt chẽ để bảo vệ tài khoản trước các chuỗi thua lỗ:

### A. Quản lý Vốn (Capital Management):

- **Quy tắc 1% Risk:** Mỗi vị thế chỉ được phép rủi ro tối đa **1%** trên số dư khả dụng (Available Balance). Ví dụ: Tài khoản $10,000 thì mức lỗ tối đa cho một lệnh là $100.
- **Tính toán Quantity tự động:** AI và bot tự động tính toán khối lượng lệnh (Quantity) dựa trên khoảng cách từ điểm vào đến Stop Loss sao cho số tiền mất đúng bằng mức rủi ro cho phép.
- **Đòn bẩy linh hoạt:** Sử dụng đòn bẩy từ 5x - 10x để tối ưu hóa hiệu quả sử dụng vốn nhưng vẫn đảm bảo an toàn.

### B. Quản lý Rủi ro (Risk Management):

- **Tỉ lệ Reward/Risk (R/R):** Luôn duy trì mức R/R tối thiểu là 1:1 hoặc cao hơn. Ưu tiên các thiết lập có tiềm năng lợi nhuận lớn hơn rủi ro.
- **Stop Loss Động (Trailing Stop):** Tự động dời Stop Loss về điểm hòa vốn (Breakeven) hoặc theo xu hướng (ATR-based) khi lệnh đã có lợi nhuận để bảo vệ thành quả.
- **Lệnh đóng nhanh (Fast Exit):** Tự động đóng lệnh hoặc siết chặt SL nếu giá không chạy đúng kỳ vọng trong vòng 3 cây nến hoặc có tín hiệu đảo chiều sớm.

---

## 4. Cơ chế Phòng ngự (Defense Mechanisms)

Hệ thống sử dụng chiến thuật phòng thủ đa lớp dựa trên biến động thị trường:

- **Phân vùng Trạng thái (Market Zones):**
  - **Green Zone (Stable):** Thị trường ổn định, AI được phép chủ động tìm kiếm nhiều cơ hội vào lệnh.
  - **Yellow Zone (Volatile):** Thị trường biến động cao, AI tăng rào cản xác nhận (cần nhiều chỉ báo đồng thuận hơn) và giảm 50% khối lượng lệnh.
  - **Red Zone (Extreme):** Thị trường cực kỳ rủi ro hoặc sụt giảm tài khoản (Drawdown) vượt mức cho phép, hệ thống sẽ tạm dừng vào lệnh mới và chỉ tập trung bảo vệ các lệnh đang mở.
- **Chống đuổi giá (Anti-Chasing):** Hệ thống từ chối vào lệnh nếu giá đã chạy quá xa (hơn 2 lần ATR) so với điểm phát tín hiệu ban đầu.
- **Bảo vệ sụt giảm (Drawdown Protection):** Nếu tài khoản sụt giảm quá một tỉ lệ nhất định trong tuần (ví dụ 3%), hệ thống tự động chuyển sang chế độ phòng thủ nghiêm ngặt nhất.

---

## 5. Kết quả Đầu ra (Output)

Hệ thống cung cấp kết quả qua ba kênh chính: JSON phản hồi từ AI, Thông báo Telegram và File kết quả Backtest.

### A. JSON phản hồi từ AI (Thực thi lệnh):

| Trường dữ liệu  | Mô tả                                          |
| :-------------- | :--------------------------------------------- |
| `signal`        | Hành động: `entry`, `hold`, `close`, `reject`. |
| `side`          | Hướng giao dịch: `long` hoặc `short`.          |
| `quantity`      | Khối lượng coin/tài sản cần giao dịch.         |
| `leverage`      | Đòn bẩy sử dụng (ví dụ: 5x, 10x).              |
| `profit_target` | Mức giá mục tiêu chốt lời.                     |
| `stop_loss`     | Mức giá dừng lỗ.                               |
| `risk_usd`      | Số tiền (USD) chấp nhận rủi ro.                |
| `justification` | Giải thích lý do phân tích MTA.                |

### B. Thông báo Telegram (Theo dõi thời gian thực):

- **Portfolio Summary:** Số dư (Balance), Tổng vốn (Equity), PnL chưa ghi nhận (Unrealized PnL).
- **Hiệu suất:** Tỉ lệ thắng (Win Rate), Profit Factor, Recovery Factor, Max Drawdown (%).
- **Chỉ số nâng cao:** Sortino Ratio (đo lường lợi nhuận trên rủi ro giảm giá).
- **Trạng thái vị thế:** Danh sách các lệnh đang mở và PnL tương ứng.

### C. File kết quả Backtest (`backtest_results.json`):

- **Capital Stats:** Vốn ban đầu, Vốn cuối cùng, Tổng lợi nhuận ròng, % Lợi nhuận.
- **Trading Stats:** Tổng số lệnh, Số lệnh thắng/thua, Chuỗi thắng/thua liên tiếp dài nhất.
- **Risk Metrics:** Sharpe Ratio, Sortino Ratio, Max Drawdown, Recovery Factor.
- **Holding Time:** Thời gian giữ lệnh trung bình.

---

## 6. Quy trình hoạt động (Workflow)

1.  **Thu thập:** Bot đọc dữ liệu từ các tệp CSV trong thư mục `dataset/` theo khung thời gian **1D**.
2.  **Đóng gói:** Bot tổng hợp dữ liệu lịch sử + trạng thái tài khoản giả lập thành 1 văn bản (Prompt).
3.  **Suy luận:** Gửi Prompt sang AI (DeepSeek/Gemini/Llama).
4.  **Thực thi:**
    - Nếu `signal` là `entry`, bot giả lập lệnh mua/bán trong môi trường backtest.
    - Nếu `signal` là `close`, bot đóng vị thế giả lập.
5.  **Ghi nhật ký:** Lưu quyết định và lý do của AI vào file `ai_decisions.csv` và xuất báo cáo tổng hợp ra `backtest_results.json`.
