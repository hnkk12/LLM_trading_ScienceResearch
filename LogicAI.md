# Logic AI - Trading Bot (Tài liệu Kỹ thuật Chi tiết)

Tài liệu này trình bày chi tiết về kiến trúc hoạt động, cơ chế kích hoạt sự kiện (Event-Driven), cấu trúc dữ liệu đầu vào (Input), cấu trúc dữ liệu đầu ra (Output), và các quy tắc quản lý vốn/giao dịch của bot sử dụng mô hình ngôn ngữ lớn (LLM).

---

## 1. Kiến trúc Tổng quan & Cơ chế hoạt động

Bot hoạt động theo mô hình **Swing Trader kết hợp với Quản lý Rủi ro Chủ động (Active Guardian)**. AI đóng vai trò như một chuyên gia phân tích và ra quyết định, trong khi hệ thống core (Python) chịu trách nhiệm cung cấp dữ liệu thị trường chính xác, tính toán chỉ báo kỹ thuật, tính toán vị thế và thực thi lệnh.

### Vòng lặp hoạt động cơ bản (Workflow):
1. **Thu thập dữ liệu:** Bot đọc dữ liệu thị trường (từ file CSV khi Backtest hoặc từ API Binance khi chạy Live) theo khung thời gian **1D**.
2. **Kích hoạt sự kiện (Event-Driven AI):** Hệ thống core kiểm tra xem các điều kiện thị trường hiện tại có cần "đánh thức" AI dậy hay không. Nếu có, bot mới đóng gói dữ liệu và gọi LLM API.
3. **Phân tích và Phản hồi:** LLM nhận prompt chứa dữ liệu kỹ thuật và trạng thái tài khoản, thực hiện suy luận logic (Reasoning) và trả về định dạng JSON chứa quyết định.
4. **Thực thi giao dịch:** Core Python xử lý kết quả JSON từ AI để đặt lệnh mua/bán (Entry), đóng lệnh (Close), hoặc dịch chuyển Stop Loss/Take Profit (Hold/Trail).
5. **Ghi nhật ký và Báo cáo:** Cập nhật nhật ký vào `ai_decisions.csv`, `trade_history.csv` và đồng bộ lên Telegram.

---

## 2. Cơ chế Kích hoạt Sự kiện (Event-Driven AI Activation)

Để tối ưu hóa chi phí API và tránh lãng phí token khi thị trường đi ngang (Sideway), bot áp dụng cơ chế kích hoạt thông minh. AI sẽ **chỉ được gọi (Woken up)** khi xảy ra một trong các điều kiện sau:

* **Điều kiện A (Quản lý lệnh đang mở - Active Management):** Bất cứ khi nào tài khoản đang có vị thế mở (Open Position), AI sẽ được gọi ở mỗi cây nến mới để giám sát, dời Stop Loss (Trailing Stop) hoặc đóng vị thế sớm nếu cấu trúc thị trường thay đổi.
* **Điều kiện B (Biến động giá mạnh - Volatility Detection):** Giá đóng cửa của cây nến hiện tại thay đổi so với nến trước đó một khoảng $\ge 0.8\%$ cho bất kỳ tài sản nào được theo dõi.
* **Điều kiện C (Động lượng đảo chiều - RSI Extremes):** Chỉ báo RSI đi vào vùng cực hạn (RSI $< 35$ hoặc RSI $> 65$) cảnh báo cơ hội đảo chiều xu hướng.
* **Điều kiện D (Chu kỳ đặc biệt - Boundary Bars):** Ngày đầu tiên (để thiết lập chiến lược ban đầu) và ngày cuối cùng (để tất toán danh mục) của phiên chạy luôn kích hoạt AI.

> [!NOTE]
> Khi không có điều kiện nào được thỏa mãn, hệ thống core sẽ bỏ qua (Skip) việc gọi AI trên cây nến đó và tự động duy trì các lệnh TP/SL đã đặt trên sàn. Điều này giúp giảm tới 80% chi phí API.

---

## 3. Cấu trúc Dữ liệu Đầu vào (Input của AI)

Khi kích hoạt AI, hệ thống core sẽ tổng hợp thông tin thị trường và trạng thái tài khoản thành một Prompt dạng văn bản chi tiết gửi cho LLM. Dữ liệu đầu vào bao gồm:

### A. Chỉ thị Chiến lược (System Prompt)
Định nghĩa phong cách giao dịch của bot (Active Guardian, Sniper, hoặc Aggressive). Quy định các quy tắc quản lý vốn (Risk 1%), đòn bẩy cho phép, cách thức nhận diện phân vùng rủi ro (Green/Yellow/Red Zone) và cơ chế phòng ngự.

### B. Trạng thái Tài khoản (Portfolio State)
* **Available Cash (Số dư khả dụng):** Lượng tiền mặt hiện có để ký quỹ.
* **Total Equity (Tổng vốn tài sản):** $Available\ Cash + Ký\ quỹ\ hiện\ tại + PnL\ chưa\ thực\ hiện$.
* **Open Positions (Các vị thế đang chạy):**
  * Tên coin, hướng lệnh (Long/Short), khối lượng (Quantity), đòn bẩy (Leverage).
  * Giá vào lệnh (Entry Price), giá Stop Loss (SL) và Take Profit (TP) hiện tại.
  * Ký quỹ đã dùng (Margin), Phí giao dịch đã trả (Fees Paid).
  * PnL chưa thực hiện hiện tại ($) và tỷ lệ ROI (%).
  * Lịch sử dời stop loss (Trail History) và lý do của các lần dời trước.

### C. Dữ liệu Thị trường & Chỉ báo Kỹ thuật (Market Data)
* **OHLCV:** Giá Mở, Cao, Thấp, Đóng và Khối lượng giao dịch của nến hiện tại.
* **EMA (20, 50, 200):** Dùng để xác định xu hướng chính dài hạn và các vùng hỗ trợ/kháng cự động.
* **RSI (14):** Chỉ số sức mạnh tương đối xác định động lượng và vùng quá mua/quá bán.
* **MACD & MACD Signal:** Xác định sự giao cắt tín hiệu và động lượng dòng tiền.
* **ATR (14):** Đo lường biến động thị trường thực tế để tính toán khoảng cách SL/TP.
* **Volume Ratio:** Tỷ lệ khối lượng hiện tại so với khối lượng trung bình 20 phiên trước đó.

---

## 4. Cơ chế Vào lệnh & Quản trị Rủi ro (Money Management)

Bot thực thi các quy tắc quản trị vốn nghiêm ngặt nhằm tránh cháy tài khoản và tối ưu hóa lợi nhuận:

### A. Quy tắc Quản lý Vốn 1% (Strict 1% Risk Rule)
Số tiền tối đa chấp nhận thua lỗ cho mỗi lệnh giao dịch mới luôn được giới hạn chặt chẽ ở mức **1% trên Số dư khả dụng (Available Balance)**. 

### B. Công thức Tính toán Vị thế Tự động
Khối lượng vào lệnh (Quantity) và tiền ký quỹ yêu cầu (Margin Required) được hệ thống core tự động tính toán dựa trên mức Stop Loss mà AI đề xuất:

1. **Khoảng cách dừng lỗ (Stop Distance):**
   $$Stop\ Distance = |Entry\ Price - Stop\ Loss\ Price|$$
2. **Khối lượng giao dịch (Quantity):**
   $$Quantity = \frac{Available\ Balance \times 0.01}{Stop\ Distance}$$
3. **Giá trị vị thế (Position Value):**
   $$Position\ Value = Quantity \times Entry\ Price$$
4. **Tiền ký quỹ cần thiết (Margin Required):**
   $$Margin\ Required = \frac{Position\ Value}{Leverage}$$

> [!WARNING]
> Nếu giá trị $Margin\ Required + Phí\ vào\ lệnh$ vượt quá số dư khả dụng hiện tại, lệnh sẽ bị **từ chối (Rejected due to Insufficient Balance)** để đảm bảo an toàn.

### C. Hệ thống Phân vùng Trạng thái Thị trường (Market Risk Zones)
Dựa vào chỉ số biến động **Volatility Ratio (VR = ATR Ngắn hạn / ATR Dài hạn)**, hệ thống chia thị trường làm 3 vùng:
* **Green Zone (VR < 1.6 - Stable):** Thị trường ổn định. AI được phép vào lệnh năng nổ khi có tối thiểu **2 yếu tố hội tụ** (Confluence Factors).
* **Yellow Zone (VR 1.6 - 2.2 - Volatile):** Thị trường biến động cao. AI chỉ được phép vào lệnh khi có tối thiểu **3 yếu tố hội tụ** trở lên và tự động **giảm 50% khối lượng** vị thế (giảm mức Risk xuống 0.5%).
* **Red Zone (VR > 2.2 - Extreme):** Thị trường cực kỳ rủi ro hoặc tài khoản đang sụt giảm (Drawdown) vượt hạn mức. Hệ thống **khóa toàn bộ lệnh vào mới (Pause entries)** và chỉ tập trung quản lý dời SL/TP để bảo vệ các lệnh đang mở.

---

## 5. Cơ chế Quản lý Vị thế & Chốt Lệnh (Position & In-Trade Management)

Sau khi vị thế được mở, bot duy trì sự giám sát liên tục thông qua AI và cơ chế tự động của Core:

### A. Tự động Khớp lệnh SL/TP trên Sàn (Intrabar TP/SL - Không tốn phí API)
Tại mỗi cây nến mới, trước khi gọi AI, hệ thống core sẽ tự động kiểm tra mức biến động High/Low của cây nến:
* Nếu nến chạm/vượt giá **Stop Loss** của vị thế: Kích hoạt đóng vị thế ngay lập tức với lý do `"Stop loss hit"`.
* Nếu nến chạm/vượt giá **Take Profit** của vị thế: Kích hoạt đóng vị thế ngay lập tức với lý do `"Take profit hit"`.

### B. Dời Stop Loss Bảo vệ (Trailing Stop)
Khi AI nhận dữ liệu nến mới và ra quyết định `hold`, AI có thể đề xuất một mức Stop Loss mới tốt hơn (gần giá hiện tại hơn hoặc vượt qua giá Entry để bảo đảm hòa vốn/có lời):
* **Lưu ý quan trọng:** Hệ thống core chỉ chấp nhận giá Stop Loss mới nếu nó dịch chuyển theo hướng **giảm thiểu rủi ro** (tăng SL đối với lệnh Long, giảm SL đối với lệnh Short). Nếu AI đề xuất mức SL nới rộng rủi ro hoặc ngược hướng, core sẽ **bỏ qua việc cập nhật SL** để bảo vệ tài khoản khỏi cảm xúc giao dịch.

### C. Cơ chế Đóng nhanh (Fast Exit)
Nếu AI phát hiện cấu trúc thị trường đảo chiều sớm hoặc vị thế không sinh lời sau 3 cây nến, AI có thể ra quyết định `close` toàn bộ hoặc một phần vị thế (`close_quantity` / `close_fraction`) để giảm thiểu thiệt hại trước khi giá chạm Stop Loss cứng.

---

## 6. Cấu trúc Dữ liệu Đầu ra (Output của AI)

AI bắt buộc phải phản hồi dưới dạng một tài liệu JSON chuẩn duy nhất, đại diện cho quyết định giao dịch của từng đồng coin theo cấu trúc sau:

```json
{
  "BNB": {
    "signal": "entry|hold|close|reject",
    "side": "long|short",
    "quantity": 0.0,
    "profit_target": 0.0,
    "stop_loss": 0.0,
    "leverage": 10,
    "confidence": 0.85,
    "risk_usd": 10.0,
    "market_mode": "Active|Defensive|Pause",
    "justification": "Mô tả chi tiết lý do phân tích các chỉ báo kỹ thuật dẫn đến quyết định.",
    "confluence_tags": ["EMA Crossover", "RSI Support"],
    "trigger_tags": ["RSI Bounce"],
    "reasoning_categories": ["Trend Following"]
  }
}
```

### Chi tiết các trường dữ liệu:
* **`signal`**: Hành động được đề xuất.
  * `entry`: Mở vị thế mới (nếu chưa có vị thế mở cho coin này).
  * `close`: Đóng toàn bộ hoặc một phần vị thế hiện tại.
  * `hold`: Giữ nguyên vị thế, có thể cập nhật dời SL/TP hoặc chuyển pha (phase).
  * `reject`: Không có hành động nào (hoặc bỏ qua cơ hội tiềm ẩn do rủi ro cao).
* **`side`**: Hướng giao dịch (`long` hoặc `short`).
* **`quantity`**: Số lượng coin muốn giao dịch.
* **`profit_target`**: Mức giá mục tiêu để chốt lời (Take Profit).
* **`stop_loss`**: Mức giá dừng lỗ (Stop Loss).
* **`leverage`**: Đòn bẩy sử dụng (ví dụ: `5`, `10`).
* **`risk_usd`**: Số tiền (USD) chấp nhận mất nếu chạm SL (mặc định tương đương 1% balance).
* **`justification`**: Đoạn văn giải thích chi tiết các yếu tố kỹ thuật hỗ trợ cho quyết định giao dịch.
