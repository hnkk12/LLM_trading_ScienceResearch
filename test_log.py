import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add current dir to sys.path
sys.path.append(os.getcwd())

test_dir = Path("test_data_dir")
test_dir.mkdir(exist_ok=True)
os.environ["TRADEBOT_DATA_DIR"] = str(test_dir.resolve())

import bot

bot.init_csv_files()
bot.log_portfolio_state()

print(f"STATE_CSV: {bot.STATE_CSV}")
if bot.STATE_CSV.exists():
    with open(bot.STATE_CSV, 'r') as f:
        print(f"Content:\n{f.read()}")
else:
    print("File does not exist!")
