# TITANIUM Bot V6.0 🤖

TITANIUM Bot is an advanced, automated cryptocurrency trading bot designed for the spot market. It combines technical analysis, risk management protocols, and AI-powered insights to generate high-probability trading signals and manage positions effectively.

## 🚀 Key Features

### 🧠 Intelligent Trading Strategy

- **Hybrid Approach:** Combines Trend Following and Mean Reversion strategies.
- **Indicators:** Utilizes RSI, ADX, ATR, SMA, and EMA for precise market analysis.
- **Market Regime Detection:** Automatically detects Volatile vs. Ranging market conditions to adjust strategies.
- **Scoring System:** Evaluates trade setups using a weighted scoring algorithm.

### 🛡️ Robust Risk Management

- **Kill Switch:** Automatically halts trading during extreme market volatility or "Flash Crash" events.
- **Drawdown Monitor:** Tracks equity and adjusts position sizes or halts trading if drawdown limits are breached.
- **Daily Loss Limits:** Enforces daily PnL limits to prevent excessive losses.
- **Dynamic Position Sizing:** Adjusts position sizes based on volatility (ATR) and account equity.

### 💾 Reliability & Persistence

- **State Persistence:** robust `state_manager.py` ensures the bot can recover its state (signals, positions, cooldowns) after a restart or crash.
- **Graceful Shutdown:** Handles system signals (SIGINT, SIGTERM) to save state before exiting.

### 🤖 AI Integration

- **Gemini AI:** Integrates with Google's Gemini API for advanced market sentiment analysis and reporting.

### 📢 Notifications

- **Telegram Integration:** Sends real-time trade signals, daily performance reports, and critical alerts directly to a Telegram channel.

## 🛠️ Technology Stack

- **Language:** Python 3.13+
- **Data Analysis:** Pandas, NumPy
- **Exchange API:** CCXT (supports Binance, etc.)
- **Notifications:** python-telegram-bot
- **AI:** Google Gemini API

## 📦 Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/Telegram-Bot.git
    cd Telegram-Bot
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    Create a `.env` file (or set environment variables) with the following:
    ```env
    BOT_TOKEN=your_telegram_bot_token
    KANAL_ID=your_telegram_channel_id
    GEMINI_KEY=your_gemini_api_key
    ```

## 🚀 Usage

Start the bot:

```bash
python main.py
```

## 🧪 Testing

Run the comprehensive unit test suite:

```bash
pytest
```

Includes tests for:

- Indicators (RSI, ADX, ATR)
- Risk Management (Kill Switch, Drawdown)
- System State & Persistence

## 📝 License

[MIT License](LICENSE)
