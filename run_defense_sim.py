# Demo thresholds; adjust these for your feature scales.
VOLATILITY_HIGH_THRESHOLD = 0.20
BUY_SELL_RATIO_LOW_THRESHOLD = 0.50


def check_market_status(current_volatility, current_buy_sell_ratio):
    print(
        f"Checking market: volatility={current_volatility}, "
        f"buy/sell ratio={current_buy_sell_ratio}"
    )

    if (
        current_volatility > VOLATILITY_HIGH_THRESHOLD
        and current_buy_sell_ratio < BUY_SELL_RATIO_LOW_THRESHOLD
    ):
        print(
            "🚨 CRASH PREDICTED! EXECUTING KILL SWITCH: ALL OPEN ORDERS "
            "CANCELED. TRADING PAUSED FOR 15 MINUTES. 🚨"
        )
    else:
        print("✅ Market stable. Continuing normal operations.")


if __name__ == "__main__":
    check_market_status(current_volatility=0.05, current_buy_sell_ratio=1.20)
    check_market_status(current_volatility=0.35, current_buy_sell_ratio=0.20)