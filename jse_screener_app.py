def style_table(display_df, momentum_max_abs, sharpe_max_abs):
    styler = display_df.style

    styler = styler.map(
        lambda v: color_ratio(v, momentum_max_abs), subset=["6mo Momentum %"]
    )
    styler = styler.map(
        lambda v: color_ratio(v, sharpe_max_abs), subset=["Sharpe Ratio"]
    )
    for col in ["Valuation Score", "Momentum Score", "Sharpe Score", "Combined Score"]:
        styler = styler.map(color_score, subset=[col])

    styler = styler.format({
        "Price (R)": "{:.2f}",
        "P/E": "{:.2f}",
        "Market Cap (R bn)": "{:.2f}",
        "6mo Momentum %": "{:.1f}",
        "Sharpe Ratio": "{:.2f}",
        "Valuation Score": "{:.0f}",
        "Momentum Score": "{:.0f}",
        "Sharpe Score": "{:.0f}",
        "Combined Score": "{:.0f}",
    }, na_rep="—")

    return styler
