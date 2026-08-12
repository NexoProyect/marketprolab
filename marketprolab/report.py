"""Standalone HTML reports for backtests, optimizations, walk-forward and Monte Carlo.

Everything is inlined - charts are embedded as base64 PNGs and the CSS lives in
the file - so a report is a single ``.html`` you can email, archive or open
offline. Nothing is fetched from the network.

::

    result.to_html("report.html")
    opt.to_html("optimization.html")
    combined_report(result, opt=opt, mc=mc, path="overall.html")
"""

from __future__ import annotations

import base64
import html
import io
import os
import webbrowser
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import plotting

__all__ = [
    "backtest_report",
    "optimization_report",
    "walk_forward_report",
    "montecarlo_report",
    "comparison_report",
    "combined_report",
]

# --------------------------------------------------------------------- styling
_CSS_LIGHT = {
    "bg": "#F6F6F4", "surface": "#FFFFFF", "ink": "#1C1B1A", "soft": "#57534E",
    "muted": "#8A857F", "line": "#E7E5E1", "accent": "#3B6FE0",
    "pos": "#0F8F7A", "neg": "#C0392B", "shadow": "0 1px 2px rgba(0,0,0,.05)",
}
_CSS_DARK = {
    "bg": "#141413", "surface": "#1F1F1E", "ink": "#F2F0ED", "soft": "#B8B3AD",
    "muted": "#88837D", "line": "#343432", "accent": "#4F83E8",
    "pos": "#0F9C80", "neg": "#D45B4E", "shadow": "0 1px 2px rgba(0,0,0,.4)",
}


def _stylesheet(mode: str) -> str:
    c = _CSS_DARK if mode == "dark" else _CSS_LIGHT
    return f"""
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px 20px 64px;
  background: {c['bg']}; color: {c['ink']};
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
header {{ margin-bottom: 28px; }}
h1 {{ font-size: 26px; font-weight: 650; margin: 0 0 6px; letter-spacing: -.01em; }}
h2 {{ font-size: 16px; font-weight: 620; margin: 34px 0 12px; letter-spacing: -.005em; }}
.sub {{ color: {c['muted']}; font-size: 13px; margin: 0; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 12px;
         color: {c['soft']}; font-size: 12.5px; }}
.meta b {{ color: {c['ink']}; font-weight: 600; }}
.cards {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); }}
.card {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 10px;
         padding: 14px 16px; box-shadow: {c['shadow']}; }}
.card .label {{ color: {c['muted']}; font-size: 11.5px; text-transform: uppercase;
                letter-spacing: .05em; margin-bottom: 6px; }}
.card .value {{ font-size: 21px; font-weight: 650; font-variant-numeric: tabular-nums;
                letter-spacing: -.02em; }}
.card .note {{ color: {c['muted']}; font-size: 11.5px; margin-top: 4px; }}
.pos {{ color: {c['pos']}; }}
.neg {{ color: {c['neg']}; }}
figure {{ margin: 0 0 18px; background: {c['surface']}; border: 1px solid {c['line']};
          border-radius: 10px; padding: 12px; box-shadow: {c['shadow']}; }}
figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
figcaption {{ color: {c['muted']}; font-size: 12px; margin-top: 8px; }}
.tablebox {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 10px;
             overflow-x: auto; box-shadow: {c['shadow']}; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
         font-variant-numeric: tabular-nums; }}
th, td {{ padding: 8px 12px; text-align: right; white-space: nowrap;
          border-bottom: 1px solid {c['line']}; }}
th {{ position: sticky; top: 0; background: {c['surface']}; color: {c['muted']};
      font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: .04em; }}
td:first-child, th:first-child {{ text-align: left; }}
tbody tr:last-child td {{ border-bottom: none; }}
pre {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 10px;
       padding: 16px; overflow-x: auto; font-size: 12px; line-height: 1.5;
       font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.kv {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 20px;
       font-size: 12.5px; }}
.kv dt {{ color: {c['muted']}; }}
.kv dd {{ margin: 0; }}
footer {{ margin-top: 46px; padding-top: 16px; border-top: 1px solid {c['line']};
          color: {c['muted']}; font-size: 12px; }}
@media (max-width: 640px) {{ body {{ padding: 20px 12px 48px; }} h1 {{ font-size: 21px; }} }}
"""


# ---------------------------------------------------------------- HTML helpers
def _esc(value) -> str:
    return html.escape(str(value))


def _fig_to_img(fig, dpi: int = 120, close: bool = True) -> str:
    """Render a matplotlib figure into an inline base64 ``<img>`` payload."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure(fig, caption: str = "", dpi: int = 120) -> str:
    if fig is None:
        return ""
    src = _fig_to_img(fig, dpi=dpi)
    cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    return f'<figure><img alt="{_esc(caption or "chart")}" src="{src}">{cap}</figure>'


def _card(label: str, value, note: str = "", good: Optional[bool] = None) -> str:
    cls = "" if good is None else (" pos" if good else " neg")
    note_html = f'<div class="note">{_esc(note)}</div>' if note else ""
    return (f'<div class="card"><div class="label">{_esc(label)}</div>'
            f'<div class="value{cls}">{_esc(value)}</div>{note_html}</div>')


def _cards(items: Sequence[tuple]) -> str:
    return '<div class="cards">' + "".join(_card(*item) for item in items) + "</div>"


def _table(df: pd.DataFrame, max_rows: Optional[int] = None, index: bool = False,
           float_format: str = "{:,.4g}") -> str:
    if df is None or len(df) == 0:
        return '<div class="tablebox"><table><tbody><tr><td>No data</td></tr></tbody></table></div>'
    frame = df.head(max_rows) if max_rows else df
    body = frame.to_html(
        index=index, border=0, escape=True, na_rep="-",
        float_format=lambda v: float_format.format(v),
    )
    more = ""
    if max_rows and len(df) > max_rows:
        more = f'<figcaption>Showing {max_rows:,} of {len(df):,} rows.</figcaption>'
    return f'<div class="tablebox">{body}</div>{more}'


def _kv(pairs: Dict[str, Any]) -> str:
    rows = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in pairs.items())
    return f'<dl class="kv">{rows}</dl>'


def _page(title: str, subtitle: str, meta: Sequence[tuple], body: str,
          mode: str = "light") -> str:
    meta_html = "".join(f"<span>{_esc(k)}: <b>{_esc(v)}</b></span>" for k, v in meta)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_stylesheet(mode)}</style>
</head><body><div class="wrap">
<header>
  <h1>{_esc(title)}</h1>
  <p class="sub">{_esc(subtitle)}</p>
  <div class="meta">{meta_html}</div>
</header>
{body}
<footer>Generated by marketprolab on {stamp}. All charts and styles are embedded;
this file works offline.</footer>
</div></body></html>
"""


def _write(path: str, content: str, open_browser: bool = False) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(path)}")
    return os.path.abspath(path)


def _fmt(value, spec: str = ",.2f") -> str:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "-"
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


# ------------------------------------------------------------ backtest report
def _backtest_sections(result, charts: bool = True, max_trades: int = 300,
                       theme: str = "light", dpi: int = 120) -> str:
    s = result.stats
    previous = plotting._ACTIVE
    plotting.set_theme(theme)
    try:
        blocks: List[str] = []

        blocks.append("<h2>Headline</h2>")
        blocks.append(_cards([
            ("Net profit", _fmt(s["net_profit"]), f"{_fmt(s['return_pct'])}% return",
             s["net_profit"] >= 0),
            ("Final equity", _fmt(s["final_equity"]),
             f"from {_fmt(s['initial_balance'])}", s["final_equity"] >= s["initial_balance"]),
            ("Max drawdown", f"{_fmt(s['max_dd_pct'])}%", _fmt(s["max_dd_abs"]), False),
            ("Profit factor", _fmt(s["profit_factor"]), f"{s['trades']:,} trades",
             s["profit_factor"] >= 1),
            ("Sharpe", _fmt(s["sharpe"]), f"Sortino {_fmt(s['sortino'])}", s["sharpe"] >= 0),
            ("Win rate", f"{_fmt(s['win_rate'])}%", f"{s['wins']}/{s['trades']}",
             s["win_rate"] >= 50),
            ("Expectancy", _fmt(s["expectancy"]), "per trade", s["expectancy"] >= 0),
            ("CAGR", f"{_fmt(s['cagr_pct'])}%", f"Calmar {_fmt(s['calmar'])}",
             s["cagr_pct"] >= 0),
        ]))

        if charts:
            blocks.append("<h2>Equity and drawdown</h2>")
            blocks.append(_figure(result.plot_equity(),
                                  "Equity, balance and the underwater curve.", dpi))

        blocks.append("<h2>All metrics</h2>")
        blocks.append(_metric_table(s))

        if charts:
            blocks.append("<h2>Trade analysis</h2>")
            blocks.append(_figure(result.plot_distribution(),
                                  "Outcome distribution, cumulative P&L, holding time, MAE/MFE.",
                                  dpi))
            blocks.append(_figure(result.plot_trades(),
                                  "Entries and exits on the price series.", dpi))

        monthly = result.monthly_table()
        if not monthly.empty:
            blocks.append("<h2>Monthly returns (%)</h2>")
            blocks.append(_table(monthly.round(2), index=True, float_format="{:,.2f}"))

        blocks.append("<h2>Setup</h2>")
        spec = result.spec
        blocks.append(_kv({
            "Symbol": f"{spec.symbol} ({spec.category or spec.asset_type})",
            "Digits / contract size": f"{spec.digits} / {spec.contract_size:g}",
            "Currencies": f"margin {spec.margin_currency}, profit {spec.profit_currency}",
            "Spread": f"{spec.spread_points:g} points "
                      f"({'floating' if spec.spread_float else 'fixed'})",
            "Commission": f"{spec.commission_per_lot:g}/lot + {spec.commission_per_deal:g}/deal"
                          f" + {spec.commission_percent:g}%",
            "Swap": f"{spec.swap_type.value}: long {spec.swap_long:g}, "
                    f"short {spec.swap_short:g}",
            "Leverage": f"1:{spec.leverage:g}",
            "Broker profile": f"{result.profile.name} "
                              f"(stop out {result.profile.stop_out_level:g}%, "
                              f"{result.profile.margin_mode.value})",
            "Strategy": f"{type(result.strategy).__name__} {result.params}",
            "Simulation": f"intrabar {result.config.intrabar.value}, "
                          f"sessions {'on' if result.config.respect_sessions else 'off'}, "
                          f"seed {result.config.seed}",
        }))

        if not result.trades.empty:
            blocks.append("<h2>Trades</h2>")
            columns = ["ticket", "type", "volume", "open_time", "open_price", "close_time",
                       "close_price", "net_profit", "commission", "swap", "reason", "bars_held"]
            available = [c for c in columns if c in result.trades.columns]
            blocks.append(_table(result.trades[available], max_rows=max_trades,
                                 float_format="{:,.4f}"))

        rejections = result.rejections
        if len(rejections):
            counts = rejections["reason"].value_counts().rename_axis("reason")
            blocks.append("<h2>Rejected orders</h2>")
            blocks.append(_table(counts.reset_index(name="count")))

        return "".join(blocks)
    finally:
        plotting.set_theme(previous)


def _metric_table(stats: Dict[str, Any]) -> str:
    groups = {
        "Account": ["initial_balance", "final_equity", "net_profit", "return_pct", "cagr_pct",
                    "peak_equity", "min_equity"],
        "Risk": ["max_dd_abs", "max_dd_pct", "max_dd_duration", "recovery_factor", "sharpe",
                 "sortino", "calmar", "sqn", "ulcer_index", "volatility_annual_pct",
                 "exposure_pct", "kelly_pct"],
        "Trades": ["trades", "wins", "losses", "win_rate", "profit_factor", "expectancy",
                   "avg_win", "avg_loss", "payoff_ratio", "largest_win", "largest_loss",
                   "max_win_streak", "max_loss_streak", "avg_bars_held", "avg_duration",
                   "trades_per_month", "long_trades", "short_trades", "long_win_rate",
                   "short_win_rate", "long_profit", "short_profit"],
        "Costs": ["total_commission", "total_swap", "total_costs", "avg_slippage_points",
                  "avg_mae", "avg_mfe"],
    }
    rows = []
    for group, keys in groups.items():
        for key in keys:
            if key not in stats:
                continue
            value = stats[key]
            if isinstance(value, float):
                value = _fmt(value)
            rows.append({"Group": group, "Metric": key.replace("_", " "), "Value": value})
    return _table(pd.DataFrame(rows))


def backtest_report(result, path: str = "backtest_report.html", title: Optional[str] = None,
                    charts: bool = True, max_trades: int = 300, theme: str = "light",
                    dpi: int = 120, open_browser: bool = False) -> str:
    """Write a full HTML report for one backtest and return the file path."""
    s = result.stats
    subtitle = (f"{result.spec.symbol} · {s['start']} to {s['end']} · "
                f"{s['bars']:,} bars · {type(result.strategy).__name__}")
    meta = [
        ("Broker", result.profile.name),
        ("Net profit", _fmt(s["net_profit"])),
        ("Return", f"{_fmt(s['return_pct'])}%"),
        ("Max DD", f"{_fmt(s['max_dd_pct'])}%"),
        ("Trades", f"{s['trades']:,}"),
        ("Runtime", f"{result.elapsed:.2f}s"),
    ]
    body = _backtest_sections(result, charts=charts, max_trades=max_trades,
                              theme=theme, dpi=dpi)
    page = _page(title or f"Backtest - {result.spec.symbol}", subtitle, meta, body, theme)
    return _write(path, page, open_browser)


# --------------------------------------------------------- optimization report
def optimization_report(opt, path: str = "optimization_report.html",
                        title: Optional[str] = None, top: int = 50, charts: bool = True,
                        theme: str = "light", dpi: int = 120,
                        open_browser: bool = False) -> str:
    """Write an HTML report for a grid or random search."""
    previous = plotting._ACTIVE
    plotting.set_theme(theme)
    try:
        df = opt.results
        metric = opt.metric
        blocks: List[str] = []

        best = df.iloc[0] if len(df) else {}
        blocks.append("<h2>Best combination</h2>")
        blocks.append(_cards([
            ("Combinations", f"{len(df):,}", f"objective: {metric}"),
            (f"Best {metric}", _fmt(best.get(metric)), str(opt.best_params)),
            ("Net profit", _fmt(best.get("net_profit")), "", best.get("net_profit", 0) >= 0),
            ("Max drawdown", f"{_fmt(best.get('max_dd_pct'))}%", "", False),
            ("Profit factor", _fmt(best.get("profit_factor")), "",
             best.get("profit_factor", 0) >= 1),
            ("Trades", f"{int(best.get('trades', 0)):,}"),
            ("Elapsed", f"{opt.elapsed:.1f}s"),
        ]))

        if charts and len(df):
            blocks.append("<h2>Parameter surface</h2>")
            blocks.append(_figure(
                opt.plot(),
                "Heatmap for two parameters, bars for one. The circle marks the best point.",
                dpi,
            ))

        blocks.append("<h2>Stability</h2>")
        blocks.append(
            "<p class='sub'>Ranked by the average score of each combination's neighbourhood "
            "in the grid. An isolated peak is usually overfitting; a setting surrounded by "
            "good neighbours is far more likely to survive live.</p>"
        )
        try:
            stable = opt.stable_top(min(top, 20))
            blocks.append(_table(stable))
        except Exception as exc:  # pragma: no cover - defensive
            blocks.append(f"<p class='sub'>Could not compute neighbourhood scores: {_esc(exc)}</p>")

        blocks.append(f"<h2>Top {min(top, len(df))} combinations</h2>")
        blocks.append(_table(df, max_rows=top))

        if opt.best_result is not None:
            blocks.append("<h2>Best combination in detail</h2>")
            blocks.append(_backtest_sections(opt.best_result, charts=charts, max_trades=100,
                                             theme=theme, dpi=dpi))

        subtitle = (f"{len(df):,} combinations over "
                    f"{', '.join(opt.param_names) or 'no parameters'} · objective {metric}")
        meta = [
            ("Objective", metric),
            ("Best", str(opt.best_params)),
            ("Elapsed", f"{opt.elapsed:.1f}s"),
        ]
        page = _page(title or "Optimization report", subtitle, meta, "".join(blocks), theme)
        return _write(path, page, open_browser)
    finally:
        plotting.set_theme(previous)


# --------------------------------------------------------- walk-forward report
def walk_forward_report(wf, path: str = "walkforward_report.html",
                        title: Optional[str] = None, charts: bool = True,
                        theme: str = "light", dpi: int = 120,
                        open_browser: bool = False) -> str:
    """Write an HTML report for a walk-forward run."""
    previous = plotting._ACTIVE
    plotting.set_theme(theme)
    try:
        s = wf.stats
        blocks = ["<h2>Out-of-sample summary</h2>", _cards([
            ("Windows", f"{len(wf.windows):,}"),
            ("Efficiency", _fmt(wf.efficiency), "OOS / IS", wf.efficiency >= 0.5),
            ("OOS net profit", _fmt(s.get("net_profit")), "",
             s.get("net_profit", 0) >= 0),
            ("OOS max drawdown", f"{_fmt(s.get('max_dd_pct'))}%", "", False),
            ("OOS profit factor", _fmt(s.get("profit_factor")), "",
             s.get("profit_factor", 0) >= 1),
            ("OOS trades", f"{int(s.get('trades', 0)):,}"),
        ])]
        blocks.append(
            "<p class='sub'>Efficiency is the average ratio between out-of-sample and "
            "in-sample performance. Above 0.5 is decent; below 0.3 suggests the optimization "
            "is fitting noise.</p>"
        )

        if charts:
            blocks.append("<h2>Window by window</h2>")
            blocks.append(_figure(wf.plot(),
                                  "In-sample vs out-of-sample per window, and the chained "
                                  "out-of-sample equity curve.", dpi))

        blocks.append("<h2>Windows</h2>")
        blocks.append(_table(wf.windows))

        if not wf.oos_trades.empty:
            blocks.append("<h2>Out-of-sample trades</h2>")
            columns = [c for c in ["window", "type", "volume", "open_time", "close_time",
                                   "net_profit", "reason"] if c in wf.oos_trades.columns]
            blocks.append(_table(wf.oos_trades[columns], max_rows=300))

        subtitle = f"{len(wf.windows)} windows · objective {wf.metric}"
        meta = [("Efficiency", _fmt(wf.efficiency)),
                ("OOS net", _fmt(s.get("net_profit"))),
                ("OOS trades", f"{int(s.get('trades', 0)):,}")]
        page = _page(title or "Walk-forward report", subtitle, meta, "".join(blocks), theme)
        return _write(path, page, open_browser)
    finally:
        plotting.set_theme(previous)


# ---------------------------------------------------------- Monte Carlo report
def montecarlo_report(mc, path: str = "montecarlo_report.html",
                      title: Optional[str] = None, charts: bool = True,
                      theme: str = "light", dpi: int = 120,
                      open_browser: bool = False) -> str:
    """Write an HTML report for a Monte Carlo simulation."""
    previous = plotting._ACTIVE
    plotting.set_theme(theme)
    try:
        p = mc.percentiles
        blocks = ["<h2>Outcome range</h2>", _cards([
            ("Scenarios", f"{mc.n_simulations:,}", f"method: {mc.method}"),
            ("Median final equity", _fmt(p["final_p50"]),
             f"from {_fmt(mc.initial_balance)}", p["final_p50"] >= mc.initial_balance),
            ("Final equity P05", _fmt(p["final_p05"]), "bad tail",
             p["final_p05"] >= mc.initial_balance),
            ("Final equity P95", _fmt(p["final_p95"]), "good tail", True),
            ("Median drawdown", f"{_fmt(p['dd_p50'])}%", "", False),
            ("Drawdown P95", f"{_fmt(p['dd_p95'])}%", "the one to plan for", False),
            ("Probability of loss", f"{mc.probability_of_loss * 100:.1f}%", "",
             mc.probability_of_loss < 0.5),
            ("Risk of ruin", f"{mc.risk_of_ruin * 100:.1f}%", "",
             mc.risk_of_ruin < 0.05),
        ])]

        if charts:
            blocks.append("<h2>Simulated paths</h2>")
            blocks.append(_figure(mc.plot(),
                                  "Percentile fan, final equity and maximum drawdown "
                                  "distributions.", dpi))

        blocks.append("<h2>Percentiles</h2>")
        table = pd.DataFrame(
            {"metric": list(p.keys()), "value": [_fmt(v) for v in p.values()]}
        )
        blocks.append(_table(table))

        from .montecarlo import required_capital

        blocks.append("<h2>Suggested capital</h2>")
        rows = [required_capital(mc, confidence=c, safety_factor=f)
                for c, f in ((95, 1.5), (99, 2.0))]
        blocks.append(_table(pd.DataFrame(rows), float_format="{:,.2f}"))

        subtitle = (f"{mc.n_simulations:,} scenarios · method {mc.method} · "
                    f"drawdown threshold {mc.dd_threshold_pct:.0f}%")
        meta = [("Median equity", _fmt(p["final_p50"])),
                ("DD P95", f"{_fmt(p['dd_p95'])}%"),
                ("Risk of ruin", f"{mc.risk_of_ruin * 100:.1f}%")]
        page = _page(title or "Monte Carlo report", subtitle, meta, "".join(blocks), theme)
        return _write(path, page, open_browser)
    finally:
        plotting.set_theme(previous)


# ---------------------------------------------------------- comparison report
def comparison_report(results: Dict[str, Any], path: str = "comparison_report.html",
                      title: Optional[str] = None, charts: bool = True,
                      theme: str = "light", dpi: int = 120,
                      open_browser: bool = False) -> str:
    """Compare several backtests side by side in one HTML page.

    ::

        comparison_report({"baseline": r1, "with filter": r2}, "compare.html")
    """
    from .optimize import compare

    previous = plotting._ACTIVE
    plotting.set_theme(theme)
    try:
        table = compare(results)
        blocks = ["<h2>Side by side</h2>", _table(table.reset_index().rename(
            columns={"index": "run"}))]

        if charts:
            import matplotlib.pyplot as plt

            t = plotting.theme()
            fig, ax = plt.subplots(figsize=(11, 5))
            for k, (name, result) in enumerate(results.items()):
                curve = result.equity_curve
                ax.plot(curve.index, curve.values,
                        color=t["series"][k % len(t["series"])], label=name)
            ax.set_ylabel("Equity")
            ax.legend(loc="best")
            plotting._dates(ax)
            plotting._finish(fig, "Equity curves", f"{len(results)} runs")
            blocks.insert(0, _figure(fig, "Equity curve of every run.", dpi))

        for name, result in results.items():
            blocks.append(f"<h2>{_esc(name)}</h2>")
            blocks.append(_cards([
                ("Net profit", _fmt(result.stats["net_profit"]), "",
                 result.stats["net_profit"] >= 0),
                ("Max drawdown", f"{_fmt(result.stats['max_dd_pct'])}%", "", False),
                ("Profit factor", _fmt(result.stats["profit_factor"]), "",
                 result.stats["profit_factor"] >= 1),
                ("Sharpe", _fmt(result.stats["sharpe"]), "", result.stats["sharpe"] >= 0),
                ("Trades", f"{result.stats['trades']:,}"),
            ]))

        page = _page(title or "Backtest comparison",
                     f"{len(results)} runs compared", [("Runs", len(results))],
                     "".join(blocks), theme)
        return _write(path, page, open_browser)
    finally:
        plotting.set_theme(previous)


# ------------------------------------------------------------ combined report
def combined_report(result, path: str = "overall_report.html", opt=None, wf=None, mc=None,
                    title: Optional[str] = None, charts: bool = True, max_trades: int = 200,
                    theme: str = "light", dpi: int = 120, open_browser: bool = False) -> str:
    """One page with the whole story: backtest, optimization, walk-forward and Monte Carlo.

    Any of ``opt``, ``wf`` and ``mc`` may be omitted; only the sections you pass
    are rendered.

    ::

        combined_report(result, "overall.html", opt=opt, wf=wf, mc=mc)
    """
    previous = plotting._ACTIVE
    plotting.set_theme(theme)
    try:
        blocks = ["<h2>Backtest</h2>",
                  _backtest_sections(result, charts=charts, max_trades=max_trades,
                                     theme=theme, dpi=dpi)]

        if opt is not None:
            blocks.append("<h2>Optimization</h2>")
            blocks.append(_cards([
                ("Combinations", f"{len(opt.results):,}", f"objective: {opt.metric}"),
                (f"Best {opt.metric}", _fmt(opt.results.iloc[0][opt.metric]),
                 str(opt.best_params)),
            ]))
            if charts:
                blocks.append(_figure(opt.plot(), "Parameter surface.", dpi))
            blocks.append(_table(opt.results, max_rows=25))

        if wf is not None:
            blocks.append("<h2>Walk-forward</h2>")
            blocks.append(_cards([
                ("Windows", f"{len(wf.windows):,}"),
                ("Efficiency", _fmt(wf.efficiency), "OOS / IS", wf.efficiency >= 0.5),
                ("OOS net profit", _fmt(wf.stats.get("net_profit")), "",
                 wf.stats.get("net_profit", 0) >= 0),
            ]))
            if charts:
                blocks.append(_figure(wf.plot(), "In-sample vs out-of-sample.", dpi))
            blocks.append(_table(wf.windows, max_rows=40))

        if mc is not None:
            blocks.append("<h2>Monte Carlo</h2>")
            p = mc.percentiles
            blocks.append(_cards([
                ("Scenarios", f"{mc.n_simulations:,}", f"method: {mc.method}"),
                ("Median final equity", _fmt(p["final_p50"]), "",
                 p["final_p50"] >= mc.initial_balance),
                ("Drawdown P95", f"{_fmt(p['dd_p95'])}%", "the one to plan for", False),
                ("Risk of ruin", f"{mc.risk_of_ruin * 100:.1f}%", "",
                 mc.risk_of_ruin < 0.05),
            ]))
            if charts:
                blocks.append(_figure(mc.plot(), "Simulated outcome range.", dpi))

        s = result.stats
        subtitle = (f"{result.spec.symbol} · {s['start']} to {s['end']} · "
                    f"{type(result.strategy).__name__}")
        meta = [
            ("Broker", result.profile.name),
            ("Net profit", _fmt(s["net_profit"])),
            ("Max DD", f"{_fmt(s['max_dd_pct'])}%"),
            ("Sections", ", ".join(
                ["backtest"] + [n for n, o in (("optimization", opt), ("walk-forward", wf),
                                               ("monte carlo", mc)) if o is not None]
            )),
        ]
        page = _page(title or f"Overall report - {result.spec.symbol}", subtitle, meta,
                     "".join(blocks), theme)
        return _write(path, page, open_browser)
    finally:
        plotting.set_theme(previous)
