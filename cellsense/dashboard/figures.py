"""Plotly figure builders, styled to match the CellSense design."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..analytics import CellForecast, Snapshot
from ..config import THRESHOLDS

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # fixed order: focus cell, comparison, third, lab target
BAND, OTHER, GOOD = "#e6e5df", "#cfcec7", "#0ca30c"
FONT = dict(family="Segoe UI, system-ui, sans-serif", size=12, color=INK2)


def _base(fig: go.Figure, height: int, ytitle: str = "", xtitle: str = "", legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=52, r=16, t=8, b=36 if xtitle else 26), paper_bgcolor="white", plot_bgcolor="white",
        font=FONT, showlegend=legend, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, title_text=xtitle, title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=ytitle, title_font=dict(size=12))
    return fig


def _window(s: Snapshot, seconds: int = 600):
    if len(s.hist_t) == 0:
        return slice(0, 0)
    return slice(int(np.searchsorted(s.hist_t, s.hist_t[-1] - seconds)), None)


def voltage_chart(s: Snapshot, cell: int, compare: int | None, seconds: int = 600) -> go.Figure:
    w = _window(s, seconds)
    x = s.hist_t[w] / 60.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=s.hist_p95[w], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=s.hist_p05[w], fill="tonexty", fillcolor=BAND, line=dict(width=0), name="Pack 5–95% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=s.hist_med[w], line=dict(color=INK2, width=1.5), name="Pack median", hovertemplate="%{y:.3f} V"))
    if compare:
        fig.add_trace(go.Scatter(x=x, y=s.hist_v[w, compare - 1], line=dict(color=SERIES[1], width=1.8), name=f"Cell {compare}", hovertemplate="%{y:.3f} V"))
    fig.add_trace(go.Scatter(x=x, y=s.hist_v[w, cell - 1], line=dict(color=SERIES[0], width=2.2), name=f"Cell {cell}", hovertemplate="%{y:.3f} V"))
    return _base(fig, 260, "Cell voltage (V)", legend=True)


def current_chart(s: Snapshot, seconds: int = 600) -> go.Figure:
    w = _window(s, seconds)
    fig = go.Figure(go.Scatter(x=s.hist_t[w] / 60.0, y=s.hist_i[w], fill="tozeroy", fillcolor=BAND, line=dict(color=MUTED, width=1.4),
                               name="Pack current", hovertemplate="%{y:.0f} A"))
    return _base(fig, 140, "Current (A)", "Drive time (min)")


def z_chart(s: Snapshot, cells: list[int], seconds: int = 600) -> go.Figure:
    w = _window(s, seconds)
    x = s.hist_t[w] / 60.0
    fig = go.Figure()
    fig.add_hrect(y0=-THRESHOLDS.z_alert, y1=THRESHOLDS.z_alert, fillcolor="#f3f7f5", line_width=0, layer="below")
    for k, c in enumerate(cells):
        fig.add_trace(go.Scatter(x=x, y=s.hist_z[w, c - 1], line=dict(color=SERIES[k % len(SERIES)], width=2.2 if k == 0 else 1.8),
                                 name=f"Cell {c}", hovertemplate="z = %{y:.1f}"))
    fig.add_hline(y=-THRESHOLDS.z_alert, line=dict(color=INK2, dash="dash", width=1.2), annotation_text=f"alert threshold z = −{THRESHOLDS.z_alert:.0f}",
                  annotation_position="bottom right", annotation_font=dict(size=11, color=INK2))
    return _base(fig, 230, "Rolling z-score", "Drive time (min)", legend=True)


def resistance_dots(s: Snapshot, highlight: list[int]) -> go.Figure:
    r = s.features["r_ohm"] * 1000
    med = float(np.nanmedian(r)) if np.any(np.isfinite(r)) else np.nan
    cells = np.arange(1, len(r) + 1)
    fig = go.Figure()
    fig.add_hline(y=med, line=dict(color=INK2, dash="dash", width=1.2))
    rest = np.array([c not in highlight for c in cells])
    fig.add_trace(go.Scatter(x=cells[rest], y=r[rest], mode="markers", marker=dict(color=OTHER, size=7), name="Cells",
                             hovertemplate="Cell %{x}: %{y:.3f} mΩ<extra></extra>"))
    for k, c in enumerate(highlight):
        if np.isfinite(r[c - 1]):
            fig.add_trace(go.Scatter(x=[c], y=[r[c - 1]], mode="markers+text", marker=dict(color=SERIES[k % len(SERIES)], size=12, line=dict(color="white", width=2)),
                                     text=[f"Cell {c} · {(r[c - 1] / med - 1) * 100:+.0f}%"], textposition="middle right", textfont=dict(size=12, color=INK),
                                     name=f"Cell {c}", hovertemplate=f"Cell {c}: %{{y:.3f}} mΩ<extra></extra>"))
    fig = _base(fig, 250, "R (mΩ)", "Cell")
    fig.update_layout(hovermode="closest")
    return fig


def forecast_chart(fcs: list[CellForecast], now_cycle: int, replace_km: float, colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    end = max(f.fx[-1] for f in fcs) if fcs else now_cycle
    fig.add_vrect(x0=now_cycle, x1=end, fillcolor="#f7f7f4", line_width=0, layer="below")
    for k, f in enumerate(fcs):
        col = colors[k] if colors else SERIES[[0, 2, 1, 3][k % 4]]
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=np.r_[f.fx, f.fx[::-1]], y=np.r_[f.hi, np.clip(f.lo, 0, None)[::-1]], fill="toself",
                                 fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.16)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=f.cycles, y=f.km, line=dict(color=col, width=2.2), name=f"Cell {f.cell}", hovertemplate="cycle %{x}: %{y:.1f} km"))
        fig.add_trace(go.Scatter(x=f.fx, y=f.fy, line=dict(color=col, width=2.2, dash="dash"), showlegend=False, hovertemplate="forecast %{y:.1f} km"))
        added = f.step_km + f.recoverable_km
        if added > 0:  # an injected fault: jump from the cell's trend to what it costs today
            base = float(f.fy[0]) - f.step_km
            kind = "recovered by balancing or charging" if f.step_km == 0 else ("part recoverable" if f.recoverable_km else "permanent")
            fig.add_trace(go.Scatter(x=[now_cycle, now_cycle], y=[base, base + added], mode="lines+markers", line=dict(color=col, width=2.6),
                                     marker=dict(color=col, size=[0, 11], symbol="diamond", line=dict(color="white", width=1.5)), showlegend=False,
                                     hovertemplate=f"Cell {f.cell} · injected {f.fault_label}: +{added:.1f} km today ({kind})<extra></extra>"))
            fig.add_annotation(x=now_cycle, y=base + added, text=f"Cell {f.cell} · injected fault: +{added:.1f} km",
                               showarrow=False, xanchor="right", xshift=-8, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12, color=col))
        if f.cross_cycle is not None:
            fig.add_trace(go.Scatter(x=[f.cross_cycle], y=[replace_km], mode="markers", marker=dict(color=col, size=11, line=dict(color="white", width=2)),
                                     showlegend=False, hovertemplate=f"Cell {f.cell} reaches {replace_km:.0f} km at cycle %{{x:.0f}}<extra></extra>"))
            fig.add_annotation(x=f.cross_cycle, y=replace_km, text=f"Cell {f.cell}: cycle {f.cross_cycle:.0f} (~{f.months_to_threshold:.0f} months)",
                               showarrow=False, yshift=-16, xanchor="right", font=dict(size=12, color=INK))
    fig.add_hline(y=replace_km, line=dict(color=INK2, dash="dash", width=1.2), annotation_text=f"Replace threshold · {replace_km:.0f} km",
                  annotation_position="top left", annotation_font=dict(size=12, color=INK2))
    fig.add_vline(x=now_cycle, line=dict(color=INK2, dash="dot", width=1.2), annotation_text=f"Today · cycle {now_cycle}",
                  annotation_position="top right", annotation_font=dict(size=12, color=INK2))
    fig = _base(fig, 430, "Range lost if this cell limits the pack (km)", "Equivalent full cycles (≈ 25 per month)", legend=True)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(uirevision="forecast")  # live redraws keep the viewer's zoom
    return fig


def lab_chart(s: Snapshot, cell: int, t_inj: float, detect_s: float | None) -> go.Figure:
    t0 = max(t_inj - 120, s.hist_t[0] if len(s.hist_t) else t_inj)
    w = slice(int(np.searchsorted(s.hist_t, t0)), None)
    x = s.hist_t[w] - t_inj
    fig = go.Figure(go.Scatter(x=x, y=s.hist_z[w, cell - 1], line=dict(color=SERIES[3], width=2.2), name=f"Cell {cell}", hovertemplate="z = %{y:.1f}"))
    fig.add_hline(y=-THRESHOLDS.z_alert, line=dict(color=INK2, dash="dash", width=1.2))
    fig.add_vline(x=0, line=dict(color=INK2, dash="dot", width=1.2), annotation_text="Injected", annotation_position="top right", annotation_font=dict(size=11))
    if detect_s is not None:
        fig.add_vline(x=detect_s, line=dict(color=GOOD, width=1.5), annotation_text=f"Detected +{detect_s:.0f} s", annotation_position="bottom right",
                      annotation_font=dict(size=11, color=INK))
    return _base(fig, 230, f"Cell {cell} z-score", "Seconds since injection")
