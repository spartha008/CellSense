"""CellSense dashboard (Plotly Dash): Pack overview, Cell diagnostics, Forecast, Fault lab."""
from __future__ import annotations

from urllib.parse import parse_qs

import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from ..analytics import ranking
from ..config import PACK, THRESHOLDS
from ..sim.faults import LABELS, SEVERITY_RANGES, Fault
from . import figures as F
from .session import LiveSession

PAGES = [("/", "Pack overview"), ("/cells", "Cell diagnostics"), ("/forecast", "Forecast"), ("/fault-lab", "Fault lab")]
CHIP_TEXT = {"replace": "Critical · Replace", "inspect": "Inspect", "balance": "Balance", "monitor": "Monitor", "ok": "Healthy"}
SEV_CLASS = {"replace": "critical", "inspect": "warning", "balance": "warning", "monitor": "monitor", "ok": "ok"}
STATUS_MARK = {"replace": "#d03b3b", "inspect": "#fab219", "balance": "#fab219", "monitor": "#52514e"}
GRAPH_CFG = {"displayModeBar": False, "responsive": True}  # charts follow their card when a phone rotates


def sgn(v: float, d: int = 0) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    a = f"{abs(v):.{d}f}"
    return ("" if float(a) == 0 else "+" if v > 0 else "−") + a


def chip(status: str, text: str | None = None):
    return html.Span([html.I(className="dot"), text or CHIP_TEXT[status]], className=f"chip {SEV_CLASS.get(status, 'ok')}")


def _mix(a: str, b: str, t: float) -> str:
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


def dev_color(mv: float, lim: float = 50.0) -> str:
    t = min(1.0, abs(mv) / lim)
    return _mix("#f0efec", "#b8312f" if mv < 0 else "#1c5cab", t)


def severity_value(kind: str, pct: float) -> float:
    lo, hi = SEVERITY_RANGES[kind]
    return lo + (hi - lo) * pct / 100.0


# ---------------------------------------------------------------------------------------- pages
def overview(sess: LiveSession):
    s, _ = sess.state()
    imp = s.impact
    act = [f for f in s.findings if f.actionable]
    counts = {k: sum(f.status == k for f in s.findings) for k in ("replace", "balance", "inspect", "monitor")}
    breakdown = " · ".join(f"{v} {k}" for k, v in counts.items() if v) or "none"
    lim = imp.limiting_cell
    kpis = html.Div(className="kpis", children=[
        html.Div(className="kpi", children=[html.Div("Pack state of charge", className="k"), html.Div([f"{s.pack_soc * 100:.0f}", html.Small("%")], className="v"),
                                            html.Div(f"Pack voltage {s.pack_voltage:.1f} V · {s.current:.0f} A", className="n")]),
        html.Div(className="kpi", children=[html.Div("Cell spread (max − min)", className="k"), html.Div([f"{s.spread_mv:.0f}", html.Small("mV")], className="v"),
                                            html.Div("90 s average", className="n")]),
        html.Div(className="kpi", children=[html.Div("Cells flagged", className="k"), html.Div([f"{len(s.findings)}", html.Small(f"of {PACK.n_series}")], className="v"),
                                            html.Div(breakdown, className="n")]),
        html.Div(className="kpi", children=[html.Div("Limiting cell", className="k"), html.Div(f"Cell {lim}", className="v"),
                                            html.Div(f"Module {PACK.module_of(lim)} · position {PACK.position_of(lim)}", className="n")]),
        html.Div(className="kpi hero", children=[html.Div("Range lost to imbalance", className="k"), html.Div([f"~{imp.lost_km:.0f}", html.Small("km")], className="v"),
                                                 html.Div(f"Cell {lim} is costing you ~{imp.lost_km:.0f} km", className="n")]),
    ])
    tiles = [html.Div()] + [html.Div(f"P{k + 1}", className="ch") for k in range(PACK.cells_per_module)]
    for m in range(PACK.modules):
        tiles.append(html.Div(f"M{m + 1}", className="ml"))
        for k in range(PACK.cells_per_module):
            c = m * PACK.cells_per_module + k + 1
            v = float(s.dev90_mv[c - 1])
            st = s.status[c - 1]
            kids = [html.Span(str(c), className="id"), html.Span(sgn(v), className="val")]
            if st != "ok":
                kids.append(html.Span(className="mark", style={"background": STATUS_MARK[st]}))
            tiles.append(dcc.Link(kids, href=f"/cells?cell={c}", className="tile flag" if st != "ok" else "tile",
                                  title=f"Cell {c}: {sgn(v, 1)} mV vs pack median" + (f" · {st}" if st != "ok" else ""),
                                  style={"background": dev_color(v), "color": "#fff" if abs(v) > 28 else "#0b0b0b"}))
    heat = html.Div(className="card", style={"flex": "1"}, children=[
        html.Div(className="hd", children=[
            html.Div([html.H3("Cell voltage deviation from pack median"), html.Div("mV · 90 s average · 8 modules × 12 cells · outlined = flagged · click a cell for detail", className="sub")]),
            html.Div(className="scale", children=[html.Span("Below pack −50"), html.Span(className="bar"), html.Span("+50 Above pack")])]),
        html.Div(tiles, className="hm"),
        html.Div(f"Flags: rolling robust z-score (z ≤ −{THRESHOLDS.z_alert:.0f} for {THRESHOLDS.persistence_s} s) and Isolation Forest "
                 f"(score ≥ {sess.engine.if_threshold:.2f}) on charge offset, load sag, ΔV/ΔI resistance and charge drift.", className="sub", style={"marginTop": "10px"}),
    ])
    shown = act[:4] + [f for f in s.findings if not f.actionable][: max(0, 4 - len(act[:4]))]
    rest = len(s.findings) - len(shown)
    alerts = []
    for f in shown:
        met = [f"IF score {f.if_score:.2f}" if np.isfinite(f.if_score) else None, f"costing {f.km_cost:.1f} km" if f.km_cost >= 0.5 else None]
        alerts.append(html.Div(className="alert", children=[
            html.Div(className="a-top", children=[chip(f.status), dcc.Link(f"Cell {f.cell}", href=f"/cells?cell={f.cell}", className="a-cell"), html.Span(f.location, className="a-loc")]),
            html.Div(f"{f.headline}. {f.detail}", className="a-txt"),
            html.Div(" · ".join(m for m in met if m) + f" · {f.action}", className="a-met")]))
    if not alerts:
        alerts = [html.Div("No cells flagged. Every cell is tracking the pack.", className="empty")]
    if rest > 0:
        alerts.append(html.Div(f"+ {rest} more on the watch list", className="sub", style={"paddingTop": "8px"}))
    actions = []
    for f in [f for f in s.findings if f.status != "monitor" or f.kind != "early_drift"][:3]:
        if f.status == "replace":
            actions.append((f"Replace Cell {f.cell} at next service", f"+{f.km_cost:.0f} km" if f.km_cost >= 0.5 else "safety"))
        elif f.status == "balance":
            actions.append((f"Run balancing on Cell {f.cell} at next charge", f"keeps {f.km_cost:.0f} km"))
        elif f.status == "inspect":
            actions.append((f"Inspect Cell {f.cell} for internal leakage", "safety"))
        else:
            actions.append((f"{f.action}: Cell {f.cell}", "watch"))
    for f in [f for f in s.findings if f.kind == "early_drift"][: 3 - len(actions)]:
        actions.append((f"Re-check Cell {f.cell} in 30 cycles", "watch"))
    usable = imp.usable_km
    act_card = html.Div(className="card", style={"flex": "1"}, children=[
        html.H3("Recommended actions"),
        *[html.Div(className="act", children=[html.Span(str(i + 1), className="n"), html.Span(t), html.Span(g, className="gain")]) for i, (t, g) in enumerate(actions)],
        *([] if actions else [html.Div("Nothing to do. The pack is balanced.", className="empty")]),
        html.Div(style={"marginTop": "12px", "paddingTop": "12px", "borderTop": "1px solid #f0efea"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "12.5px", "color": "#52514e"},
                     children=[html.Span("Usable range now"), html.Span([html.B(f"{usable:.0f} km", style={"color": "#0b0b0b", "fontSize": "15px"}), f" of {PACK.rated_range_km:.0f} km"])]),
            html.Div(className="rangebar", children=[html.Span(style={"flex": f"{usable}", "background": "#0f2328"}),
                                                     html.Span(style={"flex": f"{max(imp.lost_km, 0.01)}", "background": "#b8312f"})]),
            html.Div(className="legend", children=[html.Span([html.I(style={"background": "#0f2328"}), "Available"]),
                                                   html.Span([html.I(style={"background": "#b8312f"}), f"Lost to Cell {lim} ({imp.lost_km:.0f} km)"])]),
        ]),
    ])
    return [kpis, html.Div(className="row", children=[heat, html.Div(className="col aside", style={"width": "450px"}, children=[
        html.Div(className="card", children=[html.Div(className="hd", style={"marginBottom": "0"}, children=[html.H3("Active alerts"), html.Span(f"{len(s.findings)} open", className="sub")]), *alerts]),
        act_card])])]


def cells_static(sess: LiveSession, cell: int):
    s, _ = sess.state()
    flagged = [f.cell for f in s.findings]
    opts = [{"label": f"Cell {c} · {s.status[c - 1]}", "value": c} for c in flagged] + \
           [{"label": f"Cell {c}", "value": c} for c in range(1, PACK.n_series + 1) if c not in flagged]
    return html.Div(className="card pick", style={"display": "flex", "alignItems": "center", "gap": "14px", "padding": "10px 18px"}, children=[
        html.Span("Cell", className="sub", style={"fontWeight": 600}),
        dcc.Dropdown(id="cell-dd", options=opts, value=cell, clearable=False, style={"width": "260px"}),
        html.Span("Flagged cells are listed first. Click any tile on the Pack overview to jump here.", className="sub")])


def _compare_cell(s, cell: int) -> int:
    f0 = s.finding(cell)
    for f in s.findings:
        if f.cell != cell and (f0 is None or f.kind != f0.kind) and f.actionable:
            return f.cell
    order = np.argsort(np.abs(s.dev90_mv))
    return int(order[0] + 1) if order[0] + 1 != cell else int(order[1] + 1)


def cells_dynamic(sess: LiveSession, cell: int):
    s, _ = sess.state()
    i = cell - 1
    f = s.finding(cell)
    ft = s.features
    status = f.status if f else "ok"
    cmp = _compare_cell(s, cell)
    r_ratio = ft["r_ratio"]
    strip = html.Div(className="card strip", children=[
        html.Div([html.Div(f"Cell {cell}", className="big"), html.Div(f"Module {PACK.module_of(cell)} · position {PACK.position_of(cell)} · NMC {PACK.capacity_ah:.0f} Ah", className="sub")]),
        chip(status, f"{f.headline} · {f.action}" if f else "Healthy"),
        html.Div(className="ms", style={"marginLeft": "auto"}, children=[html.Div("Internal resistance (ΔV/ΔI)", className="k"),
                 html.Div([f"{ft['r_ohm'][i] * 1000:.2f}" if np.isfinite(ft['r_ohm'][i]) else "n/a", html.Small(f"mΩ · {sgn((r_ratio[i] - 1) * 100)}% vs pack")], className="v")]),
        html.Div(className="ms", children=[html.Div("Charge offset", className="k"), html.Div(f"{sgn(ft['rest_offset_mv'][i])} mV", className="v")]),
        html.Div(className="ms", children=[html.Div("Offset under load", className="k"), html.Div(f"{sgn(ft['load_offset_mv'][i])} mV", className="v")]),
        html.Div(className="ms", children=[html.Div("Isolation Forest score", className="k"),
                 html.Div([f"{s.if_score[i]:.2f}" if np.isfinite(s.if_score[i]) else "n/a", html.Small(f"alert ≥ {sess.engine.if_threshold:.2f}")], className="v")]),
    ])
    left = html.Div(className="col", style={"flex": "1"}, children=[
        html.Div(className="card", children=[html.Div(className="hd", children=[html.Div([html.H3("Cell voltage vs pack: last 10 minutes"), html.Div("Pack median and 5–95% band of all 96 cells", className="sub")])]),
                                             dcc.Graph(figure=F.voltage_chart(s, cell, cmp), config=GRAPH_CFG), dcc.Graph(figure=F.current_chart(s), config=GRAPH_CFG)]),
        html.Div(className="card", children=[html.Div(className="hd", children=[html.Div([html.H3("Rolling z-score vs pack median"), html.Div("30 s window, robust (median / MAD) scaling", className="sub")])]),
                                             dcc.Graph(figure=F.z_chart(s, [cell, cmp]), config=GRAPH_CFG)]),
    ])

    def row(label, a, b):
        return html.Tr([html.Td(label), html.Td(a, className="num"), html.Td(b, className="num")])

    def drift100(j):
        v = ft["fade_slope"][j]
        return f"{sgn(v * 100, 2)}%" if np.isfinite(v) else "n/a"

    def parked(j):
        v = ft["sd_drift"][j]
        return f"{sgn(v, 1)}%/h" if np.isfinite(v) else "not parked yet"

    fc = s.finding(cmp)
    table = html.Table(className="t", children=[
        html.Tr([html.Th("Signal"), html.Th(html.Span([html.I(style={"background": F.SERIES[0]}), f"Cell {cell}"], className="sw"), className="num"),
                 html.Th(html.Span([html.I(style={"background": F.SERIES[1]}), f"Cell {cmp}"], className="sw"), className="num")]),
        row("Charge offset vs pack", f"{sgn(ft['rest_offset_mv'][i])} mV", f"{sgn(ft['rest_offset_mv'][cmp - 1])} mV"),
        row("Offset under load", f"{sgn(ft['load_offset_mv'][i])} mV", f"{sgn(ft['load_offset_mv'][cmp - 1])} mV"),
        row("Resistance vs pack", f"{sgn((r_ratio[i] - 1) * 100)}%", f"{sgn((r_ratio[cmp - 1] - 1) * 100)}%"),
        row("Charge drift per 100 Ah", drift100(i), drift100(cmp - 1)),
        row("Drift while parked", parked(i), parked(cmp - 1)),
        row("Isolation Forest score", f"{s.if_score[i]:.2f}", f"{s.if_score[cmp - 1]:.2f}"),
        html.Tr([html.Td("Verdict"), html.Td(chip(status, f.action if f else "Healthy"), className="num"),
                 html.Td(chip(fc.status if fc else "ok", fc.action if fc else "Healthy"), className="num")]),
    ])
    right = html.Div(className="col aside", style={"width": "520px"}, children=[
        html.Div(className="card", children=[html.H3("Internal resistance by cell"), html.Div(f"ΔV/ΔI at current steps ≥ {THRESHOLDS.step_di_a:.0f} A, all 96 cells · dashed line = pack median", className="sub"),
                                             dcc.Graph(figure=F.resistance_dots(s, [cell, cmp]), config=GRAPH_CFG)]),
        html.Div(className="card", style={"flex": "1"}, children=[
            html.H3("Diagnosis: low on charge, or degrading?"),
            html.Div(f.detail if f else "This cell is tracking the pack: no offset, normal resistance, no drift.", className="sub"), table,
            html.Div(["A low-charge cell is offset by a ", html.B("constant"), " amount at any current, so balancing fixes it. A degrading cell sags ",
                      html.B("more as current rises"), " (higher resistance), so it needs replacing. A cell that keeps ", html.B("falling behind"),
                      " is losing capacity, or leaking if it falls behind even while parked."], className="note")]),
    ])
    return [strip, html.Div(className="row", children=[left, right])]


def forecast_view(sess: LiveSession):
    """Cells ranked by forecast (injected lab faults folded in), the cells to draw with their colours, and a key
    that changes only when the drawn picture does."""
    order = ranking(sess.forecast_view(), top=PACK.n_series)
    _, injs = sess.state()
    hot = injs[-1].fault.cell if injs else None  # the cell the lab touched last
    chart, colors = order[:2], [F.SERIES[0], F.SERIES[2]]
    if hot is not None and hot not in [f.cell for f in chart]:
        chart, colors = chart + [next(f for f in order if f.cell == hot)], colors + [F.SERIES[3]]
    key = [sess.generation] + [[f.cell, round(f.step_km, 1), round(f.recoverable_km, 1)] for f in chart]  # a reset is a new pack
    return order, hot, chart, colors, key


def forecast_figure(sess: LiveSession, view=None):
    _, _, chart, colors, _ = view or forecast_view(sess)
    return F.forecast_chart(chart, sess.hist.now_cycle, THRESHOLDS.replace_km, colors)


def ranking_card(sess: LiveSession, view=None):
    order, hot, _, _, _ = view or forecast_view(sess)
    s, injs = sess.state()
    shown = list(enumerate(order[:3]))
    if hot is not None and hot not in [fc.cell for _, fc in shown]:
        shown.append(next((k, fc) for k, fc in enumerate(order) if fc.cell == hot))
    rows = []
    for k, fc in shown:
        f = s.finding(fc.cell)
        added = fc.step_km + fc.recoverable_km
        m = fc.months_to_threshold
        when = ("now" if m < 0.5 else f"cycle {fc.cross_cycle:.0f} (~{m:.0f} mo)") if fc.cross_cycle is not None else "not within 10 months"
        colour = F.SERIES[3] if fc.cell == hot and k >= 2 else F.SERIES[[0, 2, 1][k]] if k < 3 else F.SERIES[3]
        cause = (f"{f.headline}: {f.detail}" if f else f"Injected {fc.fault_label}; CellSense still watching" if fc.fault_label
                 else "Capacity trending below the pack")
        rows.append(html.Tr([html.Td(str(k + 1), className="muted"), html.Td(html.Span([html.I(style={"background": colour}), f"Cell {fc.cell}"], className="sw")),
                             html.Td([f"{fc.now_km:.1f} km", html.Div(f"+{added:.1f} km injected", className="sub")] if added > 0 else f"{fc.now_km:.1f} km"),
                             html.Td(f"~{fc.km_6mo:.0f} km"), html.Td(when), html.Td(cause, className="sub"),
                             html.Td(chip(f.status if f else "monitor", f.action if f else "Monitor"))]))
    out = [html.H3("Weakest-link ranking"), html.Table(className="t", children=[
        html.Tr([html.Th("#"), html.Th("Cell"), html.Th("Range cost now"), html.Th("In 6 months"), html.Th(f"Hits {THRESHOLDS.replace_km:.0f} km"), html.Th("Root cause"), html.Th("Action")]), *rows])]
    if injs:
        out.append(html.Div([html.B("How an injected fault moves the forecast. "),
                             "Capacity fade and resistance growth shift the cell's trend up from today (resistance as the charge lost when the cell reaches "
                             f"{PACK.v_min:.1f} V under a {THRESHOLDS.load_current_a:.0f} A load before the rest of the pack does). Charge imbalance and "
                             "self-discharge cost range today, but balancing or charging gives it back, so the trend stays where it was."],
                            className="note", style={"marginTop": "8px"}))
    return out


def forecast_page(sess: LiveSession):
    view = forecast_view(sess)
    chart = html.Div(className="card", children=[
        html.Div(className="hd", children=[html.Div([html.H3("Which cell will limit the pack, and when?"),
                                                      html.Div("Range lost if this cell limits the pack · history (solid), quadratic trend forecast with 95% band (dashed)", className="sub")])]),
        dcc.Graph(id="fc-chart", figure=forecast_figure(sess, view), config=GRAPH_CFG), dcc.Store(id="fc-key", data=view[4])])
    table = html.Div(id="fc-rank", className="card", children=ranking_card(sess, view))
    return [chart, table]


def lab_static(sess: LiveSession, compact: bool = False):
    """Fault lab controls. ``compact`` stacks them for the narrow column beside the forecast."""
    w = (lambda px: {"width": "100%"}) if compact else (lambda px: {"width": px})
    return html.Div(className="card", children=[
        html.H3("Fault lab"), html.Div("Inject a fault into the running pack and watch CellSense catch it, next to a threshold-only BMS.", className="sub"),
        html.Div(className="col" if compact else "row", style={"marginTop": "6px", "flexWrap": "wrap", "gap": "0" if compact else "14px"}, children=[
            html.Div(className="ctl", style=w("200px"), children=[html.Div("Target cell", className="lb"),
                     dcc.Dropdown(id="lab-cell", options=[{"label": f"Cell {c} · {PACK.label(c)}", "value": c} for c in range(1, PACK.n_series + 1)], value=29, clearable=False)]),
            html.Div(className="ctl", style={"width": "100%"} if compact else {"flex": "1", "minWidth": "420px"}, children=[html.Div("Fault type", className="lb"),
                     dcc.RadioItems(id="lab-kind", options=[{"label": LABELS[k], "value": k} for k in SEVERITY_RANGES], value="resistance_growth", className="seg", inline=True)]),
            html.Div(className="ctl", style=w("330px"), children=[html.Div(["Severity: ", html.Span(id="lab-sev-text", style={"color": "#0b0b0b"})], className="lb"),
                     dcc.Slider(id="lab-sev", min=0, max=100, step=5, value=40, marks={0: "mild", 50: "", 100: "severe"})]),
        ]),
        html.Div(className="btns", children=[
            html.Button("Inject fault now", id="btn-inject", className="btn pri"),
            html.Button("Park the car for 8 min", id="btn-park", className="btn"),
            html.Button("Reset: demo pack", id="btn-demo", className="btn"),
            html.Button("Reset: healthy pack", id="btn-healthy", className="btn")]),
        html.Div(id="lab-status", className="status-line"),
    ])


def lab_dynamic(sess: LiveSession, compact: bool = False):
    """Detection result for the latest injected fault; the full page also shows the injection log."""
    s, injs = sess.state()
    if not injs:
        return [html.Div(className="card", children=[html.H3("Detection result"), html.Div("No fault injected yet. Pick a cell, a fault type and a severity, then press Inject.", className="empty")])]
    inj = injs[-1]
    d = sess.detection(inj)
    f = d["finding"]
    cs = d["cellsense_s"]
    cs_chip = chip("ok", f"Detected in {cs:.0f} s") if cs is not None else chip("monitor", f"Watching… {d['elapsed']:.0f} s")
    if cs is not None:
        cs_chip = html.Span([html.I(className="dot"), f"Detected in {cs:.0f} s"], className="chip good")
    bms = d["bms_s"]
    bms_chip = html.Span([html.I(className="dot"), f"Alarm after {bms:.0f} s"], className="chip warning") if bms is not None else html.Span([html.I(className="dot"), "No alarm"], className="chip ok")
    diag = f"{f.headline}: {f.detail} → {f.action}" if f else "No diagnosis yet."
    hist = [html.Tr([html.Td(f"{j.t_inj / 60:.1f} min"), html.Td(j.fault.describe()),
                     html.Td((lambda r: f"{r['cellsense_s']:.0f} s" if r["cellsense_s"] is not None else "not yet")(sess.detection(j))),
                     html.Td((lambda r: f"{r['bms_s']:.0f} s" if r["bms_s"] is not None else "no alarm")(sess.detection(j))),
                     html.Td((lambda r: r["finding"].action if r["finding"] else "–")(sess.detection(j)))]) for j in reversed(injs[-6:])]
    result = html.Div(className="card", style={"flex": "1"}, children=[
            html.H3("Detection result"), html.Div(inj.fault.describe() + f", injected {d['elapsed']:.0f} s ago", className="sub"),
            html.Div(className="res", children=[html.Div([html.Div("CellSense", className="who"), html.Div(diag, className="det")]), html.Div(cs_chip, className="r")]),
            html.Div(className="res", children=[html.Div([html.Div("Threshold-only BMS", className="who"),
                     html.Div(f"Watches a {PACK.v_min:.1f}–{PACK.v_max:.1f} V window and a fixed {THRESHOLDS.bms_rest_dv * 1000:.0f} mV rest-imbalance limit.", className="det")]),
                     html.Div(bms_chip, className="r")]),
            dcc.Graph(figure=F.lab_chart(s, inj.fault.cell, inj.t_inj, cs), config=GRAPH_CFG)])
    if compact:
        return [result]
    return [html.Div(className="row", children=[result,
        html.Div(className="card aside", style={"width": "560px"}, children=[html.H3("Injection log"),
                 html.Table(className="t", children=[html.Tr([html.Th("When"), html.Th("Fault"), html.Th("CellSense"), html.Th("Threshold BMS"), html.Th("Action")]), *hist])]),
    ])]


def forecast_lab_page(sess: LiveSession):
    """Deck Screen 3: weakest-link forecast and ranking, with the Fault lab beside it for live injection."""
    return html.Div(className="row", children=[
        html.Div(className="col", style={"flex": "1"}, children=forecast_page(sess)),
        html.Div(className="col aside", style={"width": "440px", "flex": "none"}, children=[
            lab_static(sess, compact=True),
            html.Div(id="fc-lab-dynamic", className="col", children=lab_dynamic(sess, compact=True))]),
    ])


# ---------------------------------------------------------------------------------------- app
def create_app(sess: LiveSession) -> Dash:
    app = Dash(__name__, title="CellSense", update_title=None, suppress_callback_exceptions=True)
    logo = html.Div(className="logo", children=[html.Img(src=app.get_asset_url("logo.svg"), style={"width": "30px"}), "CellSense"])
    app.layout = html.Div(className="app", children=[
        dcc.Location(id="url"), dcc.Interval(id="tick", interval=1000), dcc.Store(id="sel-cell", data=47),
        html.Aside(className="side", children=[logo, html.Nav(className="nav", children=[dcc.Link(n, href=p, id=f"nav-{k}") for k, (p, n) in enumerate(PAGES)]),
                                               html.Div(className="side-foot", children=[html.Div("Data source", className="sf-l"), html.Div(id="src", className="sf-v"),
                                                                                         html.Div(id="src2", style={"color": "#9fb7b1", "marginTop": "2px"})])]),
        html.Main(className="main", children=[
            html.Header(className="top", children=[
                html.Div([html.Div(id="t-title", className="t-title"), html.Div(id="t-sub", className="t-sub")]),
                html.Div(className="t-right", children=[html.Span([html.I(), "Live"], className="live"), html.Span(id="clock", className="pill"),
                                                        dcc.RadioItems(id="speed", options=[{"label": f"{x}×", "value": x} for x in (1, 5, 20)], value=int(sess.speed), className="speed", inline=True),
                                                        html.Span(id="speed-ack", style={"display": "none"})])]),
            html.Section(className="content", children=[html.Div(id="page-static"), html.Div(id="page-dynamic", className="col")]),
        ]),
    ])

    titles = {"/forecast": "Weakest-link forecast & fault lab"}

    @app.callback([Output(f"nav-{k}", "className") for k in range(len(PAGES))] + [Output("t-title", "children")], Input("url", "pathname"))
    def _nav(path):
        path = path if path in dict(PAGES) else "/"
        return [("on" if p == path else "") for p, _ in PAGES] + [titles.get(path, dict(PAGES)[path])]

    @app.callback(Output("page-static", "children"), Output("sel-cell", "data"), Output("page-dynamic", "children", allow_duplicate=True),
                  Input("url", "pathname"), Input("url", "search"), State("sel-cell", "data"), prevent_initial_call="initial_duplicate")
    def _static(path, search, cell):
        q = parse_qs((search or "").lstrip("?"))
        if "cell" in q:
            cell = int(q["cell"][0])
        if path == "/cells":
            return cells_static(sess, cell), cell, []  # [] clears the previous page at once
        if path == "/fault-lab":
            return lab_static(sess), cell, []
        if path == "/forecast":
            return forecast_lab_page(sess), cell, []
        return None, cell, []

    @app.callback(Output("sel-cell", "data", allow_duplicate=True), Input("cell-dd", "value"), prevent_initial_call=True)
    def _pick(c):
        return c

    @app.callback(Output("page-dynamic", "children"), Output("t-sub", "children"), Output("clock", "children"), Output("src", "children"), Output("src2", "children"),
                  Input("tick", "n_intervals"), Input("url", "pathname"), Input("sel-cell", "data"))
    def _dynamic(_, path, cell):
        s, _i = sess.state()
        mode = {"demo": "Demo pack (Cells 47, 12, 83 pre-set)", "healthy": "Healthy pack"}.get(sess.mode, "Recorded physics run")
        sub = f"Pack A · 96s1p NMC · {PACK.energy_kwh:.0f} kWh · rated range {PACK.rated_range_km:.0f} km · {mode}"
        clock = f"Drive time {int(s.t // 60):d}:{int(s.t % 60):02d} · {s.current:.0f} A"
        if sess.replay is not None:
            src, src2 = "PyBaMM + liionpack", f"96 SPMe cells · replay of {sess.replay_path.replace(chr(92), '/').split('/')[-1]}"
        else:
            src, src2 = "Real-time ECM simulator", f"Cells fitted to {sess.params.source}"
        if path == "/cells":
            body = cells_dynamic(sess, int(cell or 47))
        elif path == "/fault-lab":
            body = lab_dynamic(sess)
        elif path == "/forecast":
            body = []
        else:
            body = overview(sess)
        return body, sub, clock, src, src2

    @app.callback(Output("fc-chart", "figure"), Output("fc-key", "data"), Output("fc-rank", "children"), Output("fc-lab-dynamic", "children"),
                  Input("tick", "n_intervals"), State("url", "pathname"), State("fc-key", "data"))
    def _forecast_lab(_, path, key):
        if path != "/forecast":
            return no_update, no_update, no_update, no_update
        view = forecast_view(sess)
        changed = view[4] != key  # redraw the chart only when an injected fault changed it (keeps hover steady)
        return (forecast_figure(sess, view) if changed else no_update, view[4] if changed else no_update,
                ranking_card(sess, view), lab_dynamic(sess, compact=True))

    @app.callback(Output("speed-ack", "children"), Input("speed", "value"))
    def _speed(v):
        sess.set_speed(v)
        return str(v)

    @app.callback(Output("lab-sev-text", "children"), Input("lab-kind", "value"), Input("lab-sev", "value"))
    def _sev(kind, pct):
        return Fault(kind, 1, severity_value(kind, pct or 0)).describe().split(": ", 1)[1]

    # Outputs must stay inside the page this callback's buttons live on: an output that is already on screen
    # (such as page-static) makes Dash fire the callback whenever the page appears, which would inject a fault.
    # After a reset, the forecast beside the lab is redrawn by _forecast_lab (its key includes the pack generation).
    @app.callback(Output("lab-status", "children"), Input("btn-inject", "n_clicks"), Input("btn-park", "n_clicks"), Input("btn-demo", "n_clicks"),
                  Input("btn-healthy", "n_clicks"), State("lab-cell", "value"), State("lab-kind", "value"), State("lab-sev", "value"), prevent_initial_call=True)
    def _lab(_a, _b, _c, _d, cell, kind, pct):
        b = ctx.triggered_id
        if b == "btn-inject":
            f = sess.inject(kind, int(cell), severity_value(kind, pct))
            if f is None:
                return "Replaying a recorded physics run: fault injection needs the live simulator (start without --replay)."
            return f"Injected: {f.describe()} at drive time {f.t_start / 60:.1f} min."
        if b == "btn-park":
            sess.park(480)
            return "Parking for 8 minutes (simulated): self-discharge shows after ~5 min of settled data."
        if b in ("btn-demo", "btn-healthy"):
            sess.reset("demo" if b == "btn-demo" else "healthy")
            return f"Reset to a fresh {'demo' if b == 'btn-demo' else 'healthy'} pack."
        return no_update

    return app


def run(host: str = "127.0.0.1", port: int = 8050, mode: str = "demo", speed: float = 5.0, debug: bool = False, replay: str | None = None):
    sess = LiveSession(mode=mode, speed=speed, replay=replay)
    app = create_app(sess)
    print(f"CellSense dashboard on http://{host}:{port}  (mode={mode}, speed={speed}x)")
    app.run(host=host, port=port, debug=debug)
