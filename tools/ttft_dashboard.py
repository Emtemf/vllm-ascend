#!/usr/bin/env python3
"""Parse [TTFT_DEBUG] / [TTFT][BREAKDOWN] logs and emit a self-contained HTML dashboard.

Usage:
    python ttft_dashboard.py worker.log [more.log ...] -o ttft_report.html
    python ttft_dashboard.py --demo -o demo.html

Recognized lines (key=value pairs, order-independent):
  [TTFT_DEBUG][AscendStoreLoad] req_id=.. keys=.. bytes=.. queue_wait=.. total=..
      build_key=.. prepare=.. get=.. result=.. other=.. [load_async=True]
  [TTFT_DEBUG][AscendStoreLoad] req_ids=[..] start_load_kv=.. load_async=.. use_layerwise=..
  [TTFT_DEBUG][AscendStoreSave] req_id=.. keys=.. bytes=.. queue_wait=.. total=..
      build_key=.. lookup=.. prepare=.. sync=.. put=.. other=..
  [TTFT_DEBUG][AscendStoreSave] req_ids=[..] wait_for_save=..
  [TTFT][BREAKDOWN] req=.. ttft=.. hash=.. ext_lookup=.. alloc=.. overhead=..
      load_kv=.. forward=.. save_kv=.. update=.. residual=..
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field

KV_RE = re.compile(r"(\w+)=((?:\[[^\]]*\])|(?:\S+))")
STAGES_LOAD = ["build_key", "prepare", "get", "result", "other"]
STAGES_SAVE = ["build_key", "lookup", "prepare", "sync", "put", "other"]


@dataclass
class Records:
    load_detail: list[dict] = field(default_factory=list)   # per req (sync/async)
    load_summary: list[dict] = field(default_factory=list)  # per step
    save_detail: list[dict] = field(default_factory=list)   # per req
    save_summary: list[dict] = field(default_factory=list)  # per step
    breakdown: list[dict] = field(default_factory=list)     # per req e2e


def parse_line(line: str, recs: Records) -> None:
    if "[TTFT_DEBUG]" not in line and "[TTFT][BREAKDOWN]" not in line:
        return
    kv = {m.group(1): m.group(2) for m in KV_RE.finditer(line)}
    if "AscendStoreLoad" in line:
        if "req_id" in kv:
            d = {"req_id": kv["req_id"], "async": "load_async=True" in line or kv.get("load_async") == "True"}
            for k in ("keys", "bytes"):
                if k in kv:
                    d[k] = int(kv[k])
            for k in ["queue_wait", "total", *STAGES_LOAD]:
                if k in kv:
                    d[k] = float(kv[k])
            recs.load_detail.append(d)
        elif "req_ids" in kv:
            recs.load_summary.append({
                "n": len(kv["req_ids"].strip("[]'\"").split(",")) if kv["req_ids"].strip("[]'\"") else 0,
                "start_load_kv": float(kv.get("start_load_kv", 0)),
                "load_async": kv.get("load_async"),
                "use_layerwise": kv.get("use_layerwise"),
            })
    elif "AscendStoreSave" in line:
        if "req_id" in kv:
            d = {"req_id": kv["req_id"]}
            for k in ("keys", "bytes"):
                if k in kv:
                    d[k] = int(kv[k])
            for k in ["queue_wait", "total", *STAGES_SAVE]:
                if k in kv:
                    d[k] = float(kv[k])
            recs.save_detail.append(d)
        elif "req_ids" in kv:
            recs.save_summary.append({
                "n": len(kv["req_ids"].strip("[]'\"").split(",")) if kv["req_ids"].strip("[]'\"") else 0,
                "wait_for_save": float(kv.get("wait_for_save", 0)),
                "enqueue": float(kv.get("enqueue", 0)),
                "join": float(kv.get("join", 0)),
            })
    elif "req" in kv and "ttft" in kv:
        d = {"req_id": kv["req"]}
        for k in ("ttft", "hash", "ext_lookup", "alloc", "overhead", "load_kv",
                  "forward", "save_kv", "update", "residual"):
            if k in kv:
                d[k] = float(kv[k])
        recs.breakdown.append(d)


def parse_files(paths: list[str], recs: Records) -> None:
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                parse_line(line, recs)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def fmt(v: float) -> str:
    return f"{v * 1000:.1f}"  # ms


def stage_table(rows: list[dict], stages: list[str], total_key: str, label: str) -> str:
    if not rows:
        return f"<p class=muted>No {label} records.</p>"
    n = len(rows)
    out = [f"<h3>{label} <span class=muted>(n={n})</span></h3>",
           "<table><tr><th>stage</th><th>avg&nbsp;ms</th><th>p50</th><th>p90</th>"
           "<th>p99</th><th>max</th><th>share</th></tr>"]
    totals = [r[total_key] for r in rows if total_key in r]
    sum_total = sum(totals) or 1.0
    for st in stages:
        vals = [r[st] for r in rows if st in r]
        if not vals:
            continue
        share = sum(vals) / sum_total * 100
        out.append(
            f"<tr><td>{st}</td><td>{fmt(sum(vals)/len(vals))}</td><td>{fmt(pct(vals,.5))}</td>"
            f"<td>{fmt(pct(vals,.9))}</td><td>{fmt(pct(vals,.99))}</td><td>{fmt(max(vals))}</td>"
            f"<td>{share:.1f}%</td></tr>")
    out.append(
        f"<tr class=total><td>{total_key}</td><td>{fmt(sum(totals)/len(totals))}</td>"
        f"<td>{fmt(pct(totals,.5))}</td><td>{fmt(pct(totals,.9))}</td>"
        f"<td>{fmt(pct(totals,.99))}</td><td>{fmt(max(totals))}</td><td>100%</td></tr></table>")
    return "\n".join(out)


def bandwidth(rows: list[dict], time_key: str, label: str) -> list[str]:
    out = []
    for r in rows:
        if "bytes" in r and time_key in r and r.get(time_key, 0) > 0:
            gbps = r["bytes"] / r[time_key] / (1 << 30)
            out.append((r["req_id"], r["bytes"], gbps))
    if not out:
        return [f"<p class=muted>No bytes data for {label}.</p>"]
    rates = [g for _, _, g in out]
    return [f"<h3>{label} bandwidth</h3>",
            "<table><tr><th>metric</th><th>GiB/s</th></tr>",
            f"<tr><td>avg</td><td>{sum(rates)/len(rates):.2f}</td></tr>",
            f"<tr><td>p50</td><td>{pct(rates,.5):.2f}</td></tr>",
            f"<tr><td>p90</td><td>{pct(rates,.9):.2f}</td></tr>",
            f"<tr class=total><td>min</td><td>{min(rates):.2f}</td></tr></table>"]


def histogram_js(values: list[float], bins: int = 30) -> tuple[list[float], list[int]]:
    if not values:
        return [], []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [lo], [len(values)]
    w = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / w))] += 1
    return [lo + w * (i + 0.5) for i in range(bins)], counts


def kpi_card(label: str, value: str, sub: str) -> str:
    return (f"<div class=card><div class=cl>{label}</div>"
            f"<div class=cv>{value}</div><div class=cs>{sub}</div></div>")


def render(recs: Records) -> str:
    loads = recs.load_detail
    saves = recs.save_detail
    brk = recs.breakdown
    sync_loads = [r for r in loads if not r.get("async")]
    async_loads = [r for r in loads if r.get("async")]
    cards = []
    if brk:
        ttfts = [r["ttft"] for r in brk]
        cards.append(kpi_card("TTFT p50 / p90", f"{fmt(pct(ttfts,.5))} / {fmt(pct(ttfts,.9))} ms", f"n={len(ttfts)}"))
        res = [r.get("residual", 0) for r in brk]
        cards.append(kpi_card("residual p50", f"{fmt(pct(res,.5))} ms", "ttft − sum(stages)"))
    if loads:
        g = [r["get"] for r in loads if "get" in r]
        if g:
            cards.append(kpi_card("D get p50 / p90", f"{fmt(pct(g,.5))} / {fmt(pct(g,.9))} ms", f"n={len(g)}"))
    if saves:
        p = [r["put"] for r in saves if "put" in r]
        q = [r["queue_wait"] for r in saves if "queue_wait" in r]
        if p:
            cards.append(kpi_card("P put p50 / p90", f"{fmt(pct(p,.5))} / {fmt(pct(p,.9))} ms", f"n={len(p)}"))
        if q:
            cards.append(kpi_card("P queue_wait p90", f"{fmt(pct(q,.9))} ms", "backend thread backlog"))

    # scatter: bytes vs get/put time
    scatter = {"load": [[r["bytes"], r["get"]] for r in loads if "bytes" in r and "get" in r],
               "save": [[r["bytes"], r["put"]] for r in saves if "bytes" in r and "put" in r]}
    # trend: per-record order index vs total
    trend = {"load": [[i, r.get("total", 0)] for i, r in enumerate(loads) if "total" in r],
             "save": [[i, r.get("total", 0)] for i, r in enumerate(saves) if "total" in r]}
    hist_get = histogram_js([r["get"] for r in loads if "get" in r])
    hist_put = histogram_js([r["put"] for r in saves if "put" in r])

    data = json.dumps({
        "scatter": scatter, "trend": trend,
        "hist_get": hist_get, "hist_put": hist_put,
    })

    body = ["<h2>KPIs</h2><div class=cards>" + "".join(cards) + "</div>",
            stage_table(sync_loads, STAGES_LOAD, "total", "D load · sync"),
            stage_table(async_loads, ["queue_wait", *STAGES_LOAD], "total", "D load · async"),
            stage_table(saves, ["queue_wait", *STAGES_SAVE], "total", "P save"),
            "".join(f"<div class=col>{c}</div>" for c in (
                bandwidth(loads, "get", "D get") + bandwidth(saves, "put", "P put"))),
            "<h2>Distribution &amp; trends</h2>",
            "<div class=charts>"
            "<div class=chart id=c_hist_get><h3>get latency</h3></div>"
            "<div class=chart id=c_hist_put><h3>put latency</h3></div>"
            "<div class=chart id=c_scatter><h3>bytes vs time (get ○ / put ×)</h3></div>"
            "<div class=chart id=c_trend><h3>total per record</h3></div>"
            "</div>"]
    if recs.load_summary:
        sl = [r["start_load_kv"] for r in recs.load_summary]
        body.append(f"<p class=muted>load steps: {len(sl)}, start_load_kv p50={fmt(pct(sl,.5))}ms "
                    f"p90={fmt(pct(sl,.9))}ms</p>")
    if recs.save_summary:
        ws = [r["wait_for_save"] for r in recs.save_summary]
        enq = [r.get("enqueue", 0) for r in recs.save_summary]
        jn = [r.get("join", 0) for r in recs.save_summary]
        body.append(f"<p class=muted>save steps: {len(ws)}, wait_for_save p50={fmt(pct(ws,.5))}ms "
                    f"p90={fmt(pct(ws,.9))}ms (enqueue p50={fmt(pct(enq,.5))}ms, "
                    f"join p50={fmt(pct(jn,.5))}ms / p90={fmt(pct(jn,.9))}ms)</p>")
    if brk:
        stage_table(brk, ["hash", "ext_lookup", "alloc", "overhead", "load_kv",
                          "forward", "save_kv", "update", "residual"], "ttft", "End-to-end (BREAKDOWN)")
    return TEMPLATE.replace("__DATA__", html.escape(data)).replace("__BODY__", "\n".join(body))


TEMPLATE = """<!doctype html><html><head><meta charset=utf-8><title>TTFT Dashboard</title>
<style>
:root{--fg:#222;--muted:#777;--border:#ddd;--surface:#f7f7f7;--accent:#369;--success:#3a3;--warning:#c80;--error:#c33;--c1:#369;--c2:#c33}
body{font:14px/1.45 system-ui,sans-serif;color:var(--fg);margin:24px;max-width:1200px}
h2{border-bottom:1px solid var(--border);padding-bottom:4px;margin-top:32px}
h3{margin:8px 0 4px;font-size:13px;color:var(--muted)}
.muted{color:var(--muted)} table{border-collapse:collapse;margin:8px 0}
td,th{border:1px solid var(--border);padding:3px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
tr.total td{font-weight:600;background:var(--surface)}
.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{border:1px solid var(--border);border-radius:8px;padding:10px 16px;min-width:170px}
.cl{font-size:12px;color:var(--muted)}.cv{font-size:22px;font-weight:600}.cs{font-size:11px;color:var(--muted)}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.chart{border:1px solid var(--border);border-radius:8px;padding:8px}
svg text{font:10px system-ui}
.col{display:inline-block;vertical-align:top;margin-right:32px}
</style></head><body>
<h1>TTFT Dashboard — AscendStore P/D</h1>
__BODY__
<script>
const D = JSON.parse("__DATA__".replace(/&quot;/g,'"').replace(/&#x27;/g,"'").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&amp;/g,"&"));
const NS="http://www.w3.org/2000/svg";
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e}
function lineChart(id,xs,series){ // series: [{vals,color}]
 const W=520,H=200,P=36,c=document.getElementById(id);
 if(!xs.length){c.appendChild(el("text",{x:10,y:30,fill:"#777"})).textContent="no data";return}
 const maxY=Math.max(1,...series.flatMap(s=>s.vals)),maxX=Math.max(...xs),minX=Math.min(...xs);
 const sv=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
 const X=v=>P+(v-minX)/(maxX-minX||1)*(W-2*P),Y=v=>H-P-v/maxY*(H-2*P);
 sv.appendChild(el("line",{x1:P,y1:H-P,x2:W-P,y2:H-P,stroke:"#ddd"}));
 sv.appendChild(el("line",{x1:P,y1:P,x2:P,y2:H-P,stroke:"#ddd"}));
 for(const g of [0.25,0.5,0.75,1]){const y=Y(maxY*g);sv.appendChild(el("line",{x1:P,y1:y,x2:W-P,y2:y,stroke:"#eee"}));
  const t=el("text",{x:4,y:y+3,fill:"#777"});t.textContent=(maxY*g*1000).toFixed(0)+"ms";sv.appendChild(t)}
 for(const s of series){let d="";s.vals.forEach((v,i)=>{d+=(i?"L":"M")+X(xs[i]).toFixed(1)+" "+Y(v).toFixed(1)});sv.appendChild(el("path",{d,fill:"none",stroke:s.color,"stroke-width":1.5}))}
 c.appendChild(sv)}
function barChart(id,xs,vals){const c=document.getElementById(id);if(!xs||!xs.length){return}
 const W=520,H=200,P=36,maxY=Math.max(...vals),n=vals.length;
 const sv=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
 const bw=(W-2*P)/n;
 vals.forEach((v,i)=>{const h=v/maxY*(H-2*P);
  sv.appendChild(el("rect",{x:P+i*bw+1,y:H-P-h,width:Math.max(1,bw-2),height:h,fill:"#369",opacity:.75}))});
 sv.appendChild(el("line",{x1:P,y1:H-P,x2:W-P,y2:H-P,stroke:"#ddd"}));
 const mx=Math.max(...xs);
 const t=el("text",{x:4,y:14,fill:"#777"});t.textContent="max "+(mx*1000).toFixed(1)+"ms";sv.appendChild(t);
 c.appendChild(sv)}
lineChart("c_hist_get",D.hist_get[0],[{vals:D.hist_get[1],color:"#369"}]);
barChart("c_hist_put",D.hist_put[0],D.hist_put[1]);
(function(){const c=document.getElementById("c_scatter");const W=520,H=200,P=36;
 const all=[...D.scatter.load,...D.scatter.save];
 if(!all.length){c.appendChild(el("text",{x:10,y:30,fill:"#777"})).textContent="no bytes data";return}
 const mx=Math.max(...all.map(p=>p[0])),my=Math.max(...all.map(p=>p[1]));
 const sv=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
 const X=v=>P+v/mx*(W-2*P),Y=v=>H-P-v/my*(H-2*P);
 for(const p of D.scatter.load)sv.appendChild(el("circle",{cx:X(p[0]),cy:Y(p[1]),r:3,fill:"none",stroke:"#369"}));
 for(const p of D.scatter.save)sv.appendChild(el("path",{d:`M${X(p[0])-3} ${Y(p[1])-3}L${X(p[0])+3} ${Y(p[1])+3}M${X(p[0])-3} ${Y(p[1])+3}L${X(p[0])+3} ${Y(p[1])-3}`,stroke:"#c33"}));
 sv.appendChild(el("line",{x1:P,y1:H-P,x2:W-P,y2:H-P,stroke:"#ddd"}));
 sv.appendChild(el("line",{x1:P,y1:P,x2:P,y2:H-P,stroke:"#ddd"}));
 const t=el("text",{x:P+4,y:14,fill:"#777"});t.textContent=(mx/(1<<20)).toFixed(0)+" MiB / "+(my*1000).toFixed(0)+"ms";sv.appendChild(t);
 c.appendChild(sv)})();
lineChart("c_trend",D.trend.load.map(p=>p[0]),[{vals:D.trend.load.map(p=>p[1]),color:"#369"},{vals:D.trend.save.map(p=>p[1]),color:"#c33"}]);
</script></body></html>"""


def gen_demo(recs: Records) -> None:
    rng = random.Random(42)
    for i in range(120):
        keys = rng.randint(16, 128)
        nbytes = keys * rng.choice([2, 4, 8]) * (1 << 20)
        get_t = nbytes / rng.uniform(10, 14) / (1 << 30)
        recs.load_detail.append({"req_id": f"r{i}", "keys": keys, "bytes": nbytes,
                                 "total": get_t + 0.008, "build_key": rng.uniform(.002, .006),
                                 "prepare": rng.uniform(.002, .005), "get": get_t,
                                 "result": .001, "other": rng.uniform(.001, .003)})
        put_t = nbytes / rng.uniform(9, 13) / (1 << 30)
        recs.save_detail.append({"req_id": f"r{i}", "keys": keys, "bytes": nbytes,
                                 "queue_wait": rng.uniform(.002, .02) + (rng.uniform(0, .04) if i % 17 == 0 else 0),
                                 "total": put_t + .02, "build_key": rng.uniform(.002, .005),
                                 "lookup": rng.uniform(.003, .007), "prepare": rng.uniform(.002, .005),
                                 "sync": rng.uniform(.012, .025), "put": put_t,
                                 "other": rng.uniform(.002, .006)})
    for i in range(40):
        recs.load_summary.append({"n": 3, "start_load_kv": rng.uniform(.08, .15), "load_async": "False", "use_layerwise": "False"})
        recs.save_summary.append({"n": 3, "wait_for_save": rng.uniform(.1, .2)})
    for i in range(120):
        ttft = .35 + rng.uniform(0, .4)
        recs.breakdown.append({"req_id": f"r{i}", "ttft": ttft, "hash": .01, "ext_lookup": .02,
                               "alloc": .015, "overhead": .02, "load_kv": rng.uniform(.08, .14),
                               "forward": rng.uniform(.1, .2), "save_kv": .005, "update": .01,
                               "residual": rng.uniform(0, .05)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("-o", "--out", default="ttft_report.html")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    recs = Records()
    if args.demo:
        gen_demo(recs)
    elif not args.logs:
        ap.error("provide log files or --demo")
        sys.exit(2)
    else:
        parse_files(args.logs, recs)
        total = (len(recs.load_detail) + len(recs.save_detail)
                 + len(recs.load_summary) + len(recs.save_summary) + len(recs.breakdown))
        if total == 0:
            print("warning: no TTFT records found in input", file=sys.stderr)
    out = render(recs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {args.out}: {len(recs.load_detail)} loads, {len(recs.save_detail)} saves, "
          f"{len(recs.breakdown)} breakdowns")


if __name__ == "__main__":
    main()
