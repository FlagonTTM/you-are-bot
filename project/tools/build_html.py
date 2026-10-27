#!/usr/bin/env python3
"""Build the offline YouAreBot HTML report; Python standard library only.

Inputs are read-only curated evidence, never a fresh model/LLM invocation.
Unknown evidence schemas fail explicitly; an absent/incomplete reproduction proof
renders PENDING, not a successful run. Run from any directory:
    python3 project/tools/build_html.py --check
Browser acceptance belongs to the parent and is not claimed by these checks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import quote
from typing import Any

SCHEMA = "you-are-bot.coursework.facts.v1"
REPO = "https://github.com/FlagonTTM/you-are-bot"
BRANCH = "feature/coursework-html"
PAPER, INK, TEAL, LIME, RED = "#f4f2e9", "#172a35", "#123c3c", "#d8ee64", "#a13729"
SECTION_IDS = (
    "question", "task", "data", "pairs", "length", "validation", "features",
    "models", "llm", "negative-results", "reproduction", "tracking", "service", "conclusion",
)


def h(value: object) -> str:
    return escape(str(value), quote=True)


def number(value: object, digits: int = 0) -> str:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        return "UNKNOWN"
    return f"{value:,.{digits}f}".replace(",", "\u2009")


def safe_json(value: object) -> str:
    # A JSON script element is raw text: HTML escaping is not sufficient here.
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    def __init__(self, root: Path):
        self.root = root
        self.sources: dict[str, dict] = {}

    def read(self, relative: str, label: str = "") -> Any:
        path = self.root / relative
        if not path.is_file():
            raise ValueError(f"UNKNOWN: required source not found: {relative}")
        self.sources[relative] = {"path": relative, "sha256": sha(path), "label": label or path.name}
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix == ".json" else text

    def footer(self, paths: list[str], locator: str) -> str:
        links = " · ".join(f'<a href="{quote(p, safe="/-.")}">{h(Path(p).name)}</a>' for p in paths)
        return f'<footer class="source"><span>{links}</span><span>{h(locator)}</span></footer>'


def svg_open(key: str, width: int, height: int, title: str, description: str) -> str:
    return (f'<svg class="chart chart-{h(key)}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="{key}-title {key}-desc" xmlns="http://www.w3.org/2000/svg">'
            f'<title id="{key}-title">{h(title)}</title><desc id="{key}-desc">{h(description)}</desc>')


def text(x: float, y: float, value: object, size: int = 30, fill: str = INK,
         anchor: str = "start", weight: str = "normal", extra: str = "") -> str:
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{h(value)}</text>')


def rect(x: float, y: float, width: float, height: float, fill: str, extra: str = "") -> str:
    return f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" fill="{fill}" {extra}/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = INK, extra: str = "") -> str:
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="2" {extra}/>'


def model_chart(models: list[dict]) -> str:
    labels = {
        "behavior-v2": "Поведение / числовые",
        "lexical-v1": "TF-IDF + LogReg",
        "hybrid-v1": "Текст + числовые",
        "template-v1": "Шаблоны + CatBoost",
        "template-text-v2": "H7 · + native text",
        "sequence-full-v2": "H8 · + ByteCNN / GRU, смесь",
    }
    desc = "; ".join(f'{m["label"]}: {m["logloss"]:.9f}' for m in models)
    out = [svg_open("cv", 1710, 610, "Сравнение шести моделей: development, пять folds", desc)]
    left, plot, maximum = 560, 910, 0.24
    for tick in (0, 0.05, 0.10, 0.15, 0.20):
        x = left + tick / maximum * plot
        out += [line(x, 20, x, 530, INK, 'opacity=".18"'), text(x, 582, f"{tick:.2f}", 26, anchor="middle")]
    for i, m in enumerate(models):
        y = 30 + i * 82
        best = m["id"] == "template-v1"
        out += [text(0, y + 37, labels.get(m["id"], m["label"]), 32, weight="bold" if best else "normal"),
                rect(left, y, m["logloss"] / maximum * plot, 52, LIME if best else TEAL,
                     f'data-model="{h(m["id"])}" data-value="{m["logloss"]!r}"'),
                text(1700, y + 37, f'{m["logloss"]:.6f}', 32, anchor="end", weight="bold" if best else "normal")]
    return '<div class="chart-scroll" tabindex="0" role="region" aria-label="Диаграмма; при необходимости прокручивается по горизонтали">' + "".join(out) + "</svg></div>"


def length_chart(facts: dict) -> str:
    desc = []
    for name in ("train", "test"):
        d = facts[name]
        desc += [f'{name}: {k}, {v}, {v/d["dialogues"]:.6%}' for k, v in d["length_histogram"].items()]
    out = [svg_open("length-hist", 1140, 550, "Длины диалогов TRAIN и TEST, доля диалогов", "; ".join(desc))]
    for tick in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = 440 - tick / 0.5 * 340
        out += [line(85, y, 1110, y, INK, 'opacity=".17"'), text(65, y + 9, f"{tick:.0%}", 26, anchor="end")]
    for i, label in enumerate(facts["train"]["length_histogram"]):
        x = 152 + i * 250
        for j, name in enumerate(("train", "test")):
            count = facts[name]["length_histogram"][label]
            ratio = count / facts[name]["dialogues"]
            height = ratio / 0.5 * 340
            out += [rect(x + j * 85, 440 - height, 65, height, TEAL if j == 0 else LIME,
                         f'data-series="{name}" data-count="{count}" data-value="{ratio!r}"'),
                    text(x + j * 85 + 32, 425 - height, f"{ratio:.1%}", 24, anchor="middle"),
                    text(x + j * 85 + 32, 475, count, 26, anchor="middle")]
        out.append(text(x + 75, 526, label, 30, anchor="middle", weight="bold"))
    return '<div class="chart-scroll" tabindex="0" role="region" aria-label="Диаграмма; при необходимости прокручивается по горизонтали">' + "".join(out) + "</svg></div>"


def pair_chart(facts: dict) -> str:
    pairs = dict(facts["train_label_pairs"])
    # An omitted pair can only be reconstructed as zero after reconciliation.
    remainder = facts["train"]["dialogues"] - sum(pairs.values())
    if "11" not in pairs:
        if remainder != 0:
            raise ValueError("UNKNOWN: incomplete TRAIN pair counts")
        pairs["11"] = 0
    desc = "; ".join(f"{k}: {v} ({v / facts['train']['dialogues']:.2%})" for k, v in pairs.items())
    out = [svg_open("pairs-bars", 1050, 500, "Пары меток TRAIN: сторона 0, сторона 1", desc)]
    for i, pair in enumerate(("10", "01", "00", "11")):
        count = pairs[pair]
        y = 35 + i * 115
        out += [text(0, y + 33, pair, 38, weight="bold"),
                rect(100, y, count / facts["train"]["dialogues"] * 1380, 48, LIME if pair == "00" else TEAL,
                     f'data-pair="{pair}" data-count="{count}"'),
                text(1040, y + 33, f'{count} / {count / facts["train"]["dialogues"]:.1%}', 30, anchor="end")]
    return '<div class="chart-scroll" tabindex="0" role="region" aria-label="Диаграмма; при необходимости прокручивается по горизонтали">' + "".join(out) + "</svg></div>"


def two_bars(key: str, rows: list[tuple[str, float]], maximum: float, dark: bool = False) -> str:
    color = PAPER if dark else INK
    out = [svg_open(key, 1010, 110 * len(rows) + 85, "LogLoss: ниже лучше", "; ".join(f"{a}: {b}" for a, b in rows))]
    left, plot = 275, 495
    for i, (label, value) in enumerate(rows):
        y = 22 + i * 110
        out += [text(0, y + 35, label, 30, color),
                rect(left, y, value / maximum * plot, 53, LIME if dark or i == 0 else TEAL,
                     f'data-value="{value!r}"'), text(1000, y + 35, f"{value:.6f}", 32, color, "end")]
    y = 110 * len(rows) + 15
    out += [line(left, y, left + plot, y, color), text(left, y + 42, "0", 24, color),
            text(left + plot, y + 42, f"{maximum:.2f}", 24, color, "end")]
    return '<div class="chart-scroll" tabindex="0" role="region" aria-label="Диаграмма; при необходимости прокручивается по горизонтали">' + "".join(out) + "</svg></div>"


def component_diagram() -> str:
    out = [svg_open("component", 1040, 265, "Как строится группа валидации", "Две стороны диалога, точные дубликаты и явные UUID-ссылки объединяются в компоненту; компонента целиком в одном fold.")]
    for y, label in ((35, "Две стороны диалога"), (110, "Точный дубликат"), (185, "Явная UUID-ссылка")):
        out += [rect(0, y, 435, 55, PAPER, f'stroke="{INK}" stroke-width="2"'),
                text(22, y + 37, label, 28), line(435, y + 27, 535, 137)]
    out += [rect(535, 93, 460, 90, TEAL), text(765, 148, "Одна компонента → один fold", 28, PAPER, "middle")]
    return '<div class="chart-scroll" tabindex="0" role="region" aria-label="Диаграмма; при необходимости прокручивается по горизонтали">' + "".join(out) + "</svg></div>"


def service_diagram(test_rows: int) -> str:
    desc = ("Сплошная линия: реализованный offline pipeline. Данные, валидация, признаки, CPU CatBoost, CSV. "
            "Пунктир: не реализованные HTTP POST /predict, проверка схемы и лимитов, extractor, версионированная модель, вероятности и версия. "
            "LLM-очередь опциональна и не реализована; трекер отделён от serving.")
    out = [svg_open("architecture", 1720, 590, "Граница offline реализации и эскиза HTTP-сервиса", desc)]
    out += [text(0, 36, "РЕАЛИЗОВАНО · OFFLINE", 26, weight="bold")]
    nodes = [(0, "Данные", "validate / EDA"), (430, "Признаки", "общий extractor"),
             (860, "CPU CatBoost", "fit / checkpoint"), (1290, "CSV", f"{test_rows} вероятностей")]
    for i, (x, a, b) in enumerate(nodes):
        out += [rect(x, 70, 355, 105, TEAL), text(x + 22, 112, a, 32, PAPER), text(x + 22, 151, b, 25, PAPER)]
        if i < 3:
            out += [line(x + 355, 123, x + 423, 123), text(x + 389, 111, "→", 34, anchor="middle")]
    out += [text(0, 260, "ЭСКИЗ · HTTP / QUEUE НЕ РЕАЛИЗОВАНЫ", 26, weight="bold")]
    proposed = [(0, "POST /predict", "JSON диалога"), (350, "Схема + лимиты", "входной контракт"),
                (700, "Extractor", "версия признаков"), (1050, "Native CPU model", "model_version"),
                (1400, "Ответ", "p₀, p₁ + версия")]
    for i, (x, a, b) in enumerate(proposed):
        out += [rect(x, 298, 310, 105, PAPER, f'stroke="{INK}" stroke-width="2" stroke-dasharray="10 7"'),
                text(x + 16, 340, a, 27), text(x + 16, 378, b, 24)]
        if i < 4:
            out += [line(x + 310, 350, x + 345, 350, INK, 'stroke-dasharray="7 5"')]
    out += [line(1205, 403, 1205, 455, INK, 'stroke-dasharray="7 5"'),
            rect(1010, 455, 690, 72, PAPER, f'stroke="{INK}" stroke-width="2" stroke-dasharray="10 7"'),
            text(1030, 500, "Опциональная LLM-очередь · не реализована", 27),
            rect(0, 455, 805, 72, PAPER, f'stroke="{INK}" stroke-width="2"'),
            text(22, 500, "Эксперименты / registry — отдельно от serving", 28),
            text(0, 580, "Это границы компонентов, не заявление о production, SLA или нагрузке.", 27)]
    return '<div class="chart-scroll" tabindex="0" role="region" aria-label="Диаграмма; при необходимости прокручивается по горизонтали">' + "".join(out) + "</svg></div>"


CSS = r"""
:root{--paper:#f4f2e9;--ink:#172a35;--teal:#123c3c;--lime:#d8ee64;--red:#a13729;--rule:rgba(23,42,53,.25)}
*{box-sizing:border-box}html{scroll-padding-top:28px}html,body{margin:0;width:100%;background:var(--paper);color:var(--ink)}body{font:22px/1.5 Arial,Helvetica,sans-serif}a{color:inherit;text-underline-offset:.18em}button{font:inherit;cursor:pointer}a,button{-webkit-tap-highlight-color:transparent}:focus-visible{outline:3px solid var(--teal);outline-offset:4px}.dark :focus-visible{outline-color:var(--lime)}
.skip{position:fixed;left:16px;top:-100px;z-index:10;background:var(--lime);padding:12px;font-size:18px}.skip:focus{top:12px}.report-layout{max-width:1920px;display:grid;grid-template-columns:235px minmax(0,1fr);gap:48px;margin:auto;padding:30px 42px 70px}.toc{position:sticky;top:24px;align-self:start;max-height:calc(100vh - 48px);overflow-y:auto;padding-right:10px;font-size:15px}.toc-title{font:17px Menlo,Consolas,monospace;margin:0 0 22px}.toc ol{list-style:none;counter-reset:toc;margin:0;padding:0}.toc li{counter-increment:toc;border-top:1px solid var(--rule)}.toc a{display:flex;gap:12px;align-items:center;min-height:44px;padding:10px 0;text-decoration:none;line-height:1.35}.toc a::before{content:counter(toc,decimal-leading-zero);font:13px Menlo,Consolas,monospace;opacity:.7}.toc a:hover,.toc a[aria-current="location"]{text-decoration:underline}.toc button{margin-top:20px;min-height:44px;padding:10px 14px;background:var(--ink);color:var(--paper);border:0;font-size:16px}.toc .toc-note{font-size:14px;line-height:1.5;margin-top:18px;opacity:.8}.report-main{min-width:0}.report-section{position:relative;padding:58px 54px 42px;border-bottom:1px solid var(--rule);scroll-margin-top:28px}.report-section.dark{background:var(--ink);color:var(--paper);--rule:rgba(244,242,233,.3)}.report-section:first-child{padding-top:48px}.kicker{font:16px/1.5 Menlo,Consolas,monospace;letter-spacing:.05em;text-transform:uppercase;margin:0 0 20px;display:flex;justify-content:space-between;gap:24px}.kicker span:last-child{letter-spacing:0;text-transform:none;flex:none}h1,h2,h3,p{margin:0}h1,h2,h3{font-family:Georgia,'Times New Roman',serif;font-weight:400}h1{font-size:68px;line-height:1.06;letter-spacing:-.035em;max-width:1200px}h2{font-size:46px;line-height:1.12;letter-spacing:-.025em}h3{font-size:31px;line-height:1.2}.dek{margin-top:20px;font-size:22px;max-width:1120px}.content{margin-top:34px;display:flex;flex-direction:column;gap:32px}.two{display:grid;grid-template-columns:1.2fr .8fr;gap:48px;align-items:center}.two>*{min-width:0}.wide-left{grid-template-columns:1.65fr 1fr}.equal{grid-template-columns:1fr 1fr}.align-top{align-items:start}.stack{display:flex;flex-direction:column;gap:24px}.stack.tight{gap:16px}.lead,.emphasis{font:36px/1.2 Georgia,'Times New Roman',serif}.small,.caption{font-size:18px;line-height:1.5}.muted{opacity:.8}.mono,code{font-family:Menlo,Consolas,monospace}.mono{font-size:18px;overflow-wrap:anywhere}.accent{color:var(--lime)}.negative{color:var(--red)}.rule{border-top:1px solid var(--rule);padding-top:22px}.takeaway{font-size:25px;line-height:1.4;border-top:1px solid var(--rule);padding-top:24px;margin-top:6px}.pill{display:inline-block;padding:7px 12px;font:15px/1.4 Menlo,Consolas,monospace;border:1px solid currentColor;width:fit-content}.pending{background:var(--lime);color:var(--ink);padding:10px 14px;display:inline-block;font:16px/1.5 Menlo,Consolas,monospace}.verified{background:var(--teal);color:var(--paper);padding:10px 14px;display:inline-block;font:16px/1.5 Menlo,Consolas,monospace}.source{display:flex;justify-content:space-between;gap:28px;align-items:flex-start;padding-top:18px;border-top:1px solid var(--rule);font-size:15px;line-height:1.5;margin-top:34px;overflow-wrap:anywhere}.source span:first-child{max-width:72%}.source span:last-child{text-align:right}.source a{display:inline-block;min-height:44px;padding:6px 0}.chart{width:100%;height:auto;display:block;font-family:Arial,Helvetica,sans-serif}figure{margin:0;min-width:0}figcaption{font-size:18px;margin-top:18px}.chart-scroll{overflow-x:auto;max-width:100%;padding:3px 0 10px}.chart-scroll:focus-visible{outline-offset:-3px}.chart-scroll .chart{min-width:760px}.chart-scroll .chart-architecture{min-width:1200px}.chart-scroll .chart-cv{min-width:1000px}table{border-collapse:collapse;width:100%;font-size:22px}th,td{padding:15px 12px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}th{font-size:17px;font-weight:400}td:first-child,th:first-child{padding-left:0}td:last-child,th:last-child{padding-right:0}.numeric{text-align:right;font-variant-numeric:tabular-nums}.compact td,.compact th{padding-top:12px;padding-bottom:12px}.schema td:first-child{font:19px/1.5 Menlo,Consolas,monospace;width:245px}.schema td{font-size:21px}.hero-content{grid-template-columns:1.1fr .9fr;gap:70px;align-items:center}.hero-sub{font-size:23px;margin-top:25px}.hero-author{font-size:21px;margin-top:32px}.quote{font:40px/1.2 Georgia,'Times New Roman',serif;border-top:1px solid var(--rule);padding:24px 0}.quote:last-of-type{margin-left:36px}.probabilities{font:32px/1.3 Menlo,Consolas,monospace;display:flex;gap:35px;align-items:center;padding:20px 0}.formula{font:32px/1.4 Georgia,'Times New Roman',serif}.definition{display:grid;grid-template-columns:75px 1fr;gap:20px;border-top:1px solid var(--rule);padding-top:23px}.definition .symbol{font:32px/1.4 Menlo,Consolas,monospace}.legend{display:flex;flex-wrap:wrap;gap:26px;font-size:18px;align-items:center}.swatch{display:inline-block;width:24px;height:16px;margin-right:10px;background:var(--teal)}.swatch.test{background:var(--lime);border:1px solid var(--ink)}.class-track{display:flex;height:23px;width:100%;margin:16px 0 22px}.class-track span:first-child{background:var(--teal)}.class-track span:last-child{background:var(--lime)}.stat-line{display:flex;justify-content:space-between;gap:24px;border-bottom:1px solid var(--rule);padding:12px 0;font-size:23px}.stat-line strong{font-weight:400;font-variant-numeric:tabular-nums}.folds{display:flex;gap:6px}.fold{background:var(--teal);color:var(--paper);text-align:center;padding:17px 6px;flex:var(--rows);font-size:21px}.fold span{display:block;font-size:16px;margin-top:8px}.recipe{font-size:20px;display:grid;grid-template-columns:1fr 1fr;gap:0 28px}.recipe div{border-bottom:1px solid var(--rule);padding:12px 0;display:flex;justify-content:space-between;gap:14px}.recipe dt,.recipe dd{margin:0}.recipe dt{font-size:17px}.recipe dd{font:18px/1.5 Menlo,Consolas,monospace}.flowline{display:flex;align-items:center;gap:14px;font-size:20px}.flowline>span{flex:1;min-width:0;padding:16px 12px;border:1px solid var(--rule)}.flowline>b{font-weight:400;font-size:28px}.flowline small{display:block;font-size:16px;margin-top:9px}.command{display:block;white-space:normal;overflow-wrap:anywhere;padding:23px 26px;background:var(--ink);color:var(--lime);font-size:22px;line-height:1.6}.proof{display:grid;grid-template-columns:1fr 1fr;gap:44px;align-items:start;border-top:1px solid var(--rule);padding-top:25px}.proof-value{font:19px/1.5 Menlo,Consolas,monospace;overflow-wrap:anywhere}.tracking-record{padding:24px;border:1px solid var(--ink);font-size:21px}.tracking-record dl{display:grid;grid-template-columns:115px 1fr;gap:17px;margin:22px 0 0}.tracking-record dt,.tracking-record dd{margin:0}.tracking-record dd{font:18px/1.5 Menlo,Consolas,monospace;overflow-wrap:anywhere}.tracking-list{list-style:none;margin:0;padding:0}.tracking-list li{border-top:1px solid var(--rule);padding:16px 0;font-size:21px}.tracking-list li:first-child{border-top:1px solid var(--ink)}.rubric td{padding-top:12px;padding-bottom:12px}.rubric td:last-child{width:50px;text-align:right;font:24px/1.5 Menlo,Consolas,monospace}.public-score{font:38px/1.2 Menlo,Consolas,monospace}.repo-link{font:18px/1.5 Menlo,Consolas,monospace;overflow-wrap:anywhere;display:inline-block;padding:10px 0;min-height:44px}.outcome{font:32px/1.25 Georgia,'Times New Roman',serif}.backtop{display:inline-block;min-height:44px;padding:10px 0;font-size:17px;margin-top:20px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
dialog{color:var(--ink);background:var(--paper);border:1px solid var(--ink);max-width:min(1050px,94vw);max-height:88dvh;padding:32px;font-size:19px;line-height:1.5}dialog::backdrop{background:rgba(23,42,53,.85)}dialog h2{font-size:36px;margin-right:70px}dialog .close{position:absolute;top:24px;right:24px;min-width:44px;min-height:44px;padding:8px 12px;border:1px solid var(--ink);background:transparent;color:var(--ink);font:20px Arial}.source-list{list-style:none;padding:0;margin-top:25px}.source-list li{padding:19px 0;border-top:1px solid var(--rule)}.source-list a{display:inline-block;min-height:44px;overflow-wrap:anywhere;padding:7px 0}.digest{font:13px/1.6 Menlo,Consolas,monospace;overflow-wrap:anywhere;display:block}.sources-fallback{margin:40px 54px;font-size:18px}.sources-fallback summary{cursor:pointer;min-height:44px;padding:10px 0}.sources-fallback :focus-visible{outline-color:var(--teal)}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto!important}*{animation:none!important;transition:none!important}}
@media(max-width:1500px){.report-layout{grid-template-columns:200px minmax(0,1fr);gap:22px;padding:24px}.report-section{padding:44px 30px 32px}.two,.two.hero-content,.two.wide-left,.two.equal,.proof{grid-template-columns:1fr;gap:30px}.hero-content{display:grid}.hero-content>div:last-child{max-width:650px}.hero-content .quote{display:inline-block;vertical-align:top;margin:20px 28px 15px 0;font-size:34px}.hero-content .quote:last-of-type{margin-left:20px}.hero-author{margin-top:22px}.source{display:block}.source span{display:block;max-width:none!important;text-align:left!important}.source span:last-child{margin-top:7px}.recipe{max-width:850px}.sources-fallback{margin:30px}.flowline{gap:10px;font-size:18px}.flowline>span{padding:14px 10px}.flowline small{font-size:15px}h1{font-size:61px}h2{font-size:40px}}
@media(max-width:1000px){.report-layout{display:block;padding:20px}.toc{position:relative;top:0;max-height:none;overflow:visible;padding:0 12px 24px}.toc ol{columns:2;column-gap:30px}.toc li{break-inside:avoid}.toc-title{margin-bottom:15px}.toc .toc-note{display:inline;margin-left:16px}.toc button{margin-top:10px}.report-section{padding:38px 28px 30px}.flowline{flex-direction:column;align-items:stretch;font-size:21px}.flowline>b{align-self:center;transform:rotate(90deg)}.flowline small{font-size:17px}.content{gap:28px}.chart-scroll .chart{min-width:760px}}
@media(max-width:600px){html{scroll-padding-top:12px}body{font-size:19px;line-height:1.5}.report-layout{padding:12px 0 35px}.toc{padding:18px 20px 28px}.toc ol{columns:1}.toc a{min-height:44px}.toc .toc-note{display:block;margin:14px 0 0}.report-section,.report-section:first-child{padding:32px 20px 28px;scroll-margin-top:12px}.kicker{font-size:12px;gap:10px;letter-spacing:0;margin-bottom:18px}.kicker span:first-child{max-width:80%}h1{font-size:43px}h2{font-size:34px}h3{font-size:27px}.dek,.hero-sub{font-size:20px}.lead,.emphasis{font-size:30px}.small,.caption,figcaption,.legend{font-size:17px}.takeaway{font-size:22px}.hero-content .quote{font-size:30px;margin-right:16px}.hero-author{font-size:19px}.probabilities{font-size:28px}.mono{font-size:16px}.source{font-size:14px}.schema td{font-size:17px}.schema td:first-child{font-size:14px;width:145px;overflow-wrap:anywhere}table{font-size:17px}th{font-size:15px}td,th{padding:12px 7px}.compact{font-size:16px}.definition{grid-template-columns:45px 1fr;gap:14px}.definition .symbol{font-size:26px}.formula{font-size:25px}.recipe{grid-template-columns:1fr;gap:0}.recipe dt,.recipe dd{font-size:16px}.folds{flex-wrap:wrap}.fold{flex:1 1 27%;font-size:19px}.fold span{font-size:15px}.stat-line{font-size:20px}.command{padding:18px;font-size:16px}.pending,.verified{font-size:14px}.tracking-record{padding:19px}.tracking-record dl{grid-template-columns:80px 1fr;gap:13px;font-size:17px}.tracking-record dd{font-size:15px}.tracking-list li{font-size:19px}.public-score{font-size:29px}.outcome{font-size:29px}.repo-link{font-size:15px}.chart-scroll .chart{min-width:720px}.chart-scroll::after{content:'Диаграмма прокручивается по горизонтали →';display:block;font-size:14px;padding-top:10px}.sources-fallback{margin:30px 20px}dialog{padding:25px 20px;font-size:17px}dialog h2{font-size:29px}.digest{font-size:11px}}
@media print{@page{size:A4;margin:16mm}html,body{background:#fff;width:auto;font-size:12pt;line-height:1.4}.report-layout{display:block;margin:0;padding:0}.toc,.skip,dialog,.backtop{display:none!important}.report-section,.report-section:first-child{padding:18pt 0;break-inside:auto;scroll-margin:0}.report-section.dark{color:var(--ink);background:transparent;--rule:rgba(23,42,53,.3)}.accent{color:var(--ink)}h1{font-size:32pt}h2{font-size:24pt;break-after:avoid}h3{font-size:18pt}.content{margin-top:18pt;gap:16pt}.two,.two.hero-content,.two.wide-left,.two.equal,.proof{display:block}.two>*+*,.proof>*+*{margin-top:18pt}.chart-scroll{overflow:visible}.chart-scroll .chart{min-width:0}.chart-scroll::after{display:none}.small,.caption,figcaption{font-size:10pt}.source{font-size:9pt;margin-top:18pt;display:block}.source span{max-width:none!important;text-align:left!important}.source a{min-height:0}.chart,table,.tracking-record{break-inside:avoid}.hero-content .quote{display:block;margin:12pt 0;font-size:20pt}.hero-sub,.hero-author{font-size:12pt}.lead,.emphasis,.outcome{font-size:20pt}.probabilities{font-size:18pt}.public-score{font-size:23pt}.sources-fallback{margin:20pt 0}.digest{font-size:8pt}a{text-decoration:none}.command{font-size:10pt}.flowline{display:block}.flowline>span{display:block;margin:8pt 0}.flowline>b{display:none}svg{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
"""

JS = r"""
(() => {
  'use strict';
  const sections = Array.from(document.querySelectorAll('.report-section'));
  const links = Array.from(document.querySelectorAll('.toc a[data-section]'));
  const dialog = document.getElementById('sources-dialog');
  const sourcesButton = document.getElementById('open-sources');
  function mark(id) {
    links.forEach(link => {
      if (link.dataset.section === id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  function hashId() { try { return decodeURIComponent(location.hash.slice(1)); } catch (_) { return ''; } }
  // All sections are visible before JavaScript; native anchors work with JS off.
  mark(sections.some(section => section.id === hashId()) ? hashId() : sections[0].id);
  links.forEach(link => link.addEventListener('click', () => mark(link.dataset.section)));
  window.addEventListener('hashchange', () => mark(hashId()));
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length) mark(visible[0].target.id);
    }, {rootMargin:'-5% 0px -60% 0px', threshold:0});
    sections.forEach(section => observer.observe(section));
  }
  if (typeof dialog.showModal === 'function') {
    sourcesButton.hidden = false;
    sourcesButton.addEventListener('click', () => dialog.showModal());
    document.getElementById('close-sources').addEventListener('click', () => dialog.close());
  }
  // Do not hijack Arrow, Page, Home, End or Space: this is a continuous report.
})();
"""


def proof_state(value: object) -> tuple[bool, str]:
    if value is None:
        return False, "PENDING · запуск ещё не подтверждён"
    if not isinstance(value, dict):
        return False, "UNKNOWN · неизвестная схема proof"
    if "schema" in value and value["schema"] != "coursework.reproduction-proof.v1":
        return False, "UNKNOWN · неизвестная схема proof"
    tracking = value.get("tracking")
    required = {"status", "run_id", "oof_logloss", "oof_rows", "test_rows", "tracking", "checks", "command"}
    if not required.issubset(value) or not isinstance(tracking, dict):
        return False, "UNKNOWN · неполный proof"
    tracking_fields = {"backend", "run_status", "database", "logged_parameters", "logged_metrics", "logged_artifacts"}
    if not tracking_fields.issubset(tracking):
        return False, "UNKNOWN · неполная схема tracking"
    score = value.get("oof_logloss")
    valid_score = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
    valid_counts = all(isinstance(tracking[k], int) and not isinstance(tracking[k], bool) and tracking[k] > 0
                       for k in ("logged_parameters", "logged_metrics", "logged_artifacts"))
    checks = value.get("checks")
    checks_pass = isinstance(checks, dict) and bool(checks) and all(v is True or v == "PASS" for v in checks.values())
    verified = (value["status"] == "VERIFIED" and tracking["run_status"] == "FINISHED"
                and tracking["backend"] == "SQLite+JSON" and value["oof_rows"] == 2416
                and value["test_rows"] == 758 and valid_score and isinstance(score, (int, float)) and 0 <= score
                and valid_counts and checks_pass and isinstance(value["run_id"], str) and bool(value["run_id"])
                and isinstance(value["command"], str) and bool(value["command"]))
    return bool(verified), "VERIFIED · native baseline" if verified else "PENDING · proof не прошёл проверку"


def build_document(project: Path) -> tuple[str, dict]:
    evidence = Evidence(project)
    facts_path = "evidence/session-facts.json"
    facts = evidence.read(facts_path, "Пересчитанные факты по официальным данным")
    if facts.get("schema") != SCHEMA:
        raise ValueError(f'UNKNOWN: unsupported facts schema: {facts.get("schema")!r}')
    tr, te = facts["train"], facts["test"]
    if sum(tr["length_histogram"].values()) != tr["dialogues"] or sum(te["length_histogram"].values()) != te["dialogues"]:
        raise ValueError("UNKNOWN: length histograms do not reconcile")
    if sum(facts["train_class_counts"].values()) != tr["participant_rows"]:
        raise ValueError("UNKNOWN: participant class counts do not reconcile")
    if sum(facts["split_rows"].values()) != tr["participant_rows"]:
        raise ValueError("UNKNOWN: split rows do not reconcile")
    models = facts["models"]
    if len(models) != 6 or any(m["contour"] != "development5fold" or m["rows"] != facts["development_rows"] for m in models):
        raise ValueError("UNKNOWN: model comparison is not a matched six-model contour")
    reports = {}
    for model in models:
        report = evidence.read(model["source"], model["label"])
        if evidence.sources[model["source"]]["sha256"] != model["source_sha256"]:
            raise ValueError(f'UNKNOWN: source hash mismatch: {model["source"]}')
        metric = report["metrics"]["primary"]["logloss"] if model["id"] == "sequence-full-v2" else report["metrics"]["logloss"]
        if metric != model["logloss"]:
            raise ValueError(f'UNKNOWN: source metric mismatch: {model["id"]}')
        membership = report.get("provenance", {}).get("sha256", report.get("source_closure", {})).get("evidence/membership-v2.json")
        if membership != facts["membership_sha256"]:
            raise ValueError(f'UNKNOWN: membership mismatch: {model["id"]}')
        reports[model["id"]] = report
    sequence = reports["sequence-full-v2"]
    conf_path = "evidence/history/confirmation-decision.json"
    confirmation = evidence.read(conf_path, "Однократная confirmation: native против смеси с LLM")
    if confirmation["against_template"] != facts["confirmation"]["against_template"]:
        raise ValueError("UNKNOWN: confirmation differs from cited report")
    conf = confirmation["against_template"]
    encoder_path = "evidence/history/encoder-pilot-report.json"
    encoder = evidence.read(encoder_path, "H9: только pilot fold 0")
    if encoder["metrics"] != facts["last_encoder_pilot"]["metrics"] or encoder["comparisons"] != facts["last_encoder_pilot"]["comparisons"]:
        raise ValueError("UNKNOWN: H9 differs from cited report")
    official_path = "evidence/history/official-submission-10588.json"
    official_report = evidence.read(official_path, "Исторический снимок собственной официальной отправки")
    official = facts["official"]
    if float(official_report["scores"][0]["score"]) != official["public_logloss"]:
        raise ValueError("UNKNOWN: official score does not match source")
    own_cells = official_report["own_leaderboard_cells"]
    author = own_cells[1].split("\n")[0]
    if int(own_cells[0]) != official["observed_rank"] or own_cells[-1] != official["observed_display_time"]:
        raise ValueError("UNKNOWN: official snapshot does not match source")
    audit_path = "evidence/history/representation-audit-v2.md"
    audit = evidence.read(audit_path, "Field map, ограничения валидации и ручные development-кейсы")
    example_id = "3b819bb3-08d9-4a82-a131-244e10f66846"
    example_line = next((s for s in audit.splitlines() if example_id in s), "")
    quotes = re.findall(r"`([^`]+)`", example_line)
    if len(quotes) != 2 or "labels00" not in example_line:
        raise ValueError("UNKNOWN: real opening example not verified in audit")
    experiment_path = "research_archive/experiment.py"
    experiment = evidence.read(experiment_path, "Архив: clean_text и native feature adapter")
    bridge_path = "research_archive/llm_bridge.py"
    evidence.read(bridge_path, "Архив: dialog payload, TRAIN-only retrieval, structured output")
    evidence.read("research_archive/botbench.py", "Архив: группировка и числовые признаки")
    if "def clean_text" not in experiment or "own[:3]" not in experiment:
        raise ValueError("UNKNOWN: archived feature implementation changed")
    proof_path = "evidence/reproduction-proof.json"
    proof = evidence.read(proof_path, "Новый реальный запуск coursework") if (project / proof_path).is_file() else None
    verified, proof_label = proof_state(proof)
    tracking = proof.get("tracking", {}) if isinstance(proof, dict) else {}
    ensemble_path = "evidence/ensemble-proof.json"
    ensemble_proof = evidence.read(ensemble_path, "Полный запуск CatBoost + LLM из исходных данных") if (project / ensemble_path).is_file() else None
    if ensemble_proof is not None and ensemble_proof.get("schema") != "coursework.ensemble-proof.v1":
        raise ValueError("UNKNOWN: unsupported ensemble proof schema")
    ensemble_states = {}
    for mode in ("cached", "fresh"):
        entry = (ensemble_proof or {}).get(mode, {})
        valid = entry.get("status") == "VERIFIED" and entry.get("trained_from_raw") is True
        if mode == "cached":
            valid = valid and entry.get("csv_sha256") == official["artifact_sha256"]
        else:
            valid = valid and entry.get("fresh_llm_completed") is True
        ensemble_states[mode] = ("VERIFIED" if valid else "НЕ ПОДТВЕРЖДЁН", entry)
    for source_path in ("run_ensemble.py", "run_ensemble.sh", "src/pipeline.py", "src/eda.py", "requirements.lock"):
        if (project / source_path).is_file():
            evidence.read(source_path, "Исполняемый код / зависимости: " + source_path)
    built_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    sections = []
    titles = []

    def add(key: str, chapter: str, title: str, body: str, paths: list[str], locator: str,
            dek: str = "", dark: bool = False, contour: str = "not-a-score"):
        i = len(sections) + 1
        heading = "h1" if i == 1 else "h2"
        sections.append(f'<section class="report-section{" dark" if dark else ""}" id="{key}" data-title="{h(title)}" '
                        f'data-contour="{h(contour)}" aria-labelledby="heading-{key}">'
                        f'<header><p class="kicker"><span>YouAreBot / {h(chapter)}</span><span>{i:02d} / 14</span></p>'
                        f'<{heading} id="heading-{key}" tabindex="-1">{h(title)}</{heading}>'
                        f'{f"<p class=dek>{dek}</p>" if dek else ""}</header>'
                        f'<div class="content">{body}</div>{evidence.footer(paths, locator)}</section>')
        titles.append((key, title))

    add("question", "рубежный контроль", "HTML-отчёт: кто в диалоге — бот?",
        f'<div class="two hero-content"><div><p class="lead">Не выбрать «подозрительного».<br>Оценить каждого участника.</p>'
        f'<div class="probabilities"><span>p₀ = ?</span><span>p₁ = ?</span></div>'
        f'<p class="hero-sub">YouAreBot / рубежный контроль<br>Две отдельные вероятности, не softmax.</p>'
        f'<p class="hero-author">{h(author)}<br><span class="small muted">Исследование: 23.10.2026 · HTML: {built_date} UTC</span></p></div>'
        f'<div><p class="caption accent">РЕАЛЬНЫЙ TRAIN-КЕЙС · МЕТКИ 00</p>'
        f'<p class="quote">{h(quotes[0])}</p><p class="quote">{h(quotes[1])}</p>'
        f'<p class="small muted">Оба — люди. Одного приветствия<br>недостаточно для уверенного вывода.</p></div></div>',
        [audit_path, official_path], "Кейс 3b819bb3… · labels00", dark=True)

    add("task", "постановка", "Предсказываем участника. Проверяем группу.",
        '<div class="two equal"><div><table class="schema"><tbody>'
        '<tr><td>text</td><td>Текст реплики, включая пустой.</td></tr>'
        '<tr><td>message</td><td>Порядок, не timestamp.</td></tr>'
        '<tr><td>participant_index</td><td>Локальная сторона 0 / 1, не автор.</td></tr>'
        '<tr><td>is_bot</td><td>TRAIN-метка участника.</td></tr></tbody></table>'
        '<p class="caption muted" style="margin-top:24px">dialog_id — ключ связи и split; не predictor.<br>Нет времени и глобального ID автора.</p></div>'
        '<div class="stack"><div class="definition"><span class="symbol">pᵢ</span><p>Вероятность, что сторона i — бот,<br>по всему диалогу.</p></div>'
        '<div class="definition"><span class="symbol">L</span><div><p class="formula">− mean [ y ln p +<br>(1 − y) ln (1 − p) ]</p>'
        '<p class="caption" style="margin-top:18px">LogLoss оценивает вероятности,<br>а не только правильный класс.</p></div></div>'
        '<p class="takeaway">Две строки участника не должны<br>попасть по разные стороны split.</p></div></div>',
        [audit_path], "§ Field map · § Проверки структуры")

    data_rows = [("Диалоги", "dialogues"), ("Участники / строки", "participant_rows"),
                 ("Сообщения", "messages"), ("Связные компоненты", "components"),
                 ("Уникальные точные диалоги", "unique_exact_dialogues")]
    add("data", "EDA / структура", "Сначала — устройство данных, потом модель.",
        '<div class="two wide-left"><table><thead><tr><th>Единица</th><th class="numeric">TRAIN</th><th class="numeric">TEST</th></tr></thead><tbody>'
        + ''.join(f'<tr><td>{label}</td><td class="numeric">{number(tr[key])}</td><td class="numeric">{number(te[key])}</td></tr>' for label, key in data_rows)
        + '</tbody></table><div class="stack"><p class="emphasis">Порядок реплик<br>≠ чередование.</p>'
        f'<p>В {tr["same_speaker_consecutive_dialogues"]} TRAIN-диалогах<br>есть серии одной стороны.</p>'
        f'<p class="rule">Пустые / пробельные сообщения:<br>TRAIN {tr["empty_or_whitespace_messages"]} · TEST {te["empty_or_whitespace_messages"]}.</p>'
        '<p class="caption muted">Участника берём из поля,<br>не из чётности message.</p></div></div>',
        [facts_path, audit_path], "train / test · структура")

    human = facts["train_class_counts"]["0"]
    bot = facts["train_class_counts"]["1"]
    pairs00 = facts["train_label_pairs"]["00"]
    add("pairs", "EDA / метки", "Два человека — не исключение из задачи.",
        '<div class="two wide-left"><figure>' + pair_chart(facts)
        + '<figcaption>Пара: метка стороны 0, затем стороны 1. 1 = бот.</figcaption></figure>'
        f'<div class="stack"><div><p class="caption">КЛАССЫ УЧАСТНИКОВ · TRAIN</p><div class="class-track">'
        f'<span style="width:{human/tr["participant_rows"]*100:.9f}%"></span><span style="width:{bot/tr["participant_rows"]*100:.9f}%"></span></div>'
        f'<p class="small">Люди {number(human)} / {human/tr["participant_rows"]:.1%}<br>Боты {number(bot)} / {bot/tr["participant_rows"]:.1%}</p></div>'
        f'<p class="emphasis">{pairs00/tr["dialogues"]:.1%} диалогов —<br>два человека.</p>'
        '<p>Не заставляем p₀ + p₁ = 1.</p><p class="caption rule">11 = 0 только в TRAIN.<br>Это наблюдение, не правило TEST.</p></div></div>',
        [facts_path, audit_path], "train_class_counts · train_label_pairs")

    short, long = sequence["slices"]["short_2_5"], sequence["slices"]["long_10_plus"]
    add("length", "EDA / длина", "В коротком диалоге меньше улик.",
        '<div class="two wide-left"><figure><div class="legend"><span><i class="swatch"></i>TRAIN</span><span><i class="swatch test"></i>TEST</span><span>доля диалогов</span></div>'
        + length_chart(facts) + '<figcaption>Числа под столбцами — диалоги; ось Y — доля внутри split.</figcaption></figure>'
        f'<div class="stack"><p>Медиана: {tr["median_messages"]} / {te["median_messages"]} реплик.<br>Максимум: {tr["max_messages"]} / {te["max_messages"]}.</p>'
        '<div class="rule"><p class="caption">ОТДЕЛЬНЫЙ КОНТУР: DEVELOPMENT<br>Native baseline · OOF-срезы</p>'
        f'<div class="stat-line"><span>2–5 реплик</span><strong>{short["base_logloss"]:.6f}</strong></div>'
        f'<div class="stat-line"><span>10+ реплик</span><strong>{long["base_logloss"]:.6f}</strong></div>'
        f'<p class="caption muted" style="margin-top:18px">LogLoss; строки: {short["n_rows"]} / {long["n_rows"]}.<br>Срез — диагностика, не причинный эффект.</p></div></div></div>',
        [facts_path, "evidence/history/sequence-full-report.json"], "histograms · slices.*.base_logloss", contour="EDA + development-slices")

    folds = ''.join(f'<div class="fold" style="--rows:{facts["split_rows"][str(i)]}">F{i}<span>{facts["split_rows"][str(i)]} строк</span></div>' for i in range(5))
    add("validation", "валидация", "Связанные записи остаются в одном fold.",
        '<div class="two wide-left align-top"><div class="stack">' + component_diagram()
        + f'<div><p class="caption" style="margin-bottom:20px">DEVELOPMENT · {number(facts["development_rows"])} СТРОК · 5 FOLDS</p><div class="folds">{folds}</div></div>'
        f'<p class="caption mono">membership v2 · SHA-256 {h(facts["membership_sha256"][:16])}…</p></div>'
        f'<div class="stack"><p class="lead">Confirmation:<br>{facts["confirmation_rows"]} строк отдельно.</p>'
        '<p>Идентификаторы связывают объекты,<br>но не служат предиктором.</p>'
        '<p class="caption rule">UUID-значения в текстовых категориях маскируются. Числовые агрегаты считаются отдельно.</p>'
        '<p class="caption muted">Это не chronological / author-disjoint split. Часть confirmation-кейсов видели при аудите; строгая analyst-blindness не заявляется.</p></div></div>',
        [facts_path, audit_path], "split_rows · membership v2 · ограничения")

    cfg = reports["template-v1"]["config"]
    ecfg = encoder["config"]
    params = [("iterations", cfg["iterations"]), ("depth", cfg["depth"]), ("learning_rate", cfg["learning_rate"]),
              ("l2_leaf_reg", cfg["l2_leaf_reg"]), ("random_seed", cfg["random_seed"]), ("thread_count", cfg["thread_count"])]
    add("features", "native baseline", "Сильная база — не только стиль текста.",
        '<div class="two equal"><div class="stack"><p class="lead">'
        f'{ecfg["numeric_columns"]} числовых + {ecfg["category_columns"]} категориальных.</p>'
        '<table class="compact"><tbody><tr><td>Свой / чужой / разность</td><td>Длины, состав, серии.</td></tr>'
        '<tr><td>Ответное поведение</td><td>Эхо, направление, пробелы.</td></tr>'
        '<tr><td>Первые три фразы</td><td>Категории по сторонам.</td></tr></tbody></table>'
        '<p class="caption muted">Raw UUID не номинальный признак. clean_text маскирует UUID в текстах / категориях; это не универсальная маска числового extractor.</p></div>'
        '<div class="stack"><h3>CatBoost · фиксированный recipe</h3><dl class="recipe">'
        + ''.join(f'<div><dt>{h(k)}</dt><dd>{h(v)}</dd></div>' for k, v in params)
        + '</dl><p class="takeaway">Без HPO и early stopping.</p>'
        '<p class="caption">Числа, текст и шаблоны проверяются<br>на одной frozen membership.</p></div></div>',
        ["evidence/history/template-v1-report.json", experiment_path], "config · feature_matrix · clean_text")

    best = min(models, key=lambda m: m["logloss"])
    add("models", "одинаковый контур", "Шаблоны дали лучший native-результат.",
        model_chart(models)
        + '<p class="caption">Цветом выделен выбранный native baseline. H7 / H8 не улучшили полный пятифолдовый результат.</p>',
        [facts_path], "6 report SHA-256 · полный список в «Источниках»",
        dek=f'Development · {number(facts["development_rows"])} одних и тех же строк · 5 folds · LogLoss ↓ · ось от нуля',
        contour="development5fold")

    ci = conf["group_bootstrap_95pct_interval"]
    add("llm", "отдельная ветка / confirmation", "LLM добавила сигнал — в проверенном контуре.",
        '<div class="flowline"><span>Диалог + native prior<small>Текст — данные, не команды.</small></span><b>→</b>'
        '<span>Ближайшие TRAIN-примеры<small>Только training-группы, не eval labels.</small></span><b>→</b>'
        '<span>Структурированные p₀, p₁<small>Фиксированная смесь 50 / 50.</small></span></div>'
        '<div class="two wide-left"><figure>'
        + two_bars("confirmation-bars", [("Native", conf["base_logloss"]), ("50 / 50 + LLM", conf["candidate_logloss"])], 0.20, True)
        + f'<figcaption>Confirmation · {conf["n_rows"]} строк / {conf["n_components"]} компоненты · не CV и не public.</figcaption></figure>'
        '<div class="stack"><p class="caption accent">Δ = СМЕСЬ − NATIVE</p>'
        f'<p class="emphasis">{conf["delta_candidate_minus_base"]:+.6f}</p><p class="mono">95% CI<br>[{ci[0]:+.6f}; {ci[1]:+.6f}]</p>'
        '<p class="caption">Group bootstrap по компонентам.</p></div></div>'
        '<p class="caption rule">Внешний black box недетерминирован. Свежая LLM-генерация не входит в default-воспроизведение; это историческая ветка, а не новая API-проверка.</p>',
        [conf_path, bridge_path], "against_template · recipe 0.5 + 0.5", dark=True, contour="confirmation")

    h7 = next(m for m in models if m["id"] == "template-text-v2")
    h8 = next(m for m in models if m["id"] == "sequence-full-v2")
    em = encoder["metrics"]
    eci = encoder["comparisons"]["native"]["group_bootstrap_95pct_interval"]
    add("negative-results", "отрицательные результаты", "Сложнее — не значит лучше.",
        '<div class="two equal align-top"><div class="stack"><p class="pill">ПОЛНЫЙ CV · 5 FOLDS</p>'
        '<table class="compact"><thead><tr><th>Вариант</th><th class="numeric">LogLoss</th><th class="numeric">Δ к native</th></tr></thead><tbody>'
        f'<tr><td>Native</td><td class="numeric">{best["logloss"]:.6f}</td><td class="numeric">—</td></tr>'
        f'<tr><td>H7 · native text</td><td class="numeric">{h7["logloss"]:.6f}</td><td class="numeric negative">{h7["logloss"]-best["logloss"]:+.6f}</td></tr>'
        f'<tr><td>H8 · sequence mix</td><td class="numeric">{h8["logloss"]:.6f}</td><td class="numeric negative">{h8["logloss"]-best["logloss"]:+.6f}</td></tr>'
        '</tbody></table><p class="caption">Не продвигаем по удачному одному fold,<br>если полный CV не подтвердил выигрыш.</p></div>'
        '<div class="stack tight"><p class="pill">H9 · ТОЛЬКО FOLD 0</p>'
        + two_bars("encoder-bars", [("Native", em["native"]["logloss"]), ("Encoder mix", em["primary"]["logloss"]), ("Raw encoder", em["raw_encoder"]["logloss"])], 0.25)
        + f'<p class="caption">{encoder["comparisons"]["native"]["n_rows"]} строк · Δ 95% CI [{eci[0]:+.6f}; {eci[1]:+.6f}]<br>Интервал пересекает 0; другие folds не допущены.</p></div></div>'
        '<p class="takeaway">Закрыт этот recipe, не семейство методов. Ресурсный сбой — не положительный результат качества.</p>',
        [facts_path, encoder_path], "development5fold ≠ pilot_fold0", contour="development5fold + singlefold-negative")

    command = "./project/run_ensemble.sh --data-dir /path/to/contest --llm-mode cached"
    status_html = f'<span class="{"verified" if verified else "pending"}">{h(proof_label)}</span>'
    if verified:
        proof_detail = (f'<p class="proof-value">run_id: {h(proof["run_id"])}</p>'
                        f'<p>OOF {number(proof["oof_rows"])} · LogLoss {number(proof["oof_logloss"],9)}<br>TEST CSV: {number(proof["test_rows"])} строк.</p>'
                        f'<p class="caption">{number(proof.get("seconds"),3)} с · 6 checkpoints: exact replay.<br>AUC {number(proof.get("oof_auc"),9)}.</p>')
    else:
        proof_detail = '<p>Нет подтверждённого end-to-end результата.<br>Ожидается reproduction-proof.json.</p>'
    proof_sources = ([ensemble_path] if ensemble_proof is not None else []) + ([proof_path] if proof is not None else [facts_path])
    mode_table = '<table><thead><tr><th>LLM-режим</th><th>Что выполняется</th><th>Проверка</th></tr></thead><tbody>'
    for mode in ("cached", "fresh"):
        state, entry = ensemble_states[mode]
        description = ('Обработка и обучение с нуля + сохранённые ответы LLM, привязанные к тем же входам и prior. Точный повтор CSV.' if mode == "cached" else 'Обработка и обучение с нуля + новые запросы к LLM. Тот же рецепт, но не обещание тех же ответов / public score.')
        mode_table += f'<tr><td class="mono">--llm-mode {mode}</td><td>{h(description)}</td><td>{h(state)}<br>{h(entry.get("run_id", ""))}</td></tr>'
    mode_table += '</tbody></table>'
    add("reproduction", "воспроизводимость", "Весь ансамбль — одним запуском.",
        f'<code class="command">{h(command)}</code>'
        '<p class="caption">Свежий LLM-этап: замените флаг на <code>--llm-mode fresh</code>. Режим выбирается явно; скрытого перехода к кэшу нет.</p>'
        '<div class="flowline"><span>Входы<small>validate / EDA</small></span><b>→</b><span>Group CV<small>OOF · 5 folds</small></span><b>→</b>'
        f'<span>Full refit<small>{tr["participant_rows"]} TRAIN-меток</small></span><b>→</b><span>LLM<small>TRAIN-only retrieval</small></span><b>→</b>'
        f'<span>50 / 50<small>{te["participant_rows"]} TEST-строк + tracker</small></span></div>'
        + mode_table +
        '<p>Обработка данных и обучение · Python: <code>run_ensemble.py</code> → <code>src/eda.py</code>, <code>src/pipeline.py</code>, <code>src/experiment.py</code>.</p>'
        '<p class="caption">Python-зависимости зафиксированы в <code>requirements.lock</code>. Для fresh дополнительно нужен авторизованный доступ к внешней предобученной языковой модели. Точные требования сохранены в <code>llm/config.json</code> и <code>llm/adapter.py</code>. Установка зависимостей и получение официальных данных — явные предварительные шаги.</p>'
        '<h3>Отдельно: проверка CatBoost-компонента</h3>'
        f'<div class="proof"><div class="stack tight">{status_html}{proof_detail}</div>'
        '<div class="stack tight"><p>Код — внутри project/.<br>Зависимости — exact pinned.</p>'
        '<p class="caption">Сырые данные и веса не в Git. Путь к официальным входам задаётся явно.</p>'
        '<p class="caption muted">Эти OOF-метрики относятся к CatBoost, не к ансамблю. Public score локально не пересчитывается. Простой replay <code>ensemble/run.py</code> не заменяет этот эксперимент с обучением.</p></div></div>',
        proof_sources, "Raw-data ensemble / cached и fresh проверяются раздельно", contour="ensemble-reproduction")

    def tracked(key: str) -> str:
        return h(tracking.get(key, "UNKNOWN")) if verified else "PENDING"

    utc_time = (proof.get("finished_at_utc") or proof.get("finished_at") or tracking.get("finished_at_utc") or proof.get("verified_at_utc") or "UNKNOWN") if verified else "PENDING"
    tracking_count_text = (f'{number(tracking["logged_parameters"])} групп записей параметров · {number(tracking["logged_metrics"])} метрик · {number(tracking["logged_artifacts"])} артефактов'
                           if verified else "Количество записей: PENDING")
    ensemble_tracking_table = '<table><thead><tr><th>Ансамбль</th><th>SQL-статус</th><th>Группы параметров / метрики / артефакты</th></tr></thead><tbody>'
    for mode in ("cached", "fresh"):
        state, entry = ensemble_states[mode]
        recorded = entry.get("tracking", {})
        counts = ' / '.join(number(recorded.get(k)) for k in ("parameter_groups", "metrics", "artifacts"))
        ensemble_tracking_table += f'<tr><td>{h(mode)}<br><span class="mono">{h(entry.get("run_id", ""))}</span></td><td>{h(recorded.get("status", state))}</td><td>{h(counts)}</td></tr>'
    ensemble_tracking_table += '</tbody></table>'
    add("tracking", "трекер экспериментов", "SQLite + JSON. Не макет MLflow.",
        '<p>Полный ансамбль фиксирует режим LLM, модель, prompt / retrieval / source hashes, веса смеси, параметры обучения, зависимости, метрики и артефакты в собственном запуске. Подтверждение cached / fresh — в разделе выше.</p>'
        + ensemble_tracking_table +
        '<p class="caption">Фиксируются все параметры, доступные в нашем коде и CLI. Скрытые server-side настройки и ревизия весов внешней LLM не доступны; побайтовая повторяемость fresh не обещается.</p>'
        '<div class="two equal align-top"><div class="stack"><div class="tracking-record"><p class="caption">ПРОВЕРЕННЫЙ CATBOOST-КОМПОНЕНТ</p><dl>'
        f'<dt>run_id</dt><dd>{h(proof["run_id"]) if verified else "PENDING"}</dd>'
        f'<dt>status</dt><dd>{tracked("run_status")}</dd><dt>Проверено UTC</dt><dd>{h(utc_time)}</dd>'
        f'<dt>backend</dt><dd>{tracked("backend")}</dd><dt>database</dt><dd>{tracked("database")}</dd></dl></div><p class="caption">{tracking_count_text}</p><p class="caption">Включая 6 полных словарей effective model params; это не число scalar-параметров.</p>'
        f'{status_html}<p class="caption muted">Это собственный SQLite + JSON tracker.<br>MLflow не установлен и не изображается.</p></div>'
        '<ol class="tracking-list"><li>Конфиг + все эффективные model defaults.</li><li>Метрики каждого fold и общий OOF.</li>'
        '<li>SHA-256: код, данные, конфиг, окружение.</li><li>Индекс артефактов + checkpoints.</li>'
        '<li>run_id, UTC, статус FINISHED / FAILED.</li></ol></div>'
        '<p class="caption rule">Проверка — чтение реальной SQLite / JSON и replay модели. Исторические эксперименты остаются архивом, не «запусками задним числом».</p>',
        proof_sources, "Tracking нового run · факты только из proof", contour="new-native-reproduction")

    add("service", "архитектура / эскиз", "Offline pipeline — база. HTTP — следующий слой.",
        service_diagram(te["participant_rows"]),
        [experiment_path, "research_archive/botbench.py"], "Сплошное: offline · пунктир: NOT IMPLEMENTED",
        dek="Эскиз сервиса: POST /predict → валидация → признаки → versioned CPU model → p₀, p₁ + версия.")

    add("conclusion", "итог / критерии", "Результат есть. Границы результата — тоже.",
        '<div class="two equal align-top"><div class="stack tight"><p class="caption">СООТВЕТСТВИЕ КРИТЕРИЯМ, НЕ ОБЕЩАНИЕ ОЦЕНКИ</p>'
        '<table class="rubric"><tbody><tr><td>EDA · структура, классы, длины</td><td>2</td></tr>'
        '<tr><td>Идея / модели / эскиз сервиса</td><td>2</td></tr><tr><td>Зависимости / locked recipe</td><td>1</td></tr>'
        '<tr><td>Трекинг параметров и артефактов</td><td>3</td></tr>'
        '<tr><td>Обработка данных и обучение · Python</td><td>2</td></tr>'
        '<tr><td>Код в project/ · end-to-end одним скриптом</td><td>—</td></tr></tbody></table>'
        '<p class="caption">Основной entry point: <code>run_ensemble.py</code>; один shell-запуск — <code>run_ensemble.sh</code>.</p>'
        f'<p class="caption" style="margin-top:14px">Native reproduction: {"VERIFIED" if verified else "PENDING"}.</p>'
        f'<a class="repo-link" href="{REPO}" target="_blank" rel="noopener noreferrer">FlagonTTM/you-are-bot ↗</a>'
        f'<a class="repo-link" href="{REPO}/tree/{BRANCH}/project" target="_blank" rel="noopener noreferrer">feature/coursework-html / project ↗</a>'
        '<p class="caption muted">Приватный target-репозиторий.<br>Публикация / доступ проверяются отдельно.</p></div>'
        '<div class="stack"><p class="caption">ИСТОРИЧЕСКИЙ ОФИЦИАЛЬНЫЙ PUBLIC LOGLOSS</p>'
        f'<p class="public-score">{official["public_logloss"]:.10f}</p>'
        f'<p>Место {official["observed_rank"]} на снимке<br>{h(official["observed_display_time"])}.</p>'
        '<p class="caption muted">Не live-рейтинг. Не private-score.<br>Первое место не достигнуто.</p>'
        '<p class="outcome rule">Продолжать оптимизацию —<br>не цель этой работы.<br>Цель — доказуемый проект.</p></div></div>',
        [official_path, facts_path], "Submission #10588 · snapshot, не live", contour="official-public-snapshot")

    if tuple(key for key, _ in titles) != SECTION_IDS:
        raise AssertionError("section order/count mismatch")
    evidence.sources["tools/build_html.py"] = {"path": "tools/build_html.py", "sha256": sha(Path(__file__)), "label": "Генератор HTML, Python stdlib"}
    ledger = list(evidence.sources.values())
    source_list = []
    for source in ledger:
        local = quote(source["path"], safe="/-.")
        remote = f'{REPO}/blob/{BRANCH}/project/{local}'
        source_list.append(f'<li><strong>{h(source["label"])}</strong><br><a href="{local}">{h("project/"+source["path"])}</a>'
                           f' · <a href="{remote}" target="_blank" rel="noopener noreferrer">GitHub · private</a>'
                           f'<code class="digest">SHA-256 {h(source["sha256"])}</code></li>')
    source_list.append('<li><strong>Официальные входы: не включены в HTML / Git</strong>'
                       + ''.join(f'<p>{h(k)}<code class="digest">SHA-256 {h(v)}</code></p>' for k, v in facts["data_hashes"].items())
                       + f'<p>membership v2<code class="digest">SHA-256 {h(facts["membership_sha256"])}</code></p></li>')
    payload = {"schema": SCHEMA, "facts": facts, "sources": ledger, "build_date_utc": built_date,
               "development_slices": {"short_2_5": short, "long_10_plus": long},
               "reproduction_proof": proof, "ensemble_proof": ensemble_proof, "proof_verified": verified, "browser_qa": "PENDING_PARENT_OPENCLI"}
    toc_labels = ["Задача и реальный пример", "Постановка и метрика", "Структура данных", "Классы и пары", "Длины диалогов", "Групповая валидация", "Признаки и native recipe", "Шесть моделей / один CV", "Историческая LLM-ветка", "Отрицательные результаты", "Реальный standalone-запуск", "SQLite + JSON tracking", "Эскиз HTTP-сервиса", "Итоги и критерии"]
    source_body = ('<p style="margin-top:24px">Все числа загружены из локальных evidence. HTML работает без сети; ссылки открываются только по действию читателя. Для private GitHub нужен доступ.</p>'
                   '<p style="margin-top:16px">Development 5-fold, confirmation, pilot fold 0 и official public — разные оценки. Их нельзя ранжировать одной шкалой «прогресса».</p>'
                   '<ul class="source-list">' + ''.join(source_list) + '</ul>')
    document = ('<!doctype html>\n<html lang="ru"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<meta name="color-scheme" content="light">'
                '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\'; style-src \'unsafe-inline\'; img-src data:; connect-src \'none\'; font-src \'none\'; base-uri \'none\'; form-action \'none\'">'
                '<meta name="description" content="YouAreBot: самостоятельный HTML-отчёт рубежного контроля, 14 разделов, реальные данные, диаграммы и источники. Работает offline.">'
                '<title>HTML-отчёт — YouAreBot / рубежный контроль</title><style>' + CSS + '</style></head><body>'
                '<a href="#question" class="skip">К содержимому отчёта</a><div class="report-layout">'
                '<nav class="toc" id="contents" aria-label="Оглавление HTML-отчёта"><p class="toc-title">YOU ARE BOT<br>HTML-отчёт</p><ol>'
                + ''.join(f'<li><a href="#{key}" data-section="{key}">{h(label)}</a></li>' for (key, _), label in zip(titles, toc_labels))
                + '</ol><button id="open-sources" aria-haspopup="dialog" aria-controls="sources-dialog" hidden>Источники / SHA-256</button>'
                '<p class="toc-note">14 разделов · непрерывное чтение<br>Без сети и внешних библиотек.</p></nav>'
                '<main class="report-main" aria-label="HTML-отчёт YouAreBot">' + ''.join(sections)
                + '<details class="sources-fallback" id="source-ledger"><summary>Все источники и SHA-256 — в тексте страницы</summary>'
                + source_body + '</details><a class="backtop" href="#contents">↑ К оглавлению</a></main></div>'
                '<dialog id="sources-dialog" aria-labelledby="sources-heading"><h2 id="sources-heading">Источники и контуры</h2>'
                '<button class="close" id="close-sources" aria-label="Закрыть источники">×</button>'
                + source_body + '</dialog><script id="source-data" type="application/json">' + safe_json(payload)
                + '</script><script>' + JS + '</script></body></html>\n')
    return document, payload


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.sections = []
        self.refs = []
        self.assets = []
        self.svgs = 0
        self.svg_titles = 0
        self.in_json = False
        self.payload = []
        self.in_js = False
        self.js = []
        self.model_values = {}
        self.histogram = {}
        self.pairs = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if tag == "section" and "report-section" in a.get("class", "").split():
            self.sections.append(a["id"])
            assert "hidden" not in a and "inert" not in a and a.get("aria-hidden") != "true", "report section must be visible"
        if tag == "svg":
            self.svgs += 1
            assert a.get("role") == "img" and "aria-labelledby" in a, "SVG requires accessible name"
        if tag == "title":
            self.svg_titles += 1
        if "href" in a and a["href"].startswith("#"):
            self.refs.append(a["href"][1:])
        if tag in ("script", "link", "img", "iframe", "object", "embed", "audio", "video", "source", "use"):
            for key in ("src", "href", "data", "xlink:href"):
                if key in a:
                    self.assets.append(a[key])
        if "data-model" in a:
            self.model_values[a["data-model"]] = float(a["data-value"])
        if "data-series" in a:
            self.histogram[(a["data-series"], int(a["data-count"]))] = float(a["data-value"])
        if "data-pair" in a:
            self.pairs[a["data-pair"]] = int(a["data-count"])
        if tag == "script":
            self.in_json = a.get("id") == "source-data"
            self.in_js = not self.in_json

    def handle_data(self, data):
        if self.in_json:
            self.payload.append(data)
        if self.in_js:
            self.js.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_json = self.in_js = False


def validate_document(document: str, facts: dict | None = None) -> dict:
    parser = ReportParser()
    parser.feed(document)
    assert tuple(parser.sections) == SECTION_IDS, "exactly 14 stable section IDs required"
    assert len(parser.ids) == len(set(parser.ids)), "duplicate IDs"
    assert '<title>HTML-отчёт' in document and '<h1 id="heading-question" tabindex="-1">HTML-отчёт' in document
    assert document.count('data-section=') == len(SECTION_IDS), "anchor TOC must cover all sections"
    assert not re.search(r"addEventListener\([\'\"]keydown|preventDefault\(", JS), "keep native report scrolling"
    assert set(parser.refs).issubset(parser.ids), "broken internal fragment"
    assert parser.assets == [], f"external/file assets forbidden: {parser.assets}"
    assert parser.svgs >= 7, "real SVG charts / diagrams required"
    payload = json.loads(''.join(parser.payload))
    facts = facts or payload["facts"]
    for model in facts["models"]:
        assert parser.model_values[model["id"]] == model["logloss"], "chart/source metric mismatch"
    for series in ("train", "test"):
        for count in facts[series]["length_histogram"].values():
            assert parser.histogram[(series, count)] == count / facts[series]["dialogues"]
    assert sum(parser.pairs.values()) == facts["train"]["dialogues"]
    assert parser.pairs["00"] == facts["train_label_pairs"]["00"]
    js = ''.join(parser.js)
    assert not re.search(r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|eval)\s*\(", js)
    assert not re.search(r"@import|@font-face|url\s*\(", CSS, re.I), "external CSS asset"
    assert "requestAnimationFrame" not in js, "initialization must not depend on frames"
    for key in ("hashchange", "showModal", "IntersectionObserver"):
        assert key in js, f"missing interaction: {key}"
    assert "prefers-reduced-motion" in document and "@media print" in document
    if payload["reproduction_proof"] is None:
        assert not payload["proof_verified"] and "PENDING · запуск ещё не подтверждён" in document
    assert "UNKNOWN" in safe_json({"UNKNOWN": "</script><script>alert('x')</script>"})
    assert "</script" not in safe_json({"x": "</script>"})
    return {"status": "PASS", "sections": len(parser.sections), "svg_count": parser.svgs,
            "unique_ids": len(parser.ids), "external_assets": len(parser.assets),
            "matched_model_metrics": len(parser.model_values), "histogram_bars": len(parser.histogram),
            "pair_counts_reconciled": sum(parser.pairs.values()),
            "source_files": len(payload["sources"]), "proof_verified": payload["proof_verified"],
            "browser_layout_qa": "PENDING_PARENT_OPENCLI"}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check", action="store_true", help="build and run stdlib static/content checks")
    cli.add_argument("--check-only", action="store_true", help="check existing HTML without rebuilding")
    args = cli.parse_args()
    project = Path(__file__).resolve().parents[1]
    target = project / "report.html"
    if args.check_only:
        document = target.read_text(encoding="utf-8")
    else:
        document, _ = build_document(project)
        # Validate before replacing the deliverable; no incomplete artifact is emitted.
        validate_document(document)
        target.write_text(document, encoding="utf-8")
    result = validate_document(document) if args.check or args.check_only else {"status": "BUILT", "sections": 14}
    result.update({"html": str(target), "html_sha256": sha(target), "builder_sha256": sha(Path(__file__)), "bytes": target.stat().st_size})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
