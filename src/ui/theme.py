"""Shared look-and-feel for every page of the Streamlit app."""

from html import escape

import streamlit as st

PAPER = "#4f46e5"
MATERIAL = "#e11d48"
DEVICE = "#059669"
EXPERIMENT = "#d97706"
CHART_COLORS = ["#4f46e5", "#0891b2", "#d97706", "#059669", "#e11d48", "#7c3aed"]

_CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; }

.ms-hero {
  position: relative; overflow: hidden;
  border-radius: 16px; padding: 26px 30px 24px;
  background: radial-gradient(circle at 88% 30%, rgba(99,102,241,.35), transparent 45%),
              linear-gradient(135deg, #0b1020 0%, #1e1b4b 58%, #083344 100%);
  color: #f8fafc; margin-bottom: 1.4rem;
  box-shadow: 0 10px 30px -12px rgba(30,27,75,.45);
}
.ms-hero::after {
  content: ""; position: absolute; inset: 0;
  background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 28px 28px; pointer-events: none;
}
.ms-eyebrow { font-size: .74rem; font-weight: 600; letter-spacing: .16em;
  text-transform: uppercase; color: #22d3ee; margin: 0 0 6px; }
.ms-title { font-size: 2rem; font-weight: 750; letter-spacing: -.02em;
  margin: 0; line-height: 1.15; }
.ms-sub { color: #cbd5e1; font-size: .98rem; margin: 8px 0 0; max-width: 46rem; }
.ms-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;
  position: relative; z-index: 1; }
.ms-chip { font-size: .78rem; font-weight: 500; color: #e0e7ff;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
  border-radius: 999px; padding: 4px 11px; }
.ms-chip b { color: #fff; font-weight: 650; }

.ms-section { display: flex; align-items: baseline; gap: 10px; margin: 1.6rem 0 .2rem; }
.ms-section h3 { margin: 0; padding: 0; font-size: 1.2rem; font-weight: 650; }
.ms-section .ms-num { font-size: .75rem; font-weight: 700; color: #4f46e5;
  background: #eef2ff; border-radius: 6px; padding: 2px 7px; }

.ms-source { border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 14px;
  margin-bottom: 10px; background: #ffffff; }
.ms-source-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.ms-rank { flex: none; width: 22px; height: 22px; border-radius: 6px; background: #eef2ff;
  color: #4f46e5; font-size: .75rem; font-weight: 700; display: grid; place-items: center; }
.ms-source-title { font-weight: 600; font-size: .92rem; color: #0f172a; line-height: 1.3; }
.ms-source-title a { color: inherit; text-decoration: none; }
.ms-source-title a:hover { color: #4f46e5; text-decoration: underline; }
.ms-tag { flex: none; margin-left: auto; font-size: .68rem; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; color: #0e7490; background: #ecfeff; border: 1px solid #a5f3fc;
  border-radius: 999px; padding: 1px 8px; }
.ms-source-text { color: #475569; font-size: .85rem; line-height: 1.5; margin: 0; }

.ms-legend { display: flex; flex-direction: column; gap: 6px; font-size: .9rem; }
.ms-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 8px; vertical-align: 0; }

.ms-empty { border: 1px dashed #c7d2fe; border-radius: 14px; padding: 20px 22px;
  background: #fafaff; margin: .4rem 0 1rem; }
.ms-empty h4 { margin: 0 0 4px; font-size: 1.02rem; font-weight: 650; }
.ms-empty p { margin: 0; color: #64748b; font-size: .9rem; }
</style>
"""


def apply_theme() -> None:
    st.html(_CSS)


def hero(eyebrow: str, title: str, subtitle: str, chips: list[tuple[str, str]] = ()) -> None:
    chip_html = "".join(
        f'<span class="ms-chip"><b>{escape(v)}</b> {escape(k)}</span>' for k, v in chips
    )
    st.html(
        f'<div class="ms-hero"><p class="ms-eyebrow">{escape(eyebrow)}</p>'
        f'<p class="ms-title">{escape(title)}</p><p class="ms-sub">{escape(subtitle)}</p>'
        f'<div class="ms-chips">{chip_html}</div></div>'
    )


def section(number: str, title: str, caption: str | None = None) -> None:
    st.html(
        f'<div class="ms-section"><span class="ms-num">{escape(number)}</span>'
        f"<h3>{escape(title)}</h3></div>"
    )
    if caption:
        st.caption(caption)


def source_card(
    rank: int, title: str, paper_id: str, section_name: str | None, text: str, max_chars: int = 420
) -> None:
    snippet = text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"
    link = paper_id if paper_id.startswith("http") else ""
    title_html = (
        f'<a href="{escape(link)}" target="_blank">{escape(title)}</a>' if link else escape(title)
    )
    tag = ""
    if section_name:
        tag = f'<span class="ms-tag">{escape(section_name.replace("_", " "))}</span>'
    st.html(
        f'<div class="ms-source"><div class="ms-source-head"><span class="ms-rank">{rank}</span>'
        f'<span class="ms-source-title">{title_html}</span>{tag}</div>'
        f'<p class="ms-source-text">{escape(snippet)}</p></div>'
    )


def legend(items: list[tuple[str, str]]) -> None:
    rows = "".join(
        f'<div><span class="ms-dot" style="background:{color}"></span>{escape(label)}</div>'
        for label, color in items
    )
    st.html(f'<div class="ms-legend">{rows}</div>')


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="ms-empty"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>')
