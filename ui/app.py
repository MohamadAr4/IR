"""Streamlit UI for the IR system (requirement #9).

Covers every UI requirement:
  * pick the dataset before querying,
  * choose the representation model, incl. picking the hybrid type
    (parallel / serial) and its fusion/model options from the UI,
  * switch between the *basic* pipeline and the *additional features*
    (query refinement) with a single toggle,
  * control the probabilistic model's parameters (BM25 k1 / b) live,
  * view ranked results with score bars, query-term highlighting and the
    per-model component scores; plus the spelling corrections / expansions /
    suggestions produced by refinement,
  * an Evaluation tab that runs MAP / Recall / P@10 / nDCG@10 and the
    before/after-refinement comparison.

The UI always talks to the system through the SOA **API Gateway** over REST,
which routes to the five microservices — service orientation is the system's
architecture, not a user-facing option. Start the backend first:

    python -m services.run_all      # services + gateway
    streamlit run ui/app.py         # this UI
"""
from __future__ import annotations

import html
import os
import re
import sys
import time

import streamlit as st

# make the project importable when Streamlit runs this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_core.config import SERVICE_URLS  # noqa: E402

st.set_page_config(page_title="IR System", page_icon="🔎", layout="wide")

MODEL_LABELS = {
    "tfidf": "VSM · TF-IDF (cosine)",
    "bm25": "BM25 (probabilistic)",
    "bert": "Embedding · BERT",
    "word2vec": "Embedding · Word2Vec",
    "multilingual": "Embedding · Multilingual 🌐 (cross-lingual)",
    "hybrid_parallel": "Hybrid · Parallel (fusion)",
    "hybrid_serial": "Hybrid · Serial (rerank)",
}
MODEL_HELP = {
    "tfidf": "Sparse vector space model with TF-IDF weights, ranked by cosine similarity.",
    "bm25": "Probabilistic Okapi BM25 with tunable k1 (term-freq saturation) and b (length norm).",
    "bert": "Dense semantic search with BERT (all-MiniLM-L6-v2) sentence embeddings.",
    "word2vec": "Dense search with mean-pooled Word2Vec (skip-gram) word vectors.",
    "multilingual": "Cross-lingual search: ask in any language (Arabic, French, Spanish, …) "
                    "and retrieve the English arguments — one shared multilingual embedding space.",
    "hybrid_parallel": "Runs several models independently and fuses their rankings (weighted sum or RRF).",
    "hybrid_serial": "Two stages: a fast model retrieves candidates, a stronger model re-ranks them.",
}
# Curated example queries per dataset (clickable).
EXAMPLES = {
    "argsme": ["should teachers get tenure", "should students wear school uniforms",
               "is vaping safer than smoking", "should the death penalty be abolished",
               "do violent video games cause violence"],
    "argsme_sample": ["school condoms for students", "teen pregnancy", "contraceptive forms"],
}

CSS = """
<style>
:root{
  --bg:#0f172a; --bg2:#0b1222; --surface:#1e293b; --line:#33425b;
  --text:#e2e8f0; --muted:#94a3b8; --cyan:#22d3ee; --teal:#2dd4bf;
}
/* ---- app shell ---- */
.stApp{ background: radial-gradient(1100px 560px at 82% -12%, #12233f 0%, var(--bg) 55%) fixed; }
[data-testid="stHeader"]{ background: transparent; }
.block-container{ padding-top: 1.4rem; max-width: 1180px; }

/* ---- hero header ---- */
.hero{ margin: 2px 0 10px; }
.hero h1{ font-size:2.05rem; font-weight:800; margin:0; letter-spacing:-.02em; line-height:1.1;
  background:linear-gradient(90deg,var(--cyan),var(--teal));
  -webkit-background-clip:text; background-clip:text; color:transparent; display:inline-block; }
.hero .bar{ height:4px; width:168px; border-radius:4px; margin:9px 0 6px;
  background:linear-gradient(90deg,var(--cyan),var(--teal)); box-shadow:0 0 18px rgba(34,211,238,.6); }
.hero .sub{ color:var(--muted); font-size:.95rem; } .hero .sub b{ color:var(--text); }

/* ---- sidebar ---- */
[data-testid="stSidebar"]{ background:#0b1222; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1{ font-size:1.25rem; color:var(--text); }

/* ---- result cards ---- */
.result-card{ border:1px solid var(--line); border-radius:14px; padding:15px 18px; margin-bottom:14px;
  background:linear-gradient(180deg,var(--surface),var(--bg2)); box-shadow:0 2px 10px rgba(0,0,0,.28);
  transition:transform .12s ease, box-shadow .12s ease, border-color .12s ease; }
.result-card:hover{ transform:translateY(-2px); border-color:var(--cyan); box-shadow:0 8px 26px rgba(34,211,238,.16); }
.rank-badge{ display:inline-block; min-width:32px; padding:3px 10px; border-radius:8px;
  background:linear-gradient(135deg,var(--cyan),var(--teal)); color:#042027; font-weight:800;
  text-align:center; font-size:.82rem; box-shadow:0 0 12px rgba(34,211,238,.45); }
.docid{ color:var(--muted); font-family:ui-monospace,monospace; font-size:.76rem; margin-left:8px; }
.score-val{ float:right; font-weight:800; color:var(--cyan); text-shadow:0 0 14px rgba(34,211,238,.55); }
.score-bar-bg{ background:rgba(148,163,184,.16); border-radius:6px; height:8px; margin:9px 0 2px; overflow:hidden; }
.score-bar{ background:linear-gradient(90deg,var(--cyan),var(--teal)); height:8px; border-radius:6px;
  box-shadow:0 0 12px rgba(45,212,191,.6); }
.comp-badge{ display:inline-block; background:rgba(34,211,238,.12); color:var(--cyan);
  border:1px solid rgba(34,211,238,.28); border-radius:7px; padding:2px 9px; margin:8px 6px 0 0;
  font-size:.72rem; font-family:ui-monospace,monospace; }
.snippet{ margin-top:10px; line-height:1.6; font-size:.93rem; color:#cbd5e1; }
mark{ background:rgba(34,211,238,.28); color:#e7fbff; padding:0 3px; border-radius:4px; }
.pill{ display:inline-block; background:rgba(45,212,191,.14); color:var(--teal);
  border:1px solid rgba(45,212,191,.32); border-radius:14px; padding:2px 11px; margin:2px 4px 2px 0; font-size:.78rem; }

/* ---- buttons ---- */
.stButton>button{ border-radius:10px; border:1px solid var(--line); background:var(--surface);
  color:var(--text); font-weight:600; transition:all .12s ease; }
.stButton>button:hover{ border-color:var(--cyan); color:var(--cyan); box-shadow:0 0 0 1px rgba(34,211,238,.3); }
.stButton>button[kind="primary"], .stButton>button[data-testid="baseButton-primary"],
.stButton>button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--cyan),var(--teal)); color:#042027; border:none;
  box-shadow:0 4px 16px rgba(34,211,238,.35); }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ background:transparent; color:var(--muted); border-radius:9px 9px 0 0;
  padding:8px 18px; font-weight:600; }
.stTabs [aria-selected="true"]{ color:var(--cyan)!important; border-bottom:2px solid var(--cyan); }

/* ---- metric cards ---- */
[data-testid="stMetric"]{ background:linear-gradient(180deg,var(--surface),var(--bg2));
  border:1px solid var(--line); border-radius:12px; padding:12px 16px; }
[data-testid="stMetricValue"]{ color:var(--cyan); font-weight:800; }
[data-testid="stMetricLabel"]{ color:var(--muted); }

/* ---- inputs & expanders ---- */
[data-baseweb="input"], [data-baseweb="select"]>div{ border-radius:10px!important; }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:12px; background:var(--bg2); overflow:hidden; }
[data-testid="stExpander"] summary:hover{ color:var(--cyan); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend abstraction: in-process engine OR remote gateway
# ---------------------------------------------------------------------------
class Backend:
    def datasets(self): ...
    def models(self, ds): ...
    def status(self, ds): ...
    def search(self, **kw): ...
    def refine(self, **kw): ...
    def evaluate(self, **kw): ...
    def compare(self, **kw): ...


class GatewayBackend(Backend):
    def __init__(self, base):
        self.base = base.rstrip("/")
        from services.common import get_json, post_json
        self._get, self._post = get_json, post_json

    def datasets(self):
        return self._get(f"{self.base}/datasets")["datasets"]

    def models(self, ds):
        return self._get(f"{self.base}/models/{ds}")["models"]

    def status(self, ds):
        return self._get(f"{self.base}/status/{ds}")

    def search(self, **kw):
        return self._post(f"{self.base}/search", kw)

    def refine(self, **kw):
        return self._post(f"{self.base}/refine", kw)

    def evaluate(self, **kw):
        return self._post(f"{self.base}/evaluate", kw)

    def compare(self, **kw):
        return self._post(f"{self.base}/compare", kw)


@st.cache_resource(show_spinner=False)
def get_backend(gateway_url: str) -> Backend:
    return GatewayBackend(gateway_url)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def highlight(text: str, terms: list[str], limit: int | None = 600) -> str:
    """HTML-escape and <mark> the query/expansion terms. ``limit`` truncates to
    a preview snippet; pass ``None`` to render the complete document text."""
    if not text:
        return "<em>(no stored text)</em>"
    if limit is not None and len(text) > limit:
        snippet = text[:limit] + "…"
    else:
        snippet = text
    safe = html.escape(snippet).replace("\n", "<br>")
    for t in sorted({t for t in terms if len(t) >= 2}, key=len, reverse=True):
        safe = re.sub(rf"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", safe)
    return safe


def query_terms(query: str, refinement: dict | None) -> list[str]:
    terms = re.findall(r"[A-Za-z]+", query or "")
    if refinement:
        terms += [w for e in refinement.get("expansions", []) for w in e.split()]
        terms += list(refinement.get("corrections", {}).values())
    return terms


# session defaults
for k, v in {"history": [], "query_box": "", "do_search": False, "last_resp": None,
             "last_latency": 0.0}.items():
    st.session_state.setdefault(k, v)


def trigger_search(q: str | None = None):
    if q is not None:
        st.session_state.query_box = q
    st.session_state.do_search = True


# ---------------------------------------------------------------------------
# Sidebar — connection, dataset, model, params, features
# ---------------------------------------------------------------------------
st.sidebar.title("🔎 IR System")
st.sidebar.caption("VSM · BM25 · BERT · Word2Vec · Hybrid")

# The UI always talks to the Service-Oriented backend: the API gateway, which
# routes to the five microservices. SOA is the system's architecture, not a
# user-facing option — so there is no in-process toggle.
gateway_url = st.sidebar.text_input(
    "Gateway URL", SERVICE_URLS["gateway"],
    help="The SOA API gateway. Start the backend with: python -m services.run_all")
backend = get_backend(gateway_url)

try:
    ds_list = backend.datasets()
except Exception as e:
    st.error(f"Cannot reach the SOA backend at {gateway_url} — {e}\n\n"
             f"Start it in a terminal with **`python -m services.run_all`**, "
             f"then reload this page.")
    st.stop()

ds_options = {d["key"]: d for d in ds_list}
dataset = st.sidebar.selectbox("Dataset", list(ds_options),
                               format_func=lambda k: ds_options[k]["name"])

# index status
status = {}
try:
    status = backend.status(dataset)
except Exception as e:
    st.sidebar.warning(f"status unavailable: {e}")
if status:
    ndocs = status.get("num_docs", "—")
    if isinstance(ndocs, int):
        ndocs = f"{ndocs:,}"
    st.sidebar.caption(
        f"📚 {ndocs} docs &nbsp;|&nbsp; "
        f"idx {'✅' if status.get('inverted_index') else '❌'} · "
        f"bert {'✅' if status.get('bert') else '❌'} · "
        f"w2v {'✅' if status.get('word2vec') else '❌'}", unsafe_allow_html=True)

try:
    available = backend.models(dataset)
except Exception as e:
    st.sidebar.error(str(e))
    available = []

if not available:
    st.warning(f"No index for **{dataset}** yet. Build one:\n\n"
               f"`python scripts/build_indexes.py --dataset {dataset} --limit 50000`")
    st.stop()

st.sidebar.divider()
model = st.sidebar.selectbox("Representation model", available,
                             format_func=lambda m: MODEL_LABELS.get(m, m))
st.sidebar.caption(MODEL_HELP.get(model, ""))

# BM25 parameters (req. #2 note 2) — shown whenever BM25 is involved
bm25_k1 = bm25_b = None
if model in ("bm25", "hybrid_parallel", "hybrid_serial"):
    with st.sidebar.expander("⚙️ BM25 parameters", expanded=(model == "bm25")):
        bm25_k1 = st.slider("k1 — term-freq saturation", 0.0, 3.0, 1.5, 0.1)
        bm25_b = st.slider("b — length normalisation", 0.0, 1.0, 0.75, 0.05)

# Hybrid configuration (req. #9 "choose hybrid model from UI")
hybrid_opts = None
if model == "hybrid_parallel":
    with st.sidebar.expander("🔀 Parallel hybrid", expanded=True):
        bases = st.multiselect("Models to fuse", [m for m in available if "hybrid" not in m],
                               default=[m for m in ("bm25", "bert") if m in available])
        fusion = st.selectbox("Fusion method", ["weighted", "rrf"],
                              help="weighted = min-max normalised weighted sum; "
                                   "rrf = Reciprocal Rank Fusion (scale-free).")
    hybrid_opts = {"models": bases, "fusion": fusion}
elif model == "hybrid_serial":
    with st.sidebar.expander("⛓️ Serial hybrid", expanded=True):
        singles = [m for m in available if "hybrid" not in m]
        s1 = st.selectbox("Stage 1 · recall", singles,
                          index=singles.index("bm25") if "bm25" in singles else 0)
        s2 = st.selectbox("Stage 2 · rerank", singles,
                          index=singles.index("bert") if "bert" in singles else 0)
        cand = st.slider("Candidate pool (stage 1)", 20, 500, 100, 20)
    hybrid_opts = {"stage1": s1, "stage2": s2, "candidate_k": cand}

st.sidebar.divider()
st.sidebar.markdown("**Additional features** (req. #5)")
use_refine = st.sidebar.toggle("Query refinement", value=False,
                               help="Off = basic pipeline. On = spell-correction, "
                                    "synonym expansion & history personalisation.")
ref_spell = ref_expand = ref_hist = False
if use_refine:
    ref_spell = st.sidebar.checkbox("Spelling correction", True)
    ref_expand = st.sidebar.checkbox("Synonym expansion (WordNet)", True)
    ref_hist = st.sidebar.checkbox("Personalise from history", True)

top_k = st.sidebar.slider("Results (top-k)", 5, 50, 10, 5)


def refine_opts():
    if not use_refine:
        return {"enabled": False}
    return {"enabled": True, "spell": ref_spell, "expand": ref_expand, "history": ref_hist}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
_mode_txt = "query refinement <b>on</b>" if use_refine else "basic pipeline"
st.markdown(
    f'''<div class="hero">
  <h1>🔎 Information Retrieval System</h1>
  <div class="bar"></div>
  <div class="sub">Searching <b>{ds_options[dataset]['name']}</b> with '''
    f'''<b>{MODEL_LABELS.get(model, model)}</b> &nbsp;·&nbsp; {_mode_txt}</div>
</div>''', unsafe_allow_html=True)

tab_search, tab_eval = st.tabs(["🔍 Search", "📊 Evaluation"])

# ===========================================================================
# SEARCH TAB
# ===========================================================================
with tab_search:
    col_q, col_btn = st.columns([0.85, 0.15])
    with col_q:
        st.text_input("Query", key="query_box", label_visibility="collapsed",
                      placeholder="Type a query, or click an example below…")
    with col_btn:
        if st.button("Search", type="primary", use_container_width=True):
            trigger_search()
    query = st.session_state.query_box

    # clickable examples
    st.caption("Try an example:")
    ex_cols = st.columns(len(EXAMPLES.get(dataset, [])) or 1)
    for i, ex in enumerate(EXAMPLES.get(dataset, [])):
        ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True,
                          on_click=trigger_search, args=(ex,))

    # live suggestions when refinement is on
    if query and use_refine:
        try:
            rf = backend.refine(dataset=dataset, query=query, history=st.session_state.history,
                                spell=ref_spell, expand=ref_expand, history_personalize=ref_hist)
            sugg = rf.get("suggestions", [])[:5]
            if sugg:
                st.caption("💡 Suggestions:")
                sc = st.columns(len(sugg))
                for i, s in enumerate(sugg):
                    sc[i].button(s, key=f"sg_{i}", use_container_width=True,
                                 on_click=trigger_search, args=(s,))
        except Exception:
            pass

    # run the search when triggered
    if st.session_state.do_search and query:
        if not st.session_state.history or st.session_state.history[-1] != query:
            st.session_state.history.append(query)
        with st.spinner(f"Retrieving with {MODEL_LABELS.get(model, model)}…"):
            t0 = time.time()
            try:
                resp = backend.search(dataset=dataset, model=model, query=query, top_k=top_k,
                                      bm25_k1=bm25_k1, bm25_b=bm25_b, refine_opts=refine_opts(),
                                      history=st.session_state.history, hybrid_opts=hybrid_opts)
                st.session_state.last_resp = resp
                st.session_state.last_latency = time.time() - t0
            except Exception as e:
                st.session_state.last_resp = {"error": str(e)}
        st.session_state.do_search = False

    resp = st.session_state.last_resp
    if resp and "error" in resp:
        st.error(f"Search failed: {resp['error']}")
    elif resp:
        ref = resp.get("refinement")
        if ref:
            with st.container(border=True):
                c = st.columns([0.2, 0.2, 0.6])
                c[0].metric("Corrections", len(ref.get("corrections", {})))
                c[1].metric("Expansion terms", len(ref.get("expansions", [])))
                c[2].markdown("**Effective query**  \n`" + (resp.get("effective_query") or "") + "`")
                if ref.get("corrections"):
                    st.markdown("Spelling fixes: " + " ".join(
                        f'<span class="pill">{html.escape(k)} → {html.escape(v)}</span>'
                        for k, v in ref["corrections"].items()), unsafe_allow_html=True)
                if ref.get("expansions"):
                    st.markdown("Expanded with: " + " ".join(
                        f'<span class="pill">+ {html.escape(w)}</span>' for w in ref["expansions"]),
                        unsafe_allow_html=True)

        results = resp.get("results", [])
        st.markdown(f"**{len(results)} results** · model `{resp.get('model', model)}` · "
                    f"{st.session_state.last_latency*1000:.0f} ms")

        terms = query_terms(resp.get("query", query), ref)
        max_score = max((abs(r["score"]) for r in results), default=1.0) or 1.0
        for r in results:
            pct = max(2, min(100, abs(r["score"]) / max_score * 100))
            comps = "".join(f'<span class="comp-badge">{html.escape(k)}={v:.3f}</span>'
                            for k, v in (r.get("components") or {}).items())
            full_text = r.get("raw_text", "")
            card = (
                f'<div class="result-card">'
                f'<span class="rank-badge">#{r["rank"]}</span> '
                f'<span class="docid">{html.escape(str(r["doc_id"]))}</span>'
                f'<span class="score-val">{r["score"]:.4f}</span>'
                f'<div class="score-bar-bg"><div class="score-bar" style="width:{pct:.0f}%"></div></div>'
                f'<div class="snippet">{highlight(full_text, terms)}</div>'
                f'{comps}'
                f'</div>'
            )
            st.markdown(card, unsafe_allow_html=True)
            # Full original document, readable inline (no need to open it elsewhere).
            if full_text:
                with st.expander(f"📄 Read full document · {r['doc_id']}  ({len(full_text):,} chars)"):
                    st.markdown(
                        f'<div class="snippet">{highlight(full_text, terms, limit=None)}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("Enter a query above or click an example to begin.")

    if st.session_state.history:
        with st.expander(f"🕘 Search history ({len(st.session_state.history)})"):
            st.markdown("\n".join(f"- {h}" for h in reversed(st.session_state.history[-15:])))
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()

# ===========================================================================
# EVALUATION TAB
# ===========================================================================
with tab_eval:
    st.subheader("Standard IR evaluation (requirement #8)")
    st.caption("Metrics over the dataset's qrels: MAP · Recall · Precision@10 · nDCG@10. "
               "Only judged queries are scored.")
    ec1, ec2, ec3 = st.columns(3)
    nq = ec1.number_input("Judged queries", 5, 500, 30, 5)
    depth = ec2.number_input("Retrieval depth", 10, 1000, 100, 10)
    ec3.caption(f"Model under test: **{MODEL_LABELS.get(model, model)}**"
                + ("  ·  refinement on" if use_refine else ""))

    b1, b2 = st.columns(2)
    run_eval = b1.button("▶ Evaluate model", use_container_width=True)
    run_cmp = b2.button("⚖ Compare before/after refinement", use_container_width=True)

    if run_eval:
        sig = ("eval", gateway_url, dataset, model, int(nq), int(depth), bm25_k1, bm25_b,
               str(refine_opts()), str(hybrid_opts))
        cache = st.session_state.setdefault("_eval_cache", {})
        if sig in cache:
            r = cache[sig]
            st.caption("Showing cached result for these settings.")
        else:
            with st.spinner(f"Evaluating {MODEL_LABELS.get(model, model)} over {int(nq)} queries…"):
                r = backend.evaluate(dataset=dataset, model=model, num_queries=int(nq),
                                     eval_depth=int(depth), bm25_k1=bm25_k1, bm25_b=bm25_b,
                                     refine_opts=refine_opts(), hybrid_opts=hybrid_opts)
            cache[sig] = r
        m = r["metrics"]
        cols = st.columns(4)
        cols[0].metric("MAP", f"{m['MAP']:.4f}")
        cols[1].metric("Recall", f"{m['Recall']:.4f}")
        cols[2].metric("P@10", f"{m['P@10']:.4f}")
        cols[3].metric("nDCG@10", f"{m['nDCG@10']:.4f}")
        st.caption(f"over {m['num_queries']} queries")
        import pandas as pd
        st.bar_chart(pd.DataFrame({"score": {k: m[k] for k in ("MAP", "Recall", "P@10", "nDCG@10")}}))

    if run_cmp:
        # The "after" run uses the features you selected in the sidebar; if the
        # refinement toggle is off, it falls back to full refinement so the
        # button still demonstrates the effect.
        after_cfg = refine_opts()
        if not after_cfg.get("enabled"):
            after_cfg = {"enabled": True, "spell": True, "expand": True, "history": False}
        sig = ("cmp", gateway_url, dataset, model, int(nq), int(depth), bm25_k1, bm25_b,
               str(hybrid_opts), str(after_cfg))
        cache = st.session_state.setdefault("_eval_cache", {})
        if sig in cache:
            r = cache[sig]
            st.caption("Showing cached comparison for these settings.")
        else:
            with st.spinner("Running baseline, then refined (two full passes over the SOA backend)…"):
                r = backend.compare(dataset=dataset, model=model, num_queries=int(nq),
                                    eval_depth=int(depth), bm25_k1=bm25_k1, bm25_b=bm25_b,
                                    hybrid_opts=hybrid_opts, refine_opts=after_cfg)
            cache[sig] = r
        _feats = [n for n, on in (("spelling", after_cfg.get("spell")),
                                  ("synonym expansion", after_cfg.get("expand")),
                                  ("history", after_cfg.get("history"))) if on]
        st.caption("**After** = baseline + " + (", ".join(_feats) if _feats else "no features"))
        import pandas as pd
        metrics = ["MAP", "Recall", "P@10", "nDCG@10"]
        before = {k: r["before"][k] for k in metrics}
        after = {k: r["after"][k] for k in metrics}
        delta = {k: r["delta"][k] for k in metrics}

        # --- headline cards: after value + signed delta vs baseline ---
        st.markdown("#### Before (basic)  →  After (refined)")
        cols = st.columns(4)
        for i, k in enumerate(metrics):
            cols[i].metric(k, f"{after[k]:.4f}", delta=f"{delta[k]:+.4f}",
                           help=f"baseline {before[k]:.4f} → refined {after[k]:.4f}")

        # --- charts side by side: grouped before/after + per-metric change ---
        df_cmp = pd.DataFrame({"before": before, "after": after}).reindex(metrics)
        df_delta = pd.DataFrame({"Δ (after − before)": delta}).reindex(metrics)
        g1, g2 = st.columns([3, 2])
        with g1:
            st.markdown("**Per-metric: before vs after**")
            try:
                st.bar_chart(df_cmp, stack=False, color=["#9aa7b8", "#1d4e89"])
            except TypeError:  # older Streamlit without stack/color kwargs
                st.bar_chart(df_cmp)
        with g2:
            st.markdown("**Change (Δ)**")
            st.bar_chart(df_delta, color="#b3261e")

        # --- detailed table: before / after / delta / % change, deltas coloured ---
        def _pct(b, a):
            return ((a - b) / b * 100.0) if b else 0.0
        table = pd.DataFrame({
            "before (basic)": before,
            "after (refined)": after,
            "Δ delta": delta,
            "% change": {k: _pct(before[k], after[k]) for k in metrics},
        }).reindex(metrics)

        def _delta_color(v):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return ""
            return "color:#0b7a34;" if x > 0 else ("color:#b3261e;" if x < 0 else "color:#888;")

        sty = table.style.format({"before (basic)": "{:.4f}", "after (refined)": "{:.4f}",
                                  "Δ delta": "{:+.4f}", "% change": "{:+.1f}%"})
        try:
            sty = sty.map(_delta_color, subset=["Δ delta", "% change"])
        except AttributeError:  # pandas < 2.1
            sty = sty.applymap(_delta_color, subset=["Δ delta", "% change"])
        st.dataframe(sty, use_container_width=True)
        st.caption("Spell-correction is typically neutral on clean queries; aggressive synonym "
                   "expansion can cause query drift — see docs/REPORT.md for the analysis.")
