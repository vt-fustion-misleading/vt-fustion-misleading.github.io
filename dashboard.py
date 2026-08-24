"""Streamlit dashboard for browsing inferred misconceptions and their items.

Run from the repo root (or this folder):

    streamlit run misconception_modeling/dashboard.py

Reads from misconception_modeling/outputs/:
  - taxonomy.json                 (categories + per-distractor assignments)
  - misconception_inferences.jsonl (per-distractor inference records)
  - items.json                    (item metadata incl. image links)
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

OUTPUT_DIR = Path(__file__).parent / "outputs"
BAR_COLOR = "#4269d0"  # single hue: the chart encodes magnitude only

st.set_page_config(page_title="Misconception Explorer", layout="wide")


@st.cache_data
def load_data():
    taxonomy = json.loads((OUTPUT_DIR / "taxonomy.json").read_text())
    inferences = [
        json.loads(line)
        for line in (OUTPUT_DIR / "misconception_inferences.jsonl").read_text().splitlines()
        if line.strip()
    ]
    items = {it["item_id"]: it for it in json.loads((OUTPUT_DIR / "items.json").read_text())}

    assignments = taxonomy["assignments"]
    for rec in inferences:
        rec["category"] = assignments.get(rec["key"])
    return taxonomy, inferences, items


taxonomy, inferences, items = load_data()
categories = {c["code"]: c for c in taxonomy["categories"]}

# distractor records grouped by category
by_category = defaultdict(list)
for rec in inferences:
    if rec["category"]:
        by_category[rec["category"]].append(rec)

# ---------------- sidebar ----------------
st.sidebar.title("Misconception Explorer")

cat_codes = sorted(by_category, key=lambda c: -len(by_category[c]))


def cat_label(code):
    n_items = len({r["item_id"] for r in by_category[code]})
    return f"{categories[code]['name']}  ({n_items} items)"


selection = st.sidebar.radio(
    "Misconception",
    ["(overview)"] + cat_codes,
    format_func=lambda c: "All misconceptions (overview)" if c == "(overview)" else cat_label(c),
)

st.sidebar.divider()
tests = sorted({r["test"] for r in inferences})
test_filter = st.sidebar.multiselect("Test", tests, default=tests)
error_kinds = sorted({r["error_kind"] for r in inferences})
kind_filter = st.sidebar.multiselect("Error kind", error_kinds, default=error_kinds)
min_prop = st.sidebar.slider("Min. proportion choosing the wrong answer", 0.0, 1.0, 0.0, 0.05)


def keep(rec):
    return (
        rec["test"] in test_filter
        and rec["error_kind"] in kind_filter
        and rec["proportion"] >= min_prop
    )


# ---------------- overview ----------------
if selection == "(overview)":
    st.title("Misconception taxonomy overview")
    st.caption(
        f"{len(taxonomy['categories'])} categories · "
        f"{len(inferences)} distractor inferences · {len(items)} items · "
        f"consistency: {taxonomy.get('consistency_flag')}"
    )

    rows = []
    for code in cat_codes:
        recs = [r for r in by_category[code] if keep(r)]
        if not recs:
            continue
        rows.append(
            {
                "Misconception": categories[code]["name"],
                "Items": len({r["item_id"] for r in recs}),
                "Distractors": len(recs),
                "Mean proportion": sum(r["proportion"] for r in recs) / len(recs),
                "Definition": categories[code]["definition"],
            }
        )
    df = pd.DataFrame(rows)

    import altair as alt

    chart = (
        alt.Chart(df)
        .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("Items:Q", title="Items"),
            y=alt.Y("Misconception:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
            tooltip=["Misconception", "Items", "Distractors", alt.Tooltip("Mean proportion", format=".2f")],
        )
        .properties(height=max(300, 26 * len(df)))
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        df.style.format({"Mean proportion": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.info("Pick a misconception in the sidebar to see its items and images.")
    st.stop()

# ---------------- single category ----------------
cat = categories[selection]
recs = [r for r in by_category[selection] if keep(r)]

st.title(cat["name"])
st.caption(f"code: `{cat['code']}`")
st.markdown(f"**Definition:** {cat['definition']}")

item_ids = sorted({r["item_id"] for r in recs})
c1, c2, c3 = st.columns(3)
c1.metric("Items", len(item_ids))
c2.metric("Distractors", len(recs))
c3.metric(
    "Mean proportion",
    f"{sum(r['proportion'] for r in recs) / len(recs):.2f}" if recs else "—",
)

if not recs:
    st.warning("No records match the current filters.")
    st.stop()

sort_by = st.selectbox(
    "Sort items by",
    ["max distractor proportion (desc)", "item difficulty (hardest first)", "item id"],
)

per_item = {iid: [r for r in recs if r["item_id"] == iid] for iid in item_ids}
if sort_by.startswith("max"):
    item_ids.sort(key=lambda i: -max(r["proportion"] for r in per_item[i]))
elif sort_by.startswith("item difficulty"):
    item_ids.sort(key=lambda i: items.get(i, {}).get("prop_correct") or 1.0)

st.divider()

for iid in item_ids:
    item = items.get(iid, {})
    img_col, info_col = st.columns([2, 3], gap="large")

    with img_col:
        if item.get("image_link"):
            st.image(item["image_link"], width="stretch")
        else:
            st.caption("No image available.")

    with info_col:
        st.subheader(item.get("question", iid))
        meta = " · ".join(
            str(v)
            for v in [
                item.get("test", "").upper(),
                item.get("graph_type"),
                item.get("task_category"),
                f"{item.get('n_responses')} responses" if item.get("n_responses") else None,
                f"{item['prop_correct']:.0%} correct" if item.get("prop_correct") is not None else None,
            ]
            if v
        )
        st.caption(meta)
        if item.get("options"):
            st.markdown(f"**Options:** {', '.join(map(str, item['options']))}")
        st.markdown(f"**Correct answer:** {item.get('correct_answer', '—')}")
        if item.get("misleader_type"):
            st.markdown(f"**Misleader:** {item['misleader_type']}")

        for rec in sorted(per_item[iid], key=lambda r: -r["proportion"]):
            with st.container(border=True):
                st.markdown(
                    f"**Wrong answer:** {rec['wrong_answer']} — "
                    f"{rec['proportion']:.0%} of responses "
                    f"(mean ability {rec['mean_ability_of_choosers']:.2f}, "
                    f"{rec['error_kind']})"
                )
                st.markdown(f"*{rec['short_label']}* — {rec['misconception']}")
                with st.expander("Reasoning trace"):
                    st.write(rec["reasoning_trace"])

    st.divider()
