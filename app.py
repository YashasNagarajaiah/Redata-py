import os
import math
import time
from datetime import datetime
from urllib.parse import urlencode as _urlencode

import feedparser
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
from openai import OpenAI
from markupsafe import Markup, escape

import altair as alt
from vega_datasets import data

from charts import (
    line_trend_highlight,
    topn_bar_by_year,
    simple_line,
    grouped_bar,
    scatter_plot,
    area_chart,
    histogram,
)

app = Flask(__name__)


# --- Groq/OpenAI-compatible client ---
load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# -------- Feed groups per tab -----------------------------------------------
FEED_GROUPS = {
    "dashboard": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://rss.cnn.com/rss/edition_world.rss",
        "https://www.ft.com/rss/home/uk",
        "https://www.economist.com/business/rss.xml",
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    ],
    "climate": [
        "https://www.ft.com/rss/home/uk",  # e.g., economics focused
        "https://www.economist.com/business/rss.xml",
    ],
    "air": [
        "https://www.sciencedaily.com/rss/earth_climate/air_quality.xml",
        "http://airqualitynews.com/feed",
    ],
    "waste": [
        "https://earth911.com/feed",
        "https://recyclinginside.com/rss-feeds/",
    ],
    "markets": [
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "http://feeds.nbcnews.com/feeds/topstories",
    ],
    "energy": [
        "https://news.un.org/feed/subscribe/en/news/region/global/feed/rss.xml",  # Renewable energy industry news
    ],
    "policy": [
        "https://news.un.org/feed/subscribe/en/news/topic/un-affairs/feed/rss.xml",  # Policy & market insights
    ],
}


# -------- Feed processing ----------------------------------------------------
def get_image(entry):
    """Try media fields, then enclosures."""
    img = None
    media = entry.get("media_content") or entry.get("media_thumbnail") or []
    if media:
        img = media[0].get("url")
    if not img:
        for l in entry.get("links", []):
            if l.get("rel") == "enclosure" and str(l.get("type", "")).startswith("image/"):
                img = l.get("href")
                break
    return img

def entry_time(entry):
    """time.struct_time for sorting; safe fallback."""
    return (
        entry.get("published_parsed")
        or entry.get("updated_parsed")
        or time.gmtime(0)
    )

def normalize_entry(source, entry):
    return {
        "source": source,
        "title": (entry.get("title") or "").strip(),
        "link": entry.get("link") or "",
        "published": entry.get("published") or entry.get("updated") or "",
        "summary": entry.get("summary") or entry.get("description") or "",
        "image": get_image(entry),
        "sort_time": entry_time(entry),
    }

# Simple cache (per tab feed set), TTL 5 minutes
CACHE = {}  # key -> { ts, items }
TTL_SECONDS = 300

def load_articles_from(feeds):
    now_bucket = int(time.time() // TTL_SECONDS)
    key = ("|".join(sorted(feeds)), now_bucket)
    if key in CACHE:
        return CACHE[key]["items"]

    items = []
    for url in feeds:
        parsed = feedparser.parse(url)
        source_name = parsed.feed.get("title", url)
        for e in parsed.entries:
            a = normalize_entry(source_name, e)
            if a["title"] and a["link"]:
                items.append(a)

    # dedupe by link
    seen = set()
    deduped = []
    for a in items:
        link = a["link"]
        if link and link not in seen:
            seen.add(link)
            deduped.append(a)

    deduped.sort(key=lambda a: a["sort_time"], reverse=True)
    CACHE[key] = {"ts": time.time(), "items": deduped}
    return deduped

def paginate(items, page, per_page):
    total = len(items)
    total_pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], page, total_pages, total

# -------- Template context helpers ------------------------------------------
@app.context_processor
def inject_globals():
    return dict(
        urlencode=lambda d: _urlencode(d, doseq=True),
        now=lambda: datetime.now(),
    )

# -------- Renderer + routes --------------------------------------------------
def render_feed_page(tab_key: str, template_name: str, page_title: str):
    page     = request.args.get("page", 1, type=int)
    per_page = 10
    feeds    = FEED_GROUPS.get(tab_key, FEED_GROUPS["dashboard"])
    items    = load_articles_from(feeds)

    # tab-aware search
    q_raw = (request.args.get("q") or "").strip()
    if q_raw:
        q = q_raw.lower()
        items = [a for a in items if q in a["title"].lower() or q in a["summary"].lower()]

    page_items, page, total_pages, total = paginate(items, page, per_page)

    return render_template(
        template_name,
        page_title=page_title,
        articles=page_items,
        page=page,
        total_pages=total_pages,
        total_articles=total,
        current_tab=tab_key,
        region=request.args.get("region", "Global"),
        timeframe=request.args.get("timeframe", "7D"),
        dtype=request.args.get("dtype", "Raw"),
        q=q_raw,
    )
    
# ----- Site-wide search
@app.route("/search")
def search():
    query = (request.args.get("q") or "").strip()
    tab   = request.args.get("tab", "dashboard")  # search current tab by default
    feeds = FEED_GROUPS.get(tab, FEED_GROUPS["dashboard"])
    items = load_articles_from(feeds)
    ql = query.lower()
    results = [a for a in items if ql and (ql in a["title"].lower() or ql in a["summary"].lower())]

    # Reuse feed layout to keep look consistent
    return render_template(
        "search.html",
        page_title=f"Search — {tab.title()}",
        articles=results,
        page=1, total_pages=1, total_articles=len(results),
        current_tab=tab,
        region=request.args.get("region", "Global"),
        timeframe=request.args.get("timeframe", "7d"),
        dtype=request.args.get("dtype", "Raw"),
        q=query,
    )
    
# ----- Assistant pages (hooks your assistant.html)
@app.get("/assistant", endpoint="assistant")
def assistant():
    return render_template("assistant.html", answer=None, current_tab="assistant")

@app.post("/assistant/ask")
def assistant_ask():
    q = (request.form.get("q") or "").strip()
    if not q:
        return redirect(url_for("assistant"))  # <- fixed

    if not client.api_key:
        safe_err = Markup("<em>GROQ_API_KEY not configured.</em>")
        return render_template("assistant.html", answer=safe_err, current_tab="assistant")

    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Zain AI, a helpful assistant for climate, waste, energy, "
                        "and market signals. Keep answers concise and practical."
                    ),
                },
                {"role": "user", "content": q},
            ],
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
        safe_html = Markup(escape(text).replace("\n", "<br>"))
        return render_template("assistant.html", answer=safe_html, current_tab="assistant")

    except Exception as e:
        err_html = Markup("<em>Failed to fetch reply.</em><br>") + Markup.escape(str(e))
        return render_template("assistant.html", answer=err_html, current_tab="assistant")


@app.route("/graphs")
def graphs():
    specs = {
        "line_trend": line_trend_highlight().to_dict(),
        "topn_bar": topn_bar_by_year().to_dict(),
        "simple_line": simple_line().to_dict(),
        "grouped_bar": grouped_bar().to_dict(),
        "scatter_plot": scatter_plot().to_dict(),
        "area_chart": area_chart().to_dict(),
        "histogram": histogram().to_dict(),
    }
    return render_template("graphs.html", specs=specs, current_tab="graphs")



# ----- Feeds: make dashboard the site root
@app.route("/")
def dashboard():
    return render_feed_page("dashboard", "dashboard.html", "Global & Local")

@app.route("/climate")
def climate():
    return render_feed_page("climate", "climate.html", "Climate & Weather")

@app.route("/air")
def air():
    return render_feed_page("air", "air.html", "Air Quality & Pollution")

@app.route("/waste")
def waste():
    return render_feed_page("waste", "waste.html", "Waste & Circular Economy")

@app.route("/markets")
def markets():
    return render_feed_page("markets", "markets.html", "Market Prices & Trade")

@app.route("/energy")
def energy():
    return render_feed_page("energy", "energy.html", "Energy & Carbon")

@app.route("/policy")
def policy():
    return render_feed_page("policy", "policy.html", "Policy")
    


# ----- Main
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. /assistant/ask will show an error.")
    app.run(debug=True)
