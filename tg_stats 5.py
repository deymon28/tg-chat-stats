#!/usr/bin/env python3
# tg_stats.py
import argparse, json, re, emoji
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langdetect import detect
import regex as rej
import networkx as nx
from textblob import TextBlob
import urllib.parse

# ---------- CONFIG ----------
STOP = {
    "uk": {"і", "та", "що", "в", "на", "з", "до", "за", "як", "його", "її", "але", "якщо", "або"},
    "ru": {"и", "в", "на", "с", "что", "за", "по", "как", "его", "её", "но", "если", "или"},
    "en": {"the", "and", "or", "but", "if", "in", "on", "to", "of", "a", "an", "it", "is", "this", "that"},
}
STOPWORDS = set().union(*STOP.values())
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_RE = rej.compile(
    f"[{''.join(map(rej.escape, emoji.EMOJI_DATA.keys()))}]",
    flags=rej.UNICODE,
)

# Time buckets
TIME_BUCKETS = {
    "night": (0, 5),
    "morning": (5, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "late_night": (21, 24)
}

# Sentiment thresholds
SENTIMENT_THRESHOLDS = {
    "positive": 0.2,
    "negative": -0.2
}


# ---------- HELPER FUNCTIONS ----------
def get_time_bucket(hour):
    for bucket, (start, end) in TIME_BUCKETS.items():
        if start <= hour < end:
            return bucket
    return "night"


def get_sentiment_label(score):
    if score > SENTIMENT_THRESHOLDS["positive"]:
        return "positive"
    elif score < SENTIMENT_THRESHOLDS["negative"]:
        return "negative"
    return "neutral"


def extract_domain(url):
    try:
        domain = urllib.parse.urlparse(url).netloc
        return domain.replace("www.", "")
    except:
        return "invalid"


def analyze_sentiment(text, lang='en'):
    if not text.strip():
        return 0.0

    try:
        if lang in ['uk', 'ru']:
            # Placeholder for multilingual sentiment analysis
            return 0.0
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except:
        return 0.0


def calculate_session_stats(messages):
    if not messages:
        return []

    # Sort messages chronologically
    messages_sorted = sorted(messages, key=lambda m: int(m["date_unixtime"]))

    sessions = []
    current_session = []
    prev_time = datetime.fromtimestamp(int(messages_sorted[0]["date_unixtime"]))
    prev_user = messages_sorted[0].get("from") or "Unknown"

    for i in range(1, len(messages_sorted)):
        m = messages_sorted[i]
        if m.get("type") != "message":
            continue

        curr_time = datetime.fromtimestamp(int(m["date_unixtime"]))
        curr_user = m.get("from") or "Unknown"
        time_diff = (curr_time - prev_time).total_seconds()

        # Same user and within 1 hour gap
        if curr_user == prev_user and time_diff < 3600:
            if not current_session:
                current_session.append(messages_sorted[i - 1])
            current_session.append(m)
        else:
            if current_session:
                sessions.append({
                    "user": prev_user,
                    "start": current_session[0]["date_unixtime"],
                    "end": current_session[-1]["date_unixtime"],
                    "duration": (datetime.fromtimestamp(int(current_session[-1]["date_unixtime"])) -
                                 datetime.fromtimestamp(int(current_session[0]["date_unixtime"]))).total_seconds(),
                    "message_count": len(current_session)
                })
            current_session = []

        prev_time = curr_time
        prev_user = curr_user

    # Add last session
    if current_session:
        sessions.append({
            "user": prev_user,
            "start": current_session[0]["date_unixtime"],
            "end": current_session[-1]["date_unixtime"],
            "duration": (datetime.fromtimestamp(int(current_session[-1]["date_unixtime"])) -
                         datetime.fromtimestamp(int(current_session[0]["date_unixtime"]))).total_seconds(),
            "message_count": len(current_session)
        })

    return sessions


def find_silence_intervals(messages):
    if not messages:
        return []

    # Sort messages chronologically
    messages_sorted = sorted(messages, key=lambda m: int(m["date_unixtime"]))
    silences = []

    for i in range(1, len(messages_sorted)):
        prev_time = datetime.fromtimestamp(int(messages_sorted[i - 1]["date_unixtime"]))
        curr_time = datetime.fromtimestamp(int(messages_sorted[i]["date_unixtime"]))
        gap = (curr_time - prev_time).total_seconds()

        if gap > 3600:  # 1 hour silence
            silences.append({
                "start": messages_sorted[i - 1]["date_unixtime"],
                "end": messages_sorted[i]["date_unixtime"],
                "duration": gap
            })

    return sorted(silences, key=lambda x: x["duration"], reverse=True)


# ---------- LOAD ----------
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- ANALYSIS ----------
def analyse(messages):
    # Pre-index messages for reply analysis
    msg_map = {m["id"]: m for m in messages if m.get("type") == "message"}

    S = defaultdict(Counter)
    timeline = Counter()
    weekday = [0] * 7
    hour = [0] * 24
    time_buckets = {k: 0 for k in TIME_BUCKETS}
    lang_dist = Counter()
    reply_depth = Counter()
    reactions_total = 0
    edited_msgs = 0
    deleted_msgs = 0
    longest_msg = {"len": 0, "text": "", "author": ""}
    longest_voice = {"sec": 0, "author": ""}
    fwd_sources = Counter()
    sticker_names = Counter()
    emoji_per_user = defaultdict(Counter)
    domain_counter = Counter()
    sentiment_counter = Counter()
    keyword_counter = Counter()
    punctuation_counter = Counter()
    code_switches = Counter()
    response_times = []
    interaction_matrix = defaultdict(lambda: defaultdict(int))
    media_stats = {
        "photos": {"count": 0, "total_size": 0, "resolutions": []},
        "videos": {"count": 0, "total_size": 0, "durations": [], "resolutions": []},
        "voice": {"count": 0, "total_size": 0, "total_duration": 0, "durations": []},
        "stickers": {"count": 0, "emoji": Counter()},
        "gifs": {"count": 0, "total_size": 0},
        "documents": {"count": 0, "total_size": 0, "types": Counter()}
    }

    # Initialize parents for reply chains
    parents = {}

    def new_user():
        return {
            "msgs": 0,
            "chars": 0,
            "words": 0,
            "emojis": 0,
            "tokens": Counter(),
            "stickers": 0,
            "photos": 0,
            "videos": 0,
            "gifs": 0,
            "voice_sec": 0,
            "replies": 0,
            "forwards": 0,
            "reactions_received": 0,
            "edited": 0,
            "deleted": 0,
            "initiations": 0,
            "response_time": [],
            "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
            "media_sizes": defaultdict(int),
            "domains": Counter(),
            "punctuation": Counter(),
            "sentence_lengths": []
        }

    users = defaultdict(new_user)

    # Technical keywords
    TECH_KEYWORDS = {"pycharm", "python", "plugin", "refactor", "code", "github", "git",
                     "program", "debug", "syntax", "algorithm", "database", "server",
                     "api", "framework", "library", "function", "variable", "class", "object"}

    # Punctuation patterns
    PUNCTUATION = {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", "-", "'", "\""}

    # Session analysis
    sessions = calculate_session_stats(messages)
    silence_intervals = find_silence_intervals(messages)

    for m in messages:
        if m.get("type") != "message":
            continue

        # Basics
        dt = datetime.fromtimestamp(int(m["date_unixtime"]))
        timeline[dt] += 1
        weekday[dt.weekday()] += 1
        hour[dt.hour] += 1
        time_buckets[get_time_bucket(dt.hour)] += 1

        frm = m.get("from") or "Unknown"
        user = users[frm]
        user["msgs"] += 1

        # Check if this is a new topic initiation (not a reply)
        if not m.get("reply_to_message_id"):
            user["initiations"] += 1

        # Replies and response times
        reply_to = m.get("reply_to_message_id")
        if reply_to:
            user["replies"] += 1
            parents[m["id"]] = reply_to

            # Calculate response time
            if reply_to in msg_map:
                parent_msg = msg_map[reply_to]
                parent_time = datetime.fromtimestamp(int(parent_msg["date_unixtime"]))
                response_sec = (dt - parent_time).total_seconds()
                user["response_time"].append(response_sec)
                response_times.append(response_sec)

                # Track who replies to whom
                parent_author = parent_msg.get("from") or "Unknown"
                interaction_matrix[parent_author][frm] += 1

            # Reply depth
            d = 1
            pid = reply_to
            while pid in parents:
                d += 1
                pid = parents[pid]
            reply_depth[d] += 1

        # Reactions
        reactions = m.get("reactions", [])
        reactions_total += sum(r["count"] for r in reactions)
        for r in reactions:
            emoji_per_user[frm][r["emoji"]] += r["count"]
            user["reactions_received"] += r["count"]

        # Edited/deleted
        if m.get("edited"):
            edited_msgs += 1
            user["edited"] += 1
        if m.get("text") == "This message was deleted":
            deleted_msgs += 1
            user["deleted"] += 1

        # Media analysis
        mt = m.get("media_type")
        file_size = m.get("file_size", 0) or m.get("photo_file_size", 0)

        if mt == "voice_message":
            sec = m.get("duration_seconds", 0)
            user["voice_sec"] += sec
            media_stats["voice"]["count"] += 1
            media_stats["voice"]["total_duration"] += sec
            media_stats["voice"]["durations"].append(sec)
            user["media_sizes"]["voice"] += file_size

            if sec > longest_voice["sec"]:
                longest_voice["sec"] = sec
                longest_voice["author"] = frm

        elif mt == "video_file":
            user["videos"] += 1
            media_stats["videos"]["count"] += 1
            media_stats["videos"]["total_size"] += file_size
            media_stats["videos"]["durations"].append(m.get("duration_seconds", 0))
            if "width" in m and "height" in m:
                media_stats["videos"]["resolutions"].append(f"{m['width']}x{m['height']}")
            user["media_sizes"]["video"] += file_size

        elif mt == "photo":
            user["photos"] += 1
            media_stats["photos"]["count"] += 1
            media_stats["photos"]["total_size"] += file_size
            if "width" in m and "height" in m:
                media_stats["photos"]["resolutions"].append(f"{m['width']}x{m['height']}")
            user["media_sizes"]["photo"] += file_size

        elif mt == "sticker":
            user["stickers"] += 1
            media_stats["stickers"]["count"] += 1
            sticker_emoji = m.get("sticker_emoji", "unknown")
            media_stats["stickers"]["emoji"][sticker_emoji] += 1
            sticker_names[m.get("file", "").split("/")[-1]] += 1
            user["media_sizes"]["sticker"] += file_size

        elif mt == "animation":
            user["gifs"] += 1
            media_stats["gifs"]["count"] += 1
            media_stats["gifs"]["total_size"] += file_size
            user["media_sizes"]["gif"] += file_size

        elif mt == "document":
            media_stats["documents"]["count"] += 1
            media_stats["documents"]["total_size"] += file_size
            ext = m.get("file", "").split(".")[-1].lower()
            media_stats["documents"]["types"][ext] += 1
            user["media_sizes"]["document"] += file_size

        # Forwards
        if m.get("forwarded_from"):
            fwd_sources[m["forwarded_from"]] += 1
            user["forwards"] += 1

        # Text analysis
        txt = ""
        text_entities = m.get("text_entities", [])
        for ent in text_entities:
            text = ent.get("text", "")
            txt += text

            # Link analysis
            if ent.get("type") == "link":
                domain = extract_domain(text)
                domain_counter[domain] += 1
                user["domains"][domain] += 1

        txt = txt.strip()
        if txt:
            # Language detection
            lang = "und"
            try:
                lang = detect(txt)
                lang_dist[lang] += 1
            except Exception:
                pass

            # Sentiment analysis
            sentiment = analyze_sentiment(txt, lang)
            sentiment_label = get_sentiment_label(sentiment)
            sentiment_counter[sentiment_label] += 1
            user["sentiment"][sentiment_label] += 1

            # Technical keyword analysis
            words = re.findall(r'\b\w+\b', txt.lower())
            for word in words:
                if word in TECH_KEYWORDS:
                    keyword_counter[word] += 1

            # Punctuation analysis
            for char in txt:
                if char in PUNCTUATION:
                    punctuation_counter[char] += 1
                    user["punctuation"][char] += 1

            # Sentence length analysis
            sentences = re.split(r'[.!?]', txt)
            for sentence in sentences:
                if sentence.strip():
                    words = sentence.split()
                    user["sentence_lengths"].append(len(words))

            user["chars"] += len(txt)
            user["words"] += len(txt.split())

            if len(txt) > longest_msg["len"]:
                longest_msg["len"] = len(txt)
                longest_msg["text"] = txt[:100] + "..."
                longest_msg["author"] = frm

            # Emoji analysis
            emojis_in_text = EMOJI_RE.findall(txt)
            user["emojis"] += len(emojis_in_text)
            for e in emojis_in_text:
                emoji_per_user[frm][e] += 1

            # Token analysis
            norm = re.sub(r"\W+", " ", URL_RE.sub("", MENTION_RE.sub("", EMOJI_RE.sub("", txt.lower())))).strip()
            for w in norm.split():
                if w not in STOPWORDS and len(w) > 2:
                    S["tokens"][w] += 1
                    user["tokens"][w] += 1

    # Calculate media stats
    for media_type in ["photos", "videos", "voice"]:
        if media_stats[media_type].get("count", 0) > 0:
            media_stats[media_type]["avg_size"] = (
                    media_stats[media_type]["total_size"] / media_stats[media_type]["count"]
            )

    # Calculate user averages
    for user_data in users.values():
        if user_data["msgs"] > 0:
            user_data["avg_chars"] = user_data["chars"] / user_data["msgs"]
            user_data["avg_words"] = user_data["words"] / user_data["msgs"]
            if user_data["sentence_lengths"]:
                user_data["avg_sentence_length"] = (
                        sum(user_data["sentence_lengths"]) / len(user_data["sentence_lengths"])
                )
            else:
                user_data["avg_sentence_length"] = 0

        if user_data["response_time"]:
            user_data["avg_response_time"] = sum(user_data["response_time"]) / len(user_data["response_time"])

    # Calculate overall averages
    avg_response = sum(response_times) / len(response_times) if response_times else 0

    # Interaction network
    interaction_graph = nx.DiGraph()
    for source, targets in interaction_matrix.items():
        for target, weight in targets.items():
            interaction_graph.add_edge(source, target, weight=weight)

    # Centrality measures
    centrality = {}
    if interaction_graph.nodes():
        centrality["degree"] = nx.degree_centrality(interaction_graph)
        centrality["betweenness"] = nx.betweenness_centrality(interaction_graph)
        centrality["closeness"] = nx.closeness_centrality(interaction_graph)
        centrality["pagerank"] = nx.pagerank(interaction_graph)

    return {
        "timeline": {d.isoformat(): c for d, c in timeline.items()},
        "weekday": weekday,
        "hour": hour,
        "time_buckets": time_buckets,
        "lang_dist": lang_dist,
        "tokens": S["tokens"],
        "reactions_total": reactions_total,
        "edited_msgs": edited_msgs,
        "deleted_msgs": deleted_msgs,
        "longest_msg": longest_msg,
        "longest_voice": longest_voice,
        "fwd_sources": fwd_sources,
        "sticker_names": sticker_names,
        "reply_depth": dict(sorted(reply_depth.items())),
        "emoji_per_user": emoji_per_user,
        "user_stats": users,
        "sessions": sessions,
        "silence_intervals": silence_intervals,
        "media_stats": media_stats,
        "domains": domain_counter,
        "sentiment": sentiment_counter,
        "keywords": keyword_counter,
        "punctuation": punctuation_counter,
        "avg_response_time": avg_response,
        "interaction_matrix": interaction_matrix,
        "interaction_graph": interaction_graph,
        "centrality": centrality,
        "code_switches": code_switches,
        "response_times": response_times
    }


# ---------- PLOTS ----------
def plots(data, out="charts"):
    Path(out).mkdir(exist_ok=True)
    sns.set_theme()

    # 1. Activity heatmap
    df = pd.DataFrame(
        [(datetime.fromisoformat(d).weekday(),
          datetime.fromisoformat(d).hour, c)
         for d, c in data["timeline"].items()],
        columns=["weekday", "hour", "count"]
    )
    table = df.pivot_table(index="weekday", columns="hour", values="count", fill_value=0)
    plt.figure(figsize=(14, 6))
    sns.heatmap(table, cmap="YlGnBu", linewidths=.5)
    plt.title("Hour-of-day × Weekday activity")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Weekday (0=Mon)")
    plt.tight_layout()
    plt.savefig(f"{out}/hourly_heatmap.png", dpi=200)
    plt.close()

    # 2. Time bucket distribution
    buckets = data["time_buckets"]
    plt.figure(figsize=(10, 6))
    plt.bar(list(buckets.keys()), list(buckets.values()))
    plt.title("Message Distribution by Time of Day")
    plt.xlabel("Time Bucket")
    plt.ylabel("Message Count")
    plt.tight_layout()
    plt.savefig(f"{out}/time_buckets.png")
    plt.close()

    # 3. Sentiment analysis
    sentiment = data["sentiment"]
    plt.figure(figsize=(8, 6))
    plt.pie(list(sentiment.values()), labels=list(sentiment.keys()), autopct='%1.1f%%')
    plt.title("Message Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(f"{out}/sentiment.png")
    plt.close()

    # 4. Interaction network
    if "interaction_graph" in data and data["interaction_graph"].nodes():
        G = data["interaction_graph"]
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=1500)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.title("User Interaction Network")
        plt.tight_layout()
        plt.savefig(f"{out}/interaction_network.png")
        plt.close()

    # 5. Media type distribution
    media_counts = {
        "Photos": data["media_stats"]["photos"]["count"],
        "Videos": data["media_stats"]["videos"]["count"],
        "Voice": data["media_stats"]["voice"]["count"],
        "Stickers": data["media_stats"]["stickers"]["count"],
        "GIFs": data["media_stats"]["gifs"]["count"],
        "Documents": data["media_stats"]["documents"]["count"]
    }
    plt.figure(figsize=(10, 6))
    plt.pie(list(media_counts.values()), labels=list(media_counts.keys()), autopct='%1.1f%%')
    plt.title("Media Type Distribution")
    plt.tight_layout()
    plt.savefig(f"{out}/media_types.png")
    plt.close()

    # 6. Response time distribution
    if data["response_times"]:
        plt.figure(figsize=(10, 6))
        sns.histplot(data["response_times"], bins=50, kde=True)
        plt.title("Response Time Distribution (seconds)")
        plt.xlabel("Response Time (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{out}/response_times.png")
        plt.close()

    # 7. Technical keywords
    if "keywords" in data and data["keywords"]:
        keywords = data["keywords"]
        top_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:15])
        plt.figure(figsize=(12, 8))
        plt.barh(list(top_keywords.keys()), list(top_keywords.values()))
        plt.title("Top Technical Keywords")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(f"{out}/tech_keywords.png")
        plt.close()

    # 8. Centrality measures
    if "centrality" in data and data["centrality"]:
        centrality = data["centrality"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("User Centrality Measures")

        # Degree centrality
        degree = centrality["degree"]
        axes[0, 0].bar(list(degree.keys()), list(degree.values()))
        axes[0, 0].set_title("Degree Centrality")
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Betweenness centrality
        betweenness = centrality["betweenness"]
        axes[0, 1].bar(list(betweenness.keys()), list(betweenness.values()))
        axes[0, 1].set_title("Betweenness Centrality")
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Closeness centrality
        closeness = centrality["closeness"]
        axes[1, 0].bar(list(closeness.keys()), list(closeness.values()))
        axes[1, 0].set_title("Closeness Centrality")
        axes[1, 0].tick_params(axis='x', rotation=45)

        # PageRank
        pagerank = centrality["pagerank"]
        axes[1, 1].bar(list(pagerank.keys()), list(pagerank.values()))
        axes[1, 1].set_title("PageRank")
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{out}/centrality.png")
        plt.close()


# ---------- REPORT ----------
def md(data, file="stats.md"):
    with open(file, "w", encoding="utf-8") as f:
        f.write("# 📊 Telegram Extended Statistics\n\n")
        total = sum(v["msgs"] for v in data["user_stats"].values())
        f.write(f"- **Messages**: {total}\n")
        f.write(f"- **Edited messages**: {data['edited_msgs']}\n")
        f.write(f"- **Deleted messages**: {data['deleted_msgs']}\n")
        f.write(f"- **Reactions given**: {data['reactions_total']}\n")
        f.write(f"- **Unique words**: {len(data['tokens'])}\n")
        f.write(f"- **Longest message**: {data['longest_msg']['len']} chars "
                f"(by {data['longest_msg']['author']})\n")
        f.write(f"- **Longest voice**: {data['longest_voice']['sec']} sec "
                f"(by {data['longest_voice']['author']})\n")
        if "avg_response_time" in data:
            f.write(f"- **Average response time**: {data['avg_response_time']:.2f} seconds\n")

        # Silence intervals
        if data["silence_intervals"]:
            longest_silence = data["silence_intervals"][0]
            hours = longest_silence["duration"] / 3600
            f.write(f"- **Longest silence**: {hours:.2f} hours\n")

        # Session stats
        if data["sessions"]:
            avg_session = sum(s["message_count"] for s in data["sessions"]) / len(data["sessions"])
            f.write(f"- **Average session length**: {avg_session:.1f} messages\n")

        f.write("\n## 🌐 Languages\n")
        for l, n in data["lang_dist"].most_common():
            f.write(f"- {l}: {n}\n")

        if "sentiment" in data:
            f.write("\n## 😊 Sentiment Analysis\n")
            for sentiment, count in data["sentiment"].items():
                f.write(f"- {sentiment.capitalize()}: {count}\n")

        if "keywords" in data and data["keywords"]:
            f.write("\n## 💻 Technical Keywords\n")
            for keyword, count in data["keywords"].most_common(10):
                f.write(f"- {keyword}: {count}\n")

        f.write("\n## 📈 Activity Patterns\n")
        for bucket, count in data["time_buckets"].items():
            f.write(f"- {bucket.capitalize()}: {count} messages\n")

        f.write("\n## 👥 User Engagement\n")
        top = sorted(data["user_stats"].items(), key=lambda x: -x[1]["msgs"])
        for u, s in top:
            f.write(f"\n### {u}\n")
            f.write(f"- Messages: {s['msgs']} ({s['msgs'] / total:.1%} of total)\n")
            if "avg_chars" in s:
                f.write(f"- Average length: {s['avg_chars']:.1f} chars, {s['avg_words']:.1f} words\n")
            f.write(f"- Topic initiations: {s['initiations']}\n")
            if "avg_response_time" in s:
                f.write(f"- Avg response time: {s['avg_response_time'] / 60:.1f} minutes\n")
            f.write(f"- Media shared: ")
            media_types = []
            if s["photos"]: media_types.append(f"{s['photos']} photos")
            if s["videos"]: media_types.append(f"{s['videos']} videos")
            if s["voice_sec"]: media_types.append(f"{s['voice_sec'] // 60} min voice")
            if s["stickers"]: media_types.append(f"{s['stickers']} stickers")
            f.write(", ".join(media_types) + "\n")
            if "sentiment" in s:
                f.write(f"- Sentiment: {s['sentiment']['positive']} 👍, "
                        f"{s['sentiment']['neutral']} 😐, "
                        f"{s['sentiment']['negative']} 👎\n")

        if "domains" in data and data["domains"]:
            f.write("\n## 🔗 Domain Analysis\n")
            for domain, count in data["domains"].most_common(10):
                f.write(f"- {domain}: {count}\n")

        f.write("\n## 📊 Plots\n")
        plots_list = [
            "hourly_heatmap.png", "time_buckets.png", "sentiment.png",
            "media_types.png", "tech_keywords.png", "response_times.png"
        ]

        if "interaction_graph" in data and data["interaction_graph"].nodes():
            plots_list.append("interaction_network.png")
            if "centrality" in data:
                plots_list.append("centrality.png")

        for p in plots_list:
            if Path(f"charts/{p}").exists():
                f.write(f"![{p}](charts/{p})\n")


# ---------- MAIN ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="result.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    msgs = load(args.json).get("messages", [])
    data = analyse(msgs)
    plots(data, out=f"{args.out}/charts")
    md(data, file=f"{args.out}/stats.md")
    with open(f"{args.out}/stats.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print("✅ Done – stats.md, stats.json and charts/ created.")