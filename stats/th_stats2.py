# -*- coding: utf-8 -*-
import json
import os
import re
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# --- SCRIPT CONFIGURATION ---
CONFIG = {
    # 1. Path to the chat export folder
    "export_folder": r"C:\Users",
    # 2. Name of the JSON file
    "json_file_name": "result.json",
    # 3. Folder to save the report
    "output_folder": "telegram_report_advanced_2",
    # 4. Main language for stop-word analysis ('russian', 'ukrainian', 'english')
    #    The script will handle mixed languages, but this helps prioritize stopwords.
    "language": "ukrainian",
    # 5. Report language ('uk', 'en')
    "report_language": "uk",
    # 6. Path to a font file that supports Cyrillic and emojis (e.g., Segoe UI Emoji, Noto Color Emoji)
    #    This font will be used for the word cloud and all plots.
    "font_path": r'C:\Windows\Fonts\seguiemj.ttf',  # <-- IMPORTANT: Update this path
    # 7. PDF Generation Settings
    "generate_pdf": True,
    "wkhtmltopdf_path": r"C:\Apps\wkhtmltopdf\bin\wkhtmltopdf.exe"  # <-- IMPORTANT: Update if needed
}

# --- LOCALIZATION STRINGS ---
UI_STRINGS = {
    'uk': {
        'report_title': "📊 Аналітичний звіт по чату Telegram",
        'general_stats': "Загальна статистика",
        'total_messages': "Всього повідомлень",
        'total_users': "Учасників",
        'chat_period': "Період чату",
        'most_active_day': "Найактивніший день",
        'messages_abbr': "пов.",
        'user_activity': "Активність користувачів",
        'messages_per_author': "Повідомлення по авторах",
        'total_messages_per_author': "Загальна кількість повідомлень по авторах",
        'message_count': "Кількість повідомлень",
        'author': "Автор",
        'time_analysis': "Аналіз активності в часі",
        'activity_heatmap': "Теплова карта активності (Година vs. День тижня)",
        'day_of_week': "День тижня",
        'hour_of_day': "Година дня",
        'content_analysis': "Аналіз контенту",
        'word_cloud': "Хмара популярних слів",
        'top_emojis': "Топ-20 найпопулярніших емодзі",
        'count': "Кількість",
        'emoji': "Емодзі",
        'top_domains': "Топ-15 доменів з посилань",
        'domain': "Домен",
        'generated_at': "Звіт згенеровано",
        'media_analysis': "Аналіз медіа",
        'media_type_distribution': "Розподіл типів повідомлень",
        'interaction_analysis': "Аналіз взаємодій",
        'reply_heatmap': "Матриця відповідей (хто кому відповідає)",
        'replied_to': "Кому відповіли",
        'replier': "Хто відповів",
        'vocabulary_analysis': "Аналіз лексики",
        'lexical_diversity': "Лексичне розмаїття (TTR)",
        'night_activity': "Нічна активність (00:00-06:00)",
        'night_owls': "Нічні сови",
        'top_sticker_packs': "Топ-10 стікер-паків",
        'sticker_pack': "Стікер-пак",
        'user_summary_table': "Зведена таблиця статистики по користувачам",
    },
    'en': {
        'report_title': "📊 Telegram Chat Analytics Report",
        'general_stats': "General Statistics",
        'total_messages': "Total Messages",
        'total_users': "Users",
        'chat_period': "Chat Period",
        'most_active_day': "Most Active Day",
        'messages_abbr': "msgs",
        'user_activity': "User Activity",
        'messages_per_author': "Messages per Author",
        'total_messages_per_author': "Total Messages per Author",
        'message_count': "Message Count",
        'author': "Author",
        'time_analysis': "Time Activity Analysis",
        'activity_heatmap': "Activity Heatmap (Hour vs. Day of Week)",
        'day_of_week': "Day of Week",
        'hour_of_day': "Hour of Day",
        'content_analysis': "Content Analysis",
        'word_cloud': "Popular Words Cloud",
        'top_emojis': "Top 20 Used Emojis",
        'count': "Count",
        'emoji': "Emoji",
        'top_domains': "Top 15 Domains from Links",
        'domain': "Domain",
        'generated_at': "Report generated at",
        'media_analysis': "Media Analysis",
        'media_type_distribution': "Message Type Distribution",
        'interaction_analysis': "Interaction Analysis",
        'reply_heatmap': "Reply Matrix (who replies to whom)",
        'replied_to': "Replied To",
        'replier': "Replier",
        'vocabulary_analysis': "Vocabulary Analysis",
        'lexical_diversity': "Lexical Diversity (TTR)",
        'night_activity': "Night Activity (00:00-06:00)",
        'night_owls': "Night Owls",
        'top_sticker_packs': "Top 10 Sticker Packs",
        'sticker_pack': "Sticker Pack",
        'user_summary_table': "User Statistics Summary Table",
    }
}


# --- END OF CONFIGURATION ---

def add_readability(df):
    """Adds Flesch Reading-Ease score; falls back to NaN if textstat is missing."""
    try:
        import textstat
    except ImportError:
        df['flesch_score'] = np.nan
        return df
    df['flesch_score'] = df['text'].apply(
        lambda t: textstat.flesch_reading_ease(str(t)) if str(t).strip() else np.nan
    )
    return df


def get_sentiment(text, analyser):
    if not text.strip(): return np.nan
    return analyser.polarity_scores(text)['compound']


def add_sentiment(df):
    vader = SentimentIntensityAnalyzer()
    # Add common Ukrainian/Russian sentiment words
    new_words = {
        'круто': 3.0, 'чудово': 3.0, 'прекрасно': 3.0, 'жах': -2.5, 'погано': -2.0,
        'супер': 3.0, 'клас': 2.5, 'добре': 2.0, 'згоден': 1.5,
        'нафіксив': 1.0, 'потужна': 2.0
    }
    vader.lexicon.update(new_words)
    df['sentiment'] = df['text'].apply(lambda t: get_sentiment(str(t), vader))
    return df


def extract_questions(df):
    df['is_question'] = df['text'].str.contains(r'\?+\s*$', regex=True)
    return df


def extract_caps_ratio(df):
    df['caps_ratio'] = df['text'].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1))
    return df


def extract_most_active_hour(df):
    return df.groupby('hour').size().idxmax()


def extract_longest_msg(df):
    return df.loc[df['message_length'].idxmax(), ['author', 'text', 'date', 'message_length']]


def reaction_matrix(df):
    if 'reactions' not in df.columns: return None
    df_reactions = df.dropna(subset=['reactions']).copy()
    if df_reactions.empty: return None

    reactions_list = []
    for _, row in df_reactions.iterrows():
        if isinstance(row['reactions'], list):
            for reaction in row['reactions']:
                if isinstance(reaction, dict) and 'emoji' in reaction and 'count' in reaction:
                    reactions_list.append({'emoji': reaction['emoji'], 'count': reaction['count']})

    if not reactions_list: return None

    reaction_df = pd.DataFrame(reactions_list)
    return reaction_df.groupby('emoji')['count'].sum().sort_values(ascending=False)


def setup_nltk():
    """Download necessary NLTK packages."""
    print("Checking for NLTK packages...")
    packages_to_download = ['punkt', 'stopwords']
    for package in packages_to_download:
        try:
            if package == 'punkt':
                nltk.data.find(f'tokenizers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            print(f"Downloading NLTK package: {package}...")
            nltk.download(package, quiet=True)
    print("NLTK check complete.")


def process_text_field(text_field):
    """Handles the complex 'text' field from the Telegram JSON export."""
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        return ''.join([part.get('text', '') for part in text_field if isinstance(part, dict)])
    return ''


def get_media_type(message):
    """FIXED: Robustly determine the media type of a message."""
    if message.get('photo'): return 'photo'
    if message.get('video_file'): return 'video_file'
    if message.get('voice_message'): return 'voice_message'
    if message.get('video_message'): return 'video_message'  # Round videos
    if message.get('sticker'): return 'sticker'
    if message.get('file'): return 'file'  # Generic file
    if message.get('animated_emoji'): return 'animated_emoji'
    if not pd.isna(message.get('media_type')): return message['media_type']  # Use if exists
    return 'text'


def load_and_prepare_data(json_path):
    """Loads and prepares data from the Telegram JSON export file."""
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {json_path}")
        return None

    # Filter for valid messages only
    messages = [msg for msg in data.get('messages', []) if
                msg.get('type') == 'message' and 'from' in msg and msg.get('from') is not None]
    if not messages:
        print("No valid user messages found in the JSON file.")
        return None

    df = pd.DataFrame(messages)

    # Core data processing
    df['text'] = df['text'].apply(process_text_field)
    df['author'] = df['from'].fillna('Unknown')
    df['date'] = pd.to_datetime(df['date'])
    df['message_length'] = df['text'].str.len().fillna(0)
    df['word_count'] = df['text'].str.split().str.len().fillna(0)

    # Time features
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.date

    # Media Analysis
    df['media_type'] = df.apply(get_media_type, axis=1)
    df['photo_count'] = (df['media_type'] == 'photo').astype(int)
    df['video_count'] = df['media_type'].isin(['video_message', 'video_file']).astype(int)
    df['sticker_count'] = (df['media_type'] == 'sticker').astype(int)
    df['voice_count'] = (df['media_type'] == 'voice_message').astype(int)

    # --- ROBUST STICKER EMOJI HANDLING (NEW FIX) ---
    # The 'sticker_emoji' key is top-level in the JSON, so pandas creates the column automatically.
    # We just need to ensure the column exists for later processing, even if there are no stickers in the chat.
    if 'sticker_emoji' not in df.columns:
        df['sticker_emoji'] = None

    # Reply analysis
    reply_map = df.set_index('id')['author'].to_dict()
    df['reply_to_author'] = df['reply_to_message_id'].map(reply_map)

    print(f"Loaded {len(df)} messages.")
    return df


def analyze_chat(df, language):
    """Performs a comprehensive analysis of the chat data."""
    print("Analyzing data...")
    if df is None or df.empty:
        print("Analysis skipped due to empty data.")
        return {}

    stats = {}

    # --- User & General Stats ---
    stats['total_messages'] = len(df)
    stats['authors'] = df['author'].unique().tolist()
    stats['chat_period'] = f"{df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}"
    daily_counts = df.groupby('day').size()
    stats['most_active_day'] = daily_counts.idxmax()
    stats['most_active_day_count'] = daily_counts.max()
    stats['author_counts'] = df['author'].value_counts()

    # --- Activity Analysis ---
    stats['heatmap_data'] = df.groupby(['hour', 'weekday']).size().unstack().fillna(0)
    stats['heatmap_data'] = stats['heatmap_data'][
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

    # --- Content & Vocabulary (FIXED FOR MULTILINGUAL) ---
    emojis = sum(df['text'].apply(lambda t: [c for c in t if c in emoji.EMOJI_DATA]), [])
    stats['emoji_counts'] = Counter(emojis).most_common(20)

    # Combine stopwords for multiple languages for better filtering
    stop_words = set()
    try:
        stop_words.update(nltk.corpus.stopwords.words('english'))
    except Exception:
        print("Could not load English stopwords.")
    try:
        stop_words.update(nltk.corpus.stopwords.words('russian'))
    except Exception:
        print("Could not load Russian stopwords.")

    # Add custom Ukrainian stopwords since NLTK lacks them
    ukrainian_stop_words = set("""
    а аж також без би бо була був були було бути в вам ваш ваше вашого вашій вашою вашу ваші
    вас вже від він вона воно вони всі всього всіх втім да для до дуже є уже й ж за зі з як якби якщо їй її їм їх і
    до із та не про по це ці цім ціх цих я на він ми ти ви ще що тут там
    """.split())
    stop_words.update(ukrainian_stop_words)
    stop_words.update(['та', 'шо', 'це', 'не', 'я', 'в', 'і', 'ти', 'то', 'а', 'ну', 'на', 'по', 'за', 'же'])

    all_text = ' '.join(df['text'])
    # Tokenize without specifying a language to handle mixed content
    words = [word for word in nltk.word_tokenize(all_text.lower()) if
             word.isalpha() and word not in stop_words and len(word) > 2]

    if words:
        stats['top_words'] = Counter(words).most_common(100)
        stats['top_bigrams'] = []
        try:
            vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=list(stop_words), min_df=3)
            bigram_counts = vectorizer.fit_transform(df['text'])
            bigram_sum = bigram_counts.sum(axis=0).A1
            bigram_vocab = vectorizer.get_feature_names_out()
            top_bigrams_list = sorted(zip(bigram_vocab, bigram_sum), key=lambda x: -x[1])[:20]
            # Filter out bigrams with numbers
            stats['top_bigrams'] = [item for item in top_bigrams_list if not any(char.isdigit() for char in item[0])]
        except Exception as e:
            print(f"Could not generate bigrams: {e}")
    else:
        stats['top_words'] = []
        stats['top_bigrams'] = []
        print("WARNING: Word list is empty after filtering. Word cloud and bigrams will be empty.")

    # --- Lexical Diversity (TTR) - FIXED ---
    try:
        def calculate_ttr(text):
            # Use language-agnostic tokenization
            tokens = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha()]
            if not tokens: return 0
            return len(set(tokens)) / len(tokens)

        user_text = df.groupby('author')['text'].apply(' '.join)
        stats['lexical_diversity'] = user_text.apply(calculate_ttr).sort_values(ascending=False)
    except Exception as e:
        print(f"WARNING: Could not calculate lexical diversity. Details: {e}")
        stats['lexical_diversity'] = pd.Series()

    # --- Night Activity ---
    night_messages = df[df['hour'].between(0, 5)]
    stats['night_activity'] = night_messages['author'].value_counts()

    # --- Media Analysis ---
    stats['media_distribution'] = df['media_type'].value_counts()

    # --- Sticker Analysis ---
    valid_stickers = df.dropna(subset=['sticker_emoji'])
    if not valid_stickers.empty:
        stats['top_sticker_packs'] = valid_stickers.groupby('sticker_emoji').size().sort_values(ascending=False).head(
            10)
    else:
        stats['top_sticker_packs'] = pd.Series()

    # --- Interaction Analysis ---
    stats['reply_matrix'] = df.groupby(['author', 'reply_to_author']).size().unstack().fillna(0)

    # --- Domain Analysis ---
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    all_links = re.findall(url_pattern, ' '.join(df['text']))
    domains = [urlparse(link).netloc for link in all_links if urlparse(link).netloc]
    stats['top_domains'] = Counter(domains).most_common(15)

    # --- Advanced Metrics ---
    df = add_sentiment(df)
    df = add_readability(df)
    df = extract_questions(df)
    stats['longest_msg'] = extract_longest_msg(df)
    starters = df[df['reply_to_message_id'].isna()].groupby('author').size()
    stats['conversation_starters'] = starters.sort_values(ascending=False)

    # --- Summary Table (FIXED) ---
    summary = df.groupby('author').agg(
        messages=('id', 'count'),
        words=('word_count', 'sum'),
        avg_len=('message_length', 'mean'),
        photos=('photo_count', 'sum'),
        videos=('video_count', 'sum'),
        stickers=('sticker_count', 'sum')
    ).round(1)
    # Convert media counts to integers
    for col in ['photos', 'videos', 'stickers']:
        summary[col] = summary[col].astype(int)
    stats['user_summary_table'] = summary.sort_values('messages', ascending=False)

    print("Analysis complete.")
    return stats


def create_visualizations(stats, output_dir, font_path, loc):
    """Creates and saves all visualizations with consistent font handling."""
    print("Creating visualizations...")
    if not stats:
        print("Skipping visualizations due to no data.")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    # --- FONT SETUP (FIXED) ---
    # Set a default font for all plots to handle special characters and emoji
    try:
        font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f"Successfully set plotting font to: {font_prop.get_name()}")
    except Exception as e:
        print(f"WARNING: Could not set font from '{font_path}'. Using default. Error: {e}")
        print("Plots may have missing characters (emojis, etc.).")

    # Plot 1: Messages per Author
    plt.figure(figsize=(12, 8))
    sns.barplot(y=stats['author_counts'].index, x=stats['author_counts'].values)
    plt.title(loc['total_messages_per_author'], fontsize=16)
    plt.xlabel(loc['message_count'])
    plt.ylabel(loc['author'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "messages_per_author.png"))
    plt.close()

    # Plot 2: Activity Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(stats['heatmap_data'], cmap="YlGnBu", linewidths=.5, annot=True, fmt=".0f")
    plt.title(loc['activity_heatmap'], fontsize=16)
    plt.xlabel(loc['day_of_week'])
    plt.ylabel(loc['hour_of_day'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "activity_heatmap.png"))
    plt.close()

    # Plot 3: Word Cloud (FIXED)
    if stats.get('top_words'):
        try:
            wordcloud = WordCloud(width=1200, height=600, background_color='white', font_path=font_path,
                                  colormap='magma', max_words=100).generate_from_frequencies(dict(stats['top_words']))
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(loc['word_cloud'], fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "wordcloud.png"))
            plt.close()
        except Exception as e:
            print(f"ERROR: Could not create word cloud. Check font path: {font_path}. Details: {e}")

    # Plot 4: Top Emojis
    if stats.get('emoji_counts'):
        emoji_df = pd.DataFrame(stats['emoji_counts'], columns=['emoji', 'count'])
        plt.figure(figsize=(12, 8))
        sns.barplot(y='emoji', x='count', data=emoji_df)
        plt.title(loc['top_emojis'], fontsize=16)
        plt.xlabel(loc['count'])
        plt.ylabel(loc['emoji'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "top_emojis.png"))
        plt.close()

    # Plot 5: Media Distribution
    if stats.get('media_distribution') is not None and not stats['media_distribution'].empty:
        plt.figure(figsize=(10, 10))
        stats['media_distribution'].plot(kind='pie', autopct='%1.1f%%', startangle=140, colormap='tab20c')
        plt.title(loc['media_type_distribution'], fontsize=16)
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "media_distribution.png"))
        plt.close()

    # Plot 6: Reply Matrix Heatmap
    if 'reply_matrix' in stats and not stats['reply_matrix'].empty:
        plt.figure(figsize=(14, 10))
        sns.heatmap(stats['reply_matrix'], cmap="BuPu", annot=True, fmt=".0f", linewidths=.5)
        plt.title(loc['reply_heatmap'], fontsize=16)
        plt.xlabel(loc['replied_to'])
        plt.ylabel(loc['replier'])
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "reply_heatmap.png"))
        plt.close()

    # Plot 7: Lexical Diversity
    if 'lexical_diversity' in stats and not stats['lexical_diversity'].empty:
        plt.figure(figsize=(12, 8))
        stats['lexical_diversity'].sort_values().plot(kind='barh')
        plt.title(loc['lexical_diversity'], fontsize=16)
        plt.xlabel("TTR (Unique Words / Total Words)")
        plt.ylabel(loc['author'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "lexical_diversity.png"))
        plt.close()

    # Plot 8: Night Owls
    if 'night_activity' in stats and not stats['night_activity'].empty:
        plt.figure(figsize=(12, 8))
        stats['night_activity'].head(15).sort_values().plot(kind='barh')
        plt.title(loc['night_owls'], fontsize=16)
        plt.xlabel(loc['message_count'])
        plt.ylabel(loc['author'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "night_activity.png"))
        plt.close()

    # Plot 9: Top Bigrams (FIXED)
    if stats.get('top_bigrams'):
        bigram_df = pd.DataFrame(stats['top_bigrams'], columns=['phrase', 'count'])
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y='phrase', data=bigram_df, palette='mako')
        plt.title("Top 20 Common Phrases (Bigrams)", fontsize=16)
        plt.xlabel(loc['count'])
        plt.ylabel("Phrase")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "bigrams.png"))
        plt.close()

    print("Visualizations saved.")


def generate_html_report(stats, output_dir, loc):
    """Generates an HTML report."""
    print("Creating HTML report...")
    if not stats: return None

    def series_to_html_table(data, headers):
        if data is None or data.empty: return f"<p>No data available.</p>"
        html_content = '<table class="data-table"><tr>'
        for header in headers: html_content += f"<th>{header}</th>"
        html_content += "</tr>"
        # Check if data is a dictionary or a Series
        items = data.items() if isinstance(data, (dict, pd.Series)) else data
        for index, value in items:
            html_content += f"<tr><td>{index}</td><td>{value}</td></tr>"
        return html_content + "</table>"

    def df_to_html_table(df):
        if df is None or df.empty: return f"<p>No data available.</p>"
        return df.to_html(classes='data-table', index=True)

    # HTML content building
    html_template = f"""
    <html><head><meta charset="UTF-8"><title>{loc['report_title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Roboto, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }}
        h1 {{ text-align: center; font-size: 2.5em; }}
        h2 {{ font-size: 1.8em; margin-top: 40px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
        .card {{ background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; }}
        .card h3 {{ font-size: 1.2em; margin-top: 0; color: #333; border: none; }}
        .card p {{ font-size: 1.5em; font-weight: bold; color: #0056b3; margin: 10px 0; }}
        .full-width {{ grid-column: 1 / -1; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin-top: 15px; }}
        table.data-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }}
        .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .data-table th {{ background-color: #0056b3; color: white; }}
        .data-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        footer {{ text-align: center; margin-top: 40px; color: #777; font-size: 0.9em; }}
    </style>
    </head><body><div class="container">
        <h1>{loc['report_title']}</h1>
        <h2>{loc['general_stats']}</h2>
        <div class="grid">
            <div class="card"><h3>{loc['total_messages']}</h3><p>{stats.get('total_messages', 0):,}</p></div>
            <div class="card"><h3>{loc['total_users']}</h3><p>{len(stats.get('authors', []))}</p></div>
            <div class="card"><h3>{loc['chat_period']}</h3><p style="font-size: 1.1em;">{stats.get('chat_period', 'N/A')}</p></div>
            <div class="card"><h3>{loc['most_active_day']}</h3><p style="font-size: 1.1em;">{stats.get('most_active_day', 'N/A')} ({stats.get('most_active_day_count', 0)} {loc['messages_abbr']})</p></div>
        </div>

        <h2>{loc['user_summary_table']}</h2>
        <div class="card full-width">{df_to_html_table(stats.get('user_summary_table'))}</div>

        <h2>{loc['user_activity']}</h2>
        <div class="card full-width"><h3>{loc['messages_per_author']}</h3><img src="plots/messages_per_author.png"></div>
        <div class="grid">
            <div class="card"><h3>{loc['night_owls']}</h3><img src="plots/night_activity.png"></div>
            <div class="card"><h3>{loc['lexical_diversity']}</h3><img src="plots/lexical_diversity.png"></div>
        </div>

        <h2>{loc['interaction_analysis']}</h2>
        <div class="card full-width"><h3>{loc['reply_heatmap']}</h3><img src="plots/reply_heatmap.png"></div>

        <h2>{loc['time_analysis']}</h2>
        <div class="card full-width"><h3>{loc['activity_heatmap']}</h3><img src="plots/activity_heatmap.png"></div>

        <h2>{loc['content_analysis']}</h2>
        <div class="card full-width"><h3>{loc['word_cloud']}</h3><img src="plots/wordcloud.png"></div>
        <div class="card full-width"><h3>Top Phrases</h3><img src="plots/bigrams.png"></div>

        <div class="grid">
            <div class="card"><h3>{loc['top_emojis']}</h3><img src="plots/top_emojis.png"></div>
            <div class="card"><h3>{loc['top_sticker_packs']}</h3>{series_to_html_table(stats.get('top_sticker_packs'), [loc['sticker_pack'], loc['count']])}</div>
        </div>
        <div class="card full-width"><h3>{loc['top_domains']}</h3>{series_to_html_table(stats.get('top_domains', []), [loc['domain'], loc['count']])}</div>

        <h2>{loc['media_analysis']}</h2>
        <div class="card full-width"><h3>{loc['media_type_distribution']}</h3><img src="plots/media_distribution.png"></div>

        <footer><p>{loc['generated_at']} {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p></footer>
    </div></body></html>
    """
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html_template)
    print(f"HTML report saved to: {report_path}")
    return report_path


def generate_pdf_report(html_path, output_dir, wkhtmltopdf_path):
    """Generates a PDF report from the HTML file."""
    if not html_path: return
    print("Attempting to generate PDF report...")
    try:
        import pdfkit
    except ImportError:
        print("ERROR: 'pdfkit' library not installed. PDF report will not be created. Install with: pip install pdfkit")
        return

    pdf_path = os.path.join(output_dir, "report.pdf")
    try:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        options = {"enable-local-file-access": True, "encoding": "UTF-8", "page-size": "A4", "margin-top": "0.75in",
                   "margin-right": "0.75in", "margin-bottom": "0.75in", "margin-left": "0.75in"}
        pdfkit.from_file(html_path, pdf_path, configuration=config, options=options)
        print(f"PDF report successfully saved to: {pdf_path}")
    except FileNotFoundError:
        print(f"ERROR: 'wkhtmltopdf' executable not found. Check path in CONFIG: '{wkhtmltopdf_path}'")
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")


def main():
    """Main function to run the analysis."""
    os.makedirs(CONFIG['output_folder'], exist_ok=True)
    setup_nltk()

    loc = UI_STRINGS.get(CONFIG['report_language'], UI_STRINGS['en'])
    json_path = os.path.join(CONFIG['export_folder'], CONFIG['json_file_name'])

    df = load_and_prepare_data(json_path)
    if df is None:
        print("Exiting due to data loading failure.")
        return

    chat_stats = analyze_chat(df, CONFIG['language'])
    create_visualizations(chat_stats, CONFIG['output_folder'], CONFIG['font_path'], loc)
    html_report_path = generate_html_report(chat_stats, CONFIG['output_folder'], loc)

    if CONFIG['generate_pdf']:
        generate_pdf_report(html_report_path, CONFIG['output_folder'], CONFIG['wkhtmltopdf_path'])

    print(f"\n✅ Done! All files are in the folder: {os.path.abspath(CONFIG['output_folder'])}")


if __name__ == "__main__":
    main()