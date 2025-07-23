# -*- coding: utf-8 -*-
import json
import os
import re
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdfkit
import seaborn as sns
from wordcloud import WordCloud
import emoji
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from datetime import datetime
import matplotlib.font_manager as fm  # Import font_manager

# --- SCRIPT CONFIGURATION ---
CONFIG = {
    # 1. Path to the chat export folder
    "export_folder": r"C:\Users",
    # 2. Name of the JSON file
    "json_file_name": "result.json",
    # 3. Folder to save the report
    "output_folder": "telegram_report_advanced_final",
    # 4. Main language for stop-word analysis ('russian', 'ukrainian', 'english')
    #    The script will handle mixed languages, but this helps prioritize stopwords.
    "language": "english",  # Primary language for general processing (e.g., sentiment lexicon)
    # 5. Report language ('uk', 'en')
    "report_language": "en",
    # 6. Path to a font file that supports Cyrillic and emojis (e.g., Segoe UI Emoji, Noto Color Emoji)
    "emoji_font_path": r'fonts/seguiemj.ttf',  # For charts with emojis
    # 7. Path to a font file for text-only charts
    "text_font_path": r'fonts/segoeui.ttf',  # For text-only charts
    # 8. PDF Generation Settings
    "generate_pdf": True,
    "wkhtmltopdf_path": r"C:\Apps\wkhtmltopdf\bin\wkhtmltopdf.exe"  # <-- IMPORTANT: Update if needed
}

# --- LOCALIZATION STRINGS ---
UI_STRINGS = {
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
        'voice_messages': 'Voice Messages',
        'videos': 'Videos',
        'photos': 'Photos',
        'stickers': 'Stickers',
        'files': 'Files',
        'animated_emojis': 'Animated Emojis',
        'links': 'Links',
        'questions': 'Questions',
        'text_messages': 'Text Messages'
    }
}

# Set Matplotlib font globally for text-only plots
# Using FontProperties with full path for better reliability
matplotlib.rcParams['font.sans-serif'] = [fm.FontProperties(fname=CONFIG['text_font_path']).get_name()]
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix for minus sign in plots


def add_readability(df):
    """Adds Flesch Reading-Ease score; falls back to NaN if textstat is missing."""
    try:
        import textstat
    except ImportError:
        print("Warning: 'textstat' library not found. Flesch Reading-Ease score will not be calculated.")
        df['flesch_score'] = np.nan
        return df
    df['flesch_score'] = df['text'].apply(
        lambda t: textstat.flesch_reading_ease(str(t)) if str(t).strip() else np.nan
    )
    return df


def get_sentiment(text, analyser):
    """Calculates sentiment score for a given text."""
    if not text.strip(): return np.nan
    return analyser.polarity_scores(text)['compound']


def add_sentiment(df):
    """Adds sentiment scores to the DataFrame using VADER."""
    vader = SentimentIntensityAnalyzer()
    # Removed Ukrainian/Russian sentiment words as per request for English-only report UI
    # The default VADER lexicon is sufficient for English.
    df['sentiment'] = df['text'].apply(lambda t: get_sentiment(str(t), vader))
    return df


def extract_questions(df):
    """Identifies messages that are likely questions."""
    # Updated regex to look for English, Russian, and Ukrainian question marks
    # Russian/Ukrainian question mark is the same as English '?'
    # The regex already correctly handles it.
    df['is_question'] = df['text'].str.contains(r'\?\s*$', regex=True, na=False)
    return df


def extract_caps_ratio(df):
    """Calculates the ratio of uppercase characters in a message."""
    df['caps_ratio'] = df['text'].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1) if isinstance(t, str) else 0.0)
    return df


def extract_most_active_hour(df):
    """Finds the hour with the most messages."""
    if df.empty: return np.nan
    return df.groupby('hour').size().idxmax()


def extract_longest_msg(df):
    """Finds the longest message and its author, date."""
    if df.empty or 'message_length' not in df.columns or df['message_length'].max() == 0:
        return {'author': np.nan, 'text': np.nan, 'date': np.nan, 'message_length': 0}
    longest_msg = df.loc[df['message_length'].idxmax(), ['author', 'text', 'date', 'message_length']]
    return longest_msg.to_dict()


def reaction_matrix(df):
    """Processes reactions data to count total reactions per emoji."""
    if 'reactions' not in df.columns:
        return None
    df_reactions = df.dropna(subset=['reactions']).copy()
    if df_reactions.empty:
        return None
    reactions_list = []
    for _, row in df_reactions.iterrows():
        if isinstance(row['reactions'], list):
            for reaction in row['reactions']:
                if isinstance(reaction, dict) and 'emoji' in reaction and 'count' in reaction:
                    reactions_list.append({'emoji': reaction['emoji'], 'count': reaction['count']})
    if not reactions_list:
        return None
    reaction_df = pd.DataFrame(reactions_list)
    return reaction_df.groupby('emoji')['count'].sum().sort_values(ascending=False)


def setup_nltk():
    """Download necessary NLTK packages for English, Russian, and Ukrainian."""
    print("Checking for NLTK packages...")
    packages_to_download = ['punkt']
    for package in packages_to_download:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading NLTK package: {package}...")
            nltk.download(package, quiet=True)

    # Download stopwords for all relevant languages
    for lang in ['english', 'russian', 'ukrainian']:
        try:
            nltk.data.find(f'corpora/stopwords')  # Check if the directory exists
            # This will raise LookupError if the specific language corpus is missing.
            nltk.data.find(f'corpora/stopwords/{lang}')
        except LookupError:
            print(f"Downloading NLTK stopwords for {lang}...")
            try:
                nltk.download('stopwords', quiet=True)  # This usually downloads all common languages
            except Exception as e:
                print(f"Failed to download stopwords for {lang}: {e}. Please ensure you have internet access.")

    print("NLTK check complete.")


def process_text_field(text_field):
    """Handles the complex 'text' field from the Telegram JSON export."""
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        # Concatenate text from different parts, including 'link' and 'pre' types
        full_text = []
        for part in text_field:
            if isinstance(part, dict) and 'text' in part:
                full_text.append(str(part['text']))
            elif isinstance(part, str):
                full_text.append(part)
        return ''.join(full_text)
    return ''


def clean_author_name(name):
    """Removes unsupported characters from author names for plotting and consistency."""
    if not isinstance(name, str):
        return "Unknown"
    # Allow letters, numbers, spaces, periods, and hyphens. Remove other symbols.
    return re.sub(r'[^\w\s.\-]', '', name).strip()


def get_media_type(message):
    """Determines the media type of a message based on JSON structure, ignoring placeholder strings."""

    # 1. Prioritize text content and links
    processed_msg_text = process_text_field(message.get('text', ''))
    if processed_msg_text:
        if re.search(r'https?://\S+|www\.\S+', processed_msg_text):
            return 'link'
        return 'text_message'

    # 2. Check for explicit 'media_type' field first, as it's often the most direct indicator
    media_type = message.get('media_type')
    if media_type in ['photo', 'video_file', 'video_message', 'voice_message', 'sticker', 'animation', 'audio_file',
                      'file']:
        # If media_type is explicitly one of these, return it.
        # We assume its presence means it was that type of message, even if content not exported.
        return media_type

    # 3. Fallback to checking for specific media keys if 'media_type' wasn't precise or present.
    #    Order matters here: more specific media keys first.
    if 'photo' in message:
        return 'photo'
    # 'video_file', 'voice_message', 'sticker', 'animation', 'audio_file' typically rely on 'media_type'
    # or the 'file' key, which is handled below.

    if 'file' in message:
        # This covers generic 'file' types and potentially others where 'file' holds the content
        # but 'media_type' wasn't specific.
        return 'file'

    # 4. Other media types that don't typically involve 'file' or 'photo' keys with placeholders
    if message.get('location'): return 'location'
    if message.get('contact_information'): return 'contact'
    if message.get('game'): return 'game'
    if message.get('live_location'): return 'live_location'
    if message.get('invoice'): return 'invoice'
    if message.get('poll'): return 'poll'
    if message.get('story'): return 'story'

    return 'unknown'  # Fallback if no specific type or text is found


def load_and_prepare_data(json_path):
    """Loads and prepares data from the Telegram JSON export file."""
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {json_path}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from {json_path}: {e}")
        return None, None

    messages = [msg for msg in data.get('messages', []) if msg.get('type') == 'message' and 'from' in msg]
    service_messages = [msg for msg in data.get('messages', []) if msg.get('type') == 'service']

    if not messages:
        print("No messages found in the JSON file.")
        return None, None

    df = pd.DataFrame(messages)

    # Core data processing
    df['text'] = df['text'].apply(process_text_field)
    df['author'] = df['from'].fillna('Unknown')
    df['date'] = pd.to_datetime(df['date'])
    df['message_length'] = df['text'].apply(len)
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['day_of_week_num'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

    # Add derived features
    df = add_readability(df)
    df = add_sentiment(df)
    df = extract_questions(df)
    df = extract_caps_ratio(df)

    # Add media type
    df['media_type'] = df.apply(get_media_type, axis=1)

    # Clean author names for consistency
    df['author'] = df['author'].apply(clean_author_name)

    return df, service_messages


def get_stopwords(language_code):
    """Fetches stopwords for the specified language code (e.g., 'en', 'ru', 'uk')."""
    try:
        if language_code == 'en':
            return set(nltk.corpus.stopwords.words('english'))
        elif language_code == 'ru':
            return set(nltk.corpus.stopwords.words('russian'))
        elif language_code == 'uk':
            # NLTK might not have 'ukrainian' directly. Use a common alternative or check.
            # As per NLTK documentation, 'russian' stopwords are often used for general Cyrillic.
            # However, if 'ukrainian' is available, use it.
            # For now, let's assume 'ukrainian' exists or fall back.
            try:
                return set(nltk.corpus.stopwords.words('ukrainian'))
            except OSError:
                print("Ukrainian stopwords not found, falling back to Russian for Ukrainian text processing.")
                return set(nltk.corpus.stopwords.words('russian'))
        return set()  # Fallback for unsupported languages
    except Exception as e:
        print(f"Could not load NLTK stopwords for '{language_code}': {e}. Using empty set.")
        return set()


def detect_language_simple(text):
    """Simple heuristic to detect language (en, ru, uk) based on character ranges."""
    text = str(text).lower()
    # Check for Cyrillic characters (Russian/Ukrainian)
    if re.search(r'[\u0400-\u04FF]', text):
        # A very basic distinction between Russian and Ukrainian based on common letters
        ukrainian_letters = re.findall(r'[\u0404\u0407\u0454\u0457\u0456\u045E]', text)  # Є, І, Ї, Ґ
        russian_letters = re.findall(
            r'[\u042A\u042B\u0429\u042D\u0424\u0428\u042C\u042F\u0439\u044A\u044B\u0449\u044D\u0444\u0448\u044C\u044F]',
            text)  # ъ, ы, щ, э, ф, ш, ь, я
        if len(ukrainian_letters) > len(russian_letters):  # Simple count for distinction
            return 'uk'
        return 'ru'  # Default to Russian for Cyrillic if not clearly Ukrainian
    # Check for Latin characters (English)
    elif re.search(r'[a-zA-Z]', text):
        return 'en'
    return 'en'  # Default to English if no strong indicator


def clean_text_for_wordcloud(text):
    """Cleans text for word cloud generation, removing stopwords and links, supporting multiple languages."""
    text = str(text).lower()
    # Remove URLs/links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emojis
    text = emoji.demojize(text)
    # Remove non-alphanumeric characters and extra spaces, but keep Cyrillic
    text = re.sub(r'[^\w\s\u0400-\u04FF]', '', text)

    # Determine language for stopword removal
    lang = detect_language_simple(text)
    stopwords = get_stopwords(lang)

    tokens = nltk.word_tokenize(text)
    # Filter out stopwords and single-character tokens
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]
    return ' '.join(tokens)


def generate_word_cloud(df, output_folder, current_lang):
    """Generates and saves a word cloud from message texts."""
    all_text = ' '.join(df['text'].dropna().tolist())
    cleaned_text = clean_text_for_wordcloud(all_text)  # Language detection happens inside clean_text_for_wordcloud

    if not cleaned_text:
        print("Not enough text data to generate a word cloud.")
        return

    wordcloud = WordCloud(
        width=1200, height=800,
        background_color='white',
        collocations=False,  # Avoids showing "new york" as one term if not desired
        font_path=CONFIG['emoji_font_path']  # Use emoji font for word cloud
    ).generate(cleaned_text)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(UI_STRINGS[current_lang]['word_cloud'])
    plt.savefig(os.path.join(output_folder, 'word_cloud.png'))
    plt.close()


def generate_bigrams(df, output_folder, current_lang, top_n=20):
    """Generates and saves a bar chart of the most common bigrams."""
    all_text = ' '.join(df['text'].dropna().tolist())
    # Remove URLs/links from text before generating bigrams
    all_text = re.sub(r'https?://\S+|www\.\S+', '', all_text)

    # Determine language for stopword removal
    lang = detect_language_simple(all_text)
    stopwords = get_stopwords(lang)

    tokens = nltk.word_tokenize(all_text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stopwords]

    # Generate bigrams
    bigram_counts = Counter(nltk.bigrams(filtered_tokens))
    top_bigrams = bigram_counts.most_common(top_n)

    if not top_bigrams:
        print("Not enough data to generate bigrams.")
        return

    bigram_labels = [" ".join(bigram) for bigram, count in top_bigrams]
    bigram_values = [count for bigram, count in top_bigrams]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=bigram_values, y=bigram_labels, palette='viridis')
    plt.title(f"{UI_STRINGS[current_lang]['content_analysis']} - Top {top_n} Bigrams")
    plt.xlabel(UI_STRINGS[current_lang]['count'])
    plt.ylabel("Bigram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bigrams_chart.png'))
    plt.close()


def generate_messages_per_author_chart(df, output_folder, current_lang):
    """Generates and saves a bar chart of messages per author."""
    messages_per_author = df['author'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, max(6, len(messages_per_author) * 0.5)))
    sns.barplot(x=messages_per_author.values, y=messages_per_author.index, palette='crest')
    plt.title(UI_STRINGS[current_lang]['messages_per_author'])
    plt.xlabel(UI_STRINGS[current_lang]['message_count'])
    plt.ylabel(UI_STRINGS[current_lang]['author'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'messages_per_author.png'))
    plt.close()


def generate_activity_heatmap(df, output_folder, current_lang):
    """Generates and saves an activity heatmap."""
    # Define the order of days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Ensure 'day_of_week' is a categorical type with the correct order
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

    activity_pivot = df.pivot_table(index='hour', columns='day_of_week', aggfunc='size', fill_value=0)
    # Reindex columns to ensure correct order, filling missing days with zeros if necessary
    activity_pivot = activity_pivot.reindex(columns=day_order, fill_value=0)

    plt.figure(figsize=(14, 8))
    sns.heatmap(activity_pivot, cmap='viridis', annot=True, fmt='d', linewidths=.5)
    plt.title(UI_STRINGS[current_lang]['activity_heatmap'])
    plt.xlabel(UI_STRINGS[current_lang]['day_of_week'])
    plt.ylabel(UI_STRINGS[current_lang]['hour_of_day'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'activity_heatmap.png'))
    plt.close()


def generate_top_emojis_chart(df, output_folder, current_lang, top_n=20):
    """Generates and saves a bar chart of the most used emojis."""
    all_emojis = []
    for text in df['text'].dropna():
        all_emojis.extend([c for c in text if c in emoji.EMOJI_DATA])

    emoji_counts = Counter(all_emojis)
    top_emojis = emoji_counts.most_common(top_n)

    if not top_emojis:
        print("No emojis found to generate a chart.")
        return

    emojis = [e[0] for e in top_emojis]
    counts = [e[1] for e in top_emojis]

    plt.figure(figsize=(12, max(6, len(emojis) * 0.5)))
    # Use emoji font for emoji chart
    plt.bar(emojis, counts, color=sns.color_palette("flare", len(emojis)))
    plt.title(UI_STRINGS[current_lang]['top_emojis'])
    plt.xlabel(UI_STRINGS[current_lang]['emoji'])
    plt.ylabel(UI_STRINGS[current_lang]['count'])
    # Set the font for the emoji labels explicitly using FontProperties
    font_prop_emoji = fm.FontProperties(fname=CONFIG['emoji_font_path'])
    for label in plt.gca().get_xticklabels():
        label.set_fontproperties(font_prop_emoji)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top_emojis.png'))
    plt.close()


def generate_top_domains_chart(df, output_folder, current_lang, top_n=15):
    """Generates and saves a bar chart of the most frequent domains from links."""
    domain_counts = Counter()
    for text in df['text'].dropna():
        urls = re.findall(r'https?://\S+|www\.\S+', text)
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if domain:
                    domain_counts[domain] += 1
            except Exception:
                continue

    top_domains = domain_counts.most_common(top_n)

    if not top_domains:
        print("No links found to generate a domains chart.")
        return

    domains = [d[0] for d in top_domains]
    counts = [d[1] for d in top_domains]

    plt.figure(figsize=(12, max(6, len(domains) * 0.5)))
    sns.barplot(x=counts, y=domains, palette='rocket')
    plt.title(UI_STRINGS[current_lang]['top_domains'])
    plt.xlabel(UI_STRINGS[current_lang]['count'])
    plt.ylabel(UI_STRINGS[current_lang]['domain'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top_domains.png'))
    plt.close()


def generate_media_type_distribution_chart(df, output_folder, current_lang):
    """Generates and saves a pie chart of message type distribution."""
    media_counts = df['media_type'].value_counts()

    # Map internal media_type names to localized UI strings
    localized_labels = [UI_STRINGS[current_lang].get(mt, mt.replace('_', ' ').title()) for mt in media_counts.index]

    plt.figure(figsize=(10, 10))
    plt.pie(media_counts, labels=localized_labels, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("pastel"))
    plt.title(UI_STRINGS[current_lang]['media_type_distribution'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'media_type_distribution.png'))
    plt.close()


def generate_reply_heatmap(df, output_folder, current_lang, top_n_authors=15):
    """Generates and saves a heatmap of who replies to whom."""
    reply_df = df[df['reply_to_message_id'].notna()].copy()

    if reply_df.empty:
        print("No reply data to generate heatmap.")
        return

    # Merge to get the author of the replied-to message
    replied_messages = df[['id', 'author']].rename(columns={'id': 'reply_to_message_id', 'author': 'replied_to_author'})
    merged_df = pd.merge(reply_df, replied_messages, on='reply_to_message_id', how='left')

    # Drop replies to non-existent messages or self-replies
    merged_df.dropna(subset=['replied_to_author'], inplace=True)
    merged_df = merged_df[merged_df['author'] != merged_df['replied_to_author']]

    if merged_df.empty:
        print("No valid reply interactions to generate heatmap.")
        return

    # Count replies
    reply_counts = merged_df.groupby(['author', 'replied_to_author']).size().unstack(fill_value=0)

    # Filter for top N active repliers and replied-to authors
    # Ensure there are enough unique authors to filter by
    if len(merged_df['author'].unique()) > top_n_authors:
        active_repliers = merged_df['author'].value_counts().nlargest(top_n_authors).index
    else:
        active_repliers = merged_df['author'].unique()

    if len(merged_df['replied_to_author'].unique()) > top_n_authors:
        active_replied_to = merged_df['replied_to_author'].value_counts().nlargest(top_n_authors).index
    else:
        active_replied_to = merged_df['replied_to_author'].unique()

    filtered_reply_counts = reply_counts.loc[
        reply_counts.index.intersection(active_repliers),
        reply_counts.columns.intersection(active_replied_to)
    ]

    if filtered_reply_counts.empty:
        print("Not enough relevant reply interactions after filtering for top authors.")
        return

    plt.figure(figsize=(min(20, max(8, len(filtered_reply_counts.columns))),
                        min(20, max(8, len(filtered_reply_counts.index)))))
    sns.heatmap(filtered_reply_counts, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5, linecolor='black')
    plt.title(UI_STRINGS[current_lang]['reply_heatmap'])
    plt.xlabel(UI_STRINGS[current_lang]['replied_to'])
    plt.ylabel(UI_STRINGS[current_lang]['replier'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'reply_heatmap.png'))
    plt.close()


def generate_night_owls_chart(df, output_folder, current_lang):
    """Generates and saves a bar chart of 'night owl' activity (00:00-06:00)."""
    night_activity = df[(df['hour'] >= 0) & (df['hour'] < 6)]
    night_owls = night_activity['author'].value_counts().nlargest(10)

    if night_owls.empty:
        print("No night activity found to generate a chart.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=night_owls.values, y=night_owls.index, palette='magma')
    plt.title(UI_STRINGS[current_lang]['night_activity'] + " - " + UI_STRINGS[current_lang]['night_owls'])
    plt.xlabel(UI_STRINGS[current_lang]['message_count'])
    plt.ylabel(UI_STRINGS[current_lang]['author'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'night_owls.png'))
    plt.close()


def generate_top_sticker_packs_chart(df, output_folder, current_lang, top_n=10):
    """Generates and saves a bar chart of top sticker packs."""
    sticker_messages = df[df['media_type'] == 'sticker']
    sticker_pack_counts = Counter()

    for _, row in sticker_messages.iterrows():
        # Check if 'file' field is a dictionary and contains 'file_name'
        if isinstance(row.get('file'), dict) and row['file'].get('file_name'):
            file_name = row['file']['file_name']

            # Attempt to extract from path like "stickers/PACK_NAME/sticker.webp"
            match = re.search(r'stickers/([^/]+)/', file_name)
            if match:
                sticker_pack_counts[match.group(1)] += 1
            else:
                # Fallback: if 'file_name' itself is descriptive or just "sticker.webp"
                # Sometimes the 'file_name' might just be "sticker.webp" if not in a subfolder.
                # In such cases, if no clear pack name, categorize as 'Unknown Pack' or use sticker_emoji
                if row.get('sticker_emoji'):
                    sticker_pack_counts[f"Emoji: {row['sticker_emoji']}"] += 1
                else:
                    sticker_pack_counts['Unknown Pack'] += 1
        elif row.get('sticker_emoji'):
            sticker_pack_counts[f"Emoji: {row['sticker_emoji']}"] += 1
        else:
            sticker_pack_counts['Unknown Pack'] += 1

    top_sticker_packs = sticker_pack_counts.most_common(top_n)

    if not top_sticker_packs:
        print("No sticker data or recognizable sticker packs found.")
        return

    packs = [p[0] for p in top_sticker_packs]
    counts = [p[1] for p in top_sticker_packs]

    plt.figure(figsize=(10, max(6, len(packs) * 0.5)))
    sns.barplot(x=counts, y=packs, palette='cubehelix')
    plt.title(UI_STRINGS[current_lang]['top_sticker_packs'])
    plt.xlabel(UI_STRINGS[current_lang]['count'])
    plt.ylabel(UI_STRINGS[current_lang]['sticker_pack'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top_sticker_packs.png'))
    plt.close()


def get_message_counts(df, current_lang):
    """Calculates counts for different message categories, including voice messages and links."""
    total_messages = len(df)
    total_text_messages = df[df['media_type'] == 'text_message'].shape[0]
    total_photo_messages = df[df['media_type'] == 'photo'].shape[0]
    total_video_messages = df[df['media_type'].isin(['video_file', 'video_message'])].shape[0]
    total_voice_messages = df[df['media_type'] == 'voice_message'].shape[0]
    total_sticker_messages = df[df['media_type'] == 'sticker'].shape[0]
    total_file_messages = df[df['media_type'] == 'file'].shape[0]
    total_animated_emojis = df[df['media_type'] == 'animation'].shape[0]
    total_link_messages = df[df['media_type'] == 'link'].shape[0]
    total_question_messages = df[df['is_question']].shape[0]

    return {
        'total_messages': total_messages,
        'total_text_messages': total_text_messages,
        'total_photo_messages': total_photo_messages,
        'total_video_messages': total_video_messages,
        'total_voice_messages': total_voice_messages,
        'total_sticker_messages': total_sticker_messages,
        'total_file_messages': total_file_messages,
        'total_animated_emojis': total_animated_emojis,
        'total_link_messages': total_link_messages,
        'total_question_messages': total_question_messages
    }


def generate_general_statistics_html(df, service_messages, current_lang):
    """Generates HTML for general chat statistics."""
    total_messages = len(df)
    total_users = df['author'].nunique()
    chat_start = df['date'].min().strftime('%Y-%m-%d %H:%M:%S')
    chat_end = df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
    most_active_day = df['date'].dt.date.value_counts().idxmax().strftime('%Y-%m-%d')
    most_active_hour = extract_most_active_hour(df)

    # Get categorized message counts
    msg_counts = get_message_counts(df, current_lang)

    html_content = f"""
    <div class="section">
        <h2>{UI_STRINGS[current_lang]['general_stats']}</h2>
        <table>
            <tr><td>{UI_STRINGS[current_lang]['total_messages']}</td><td>{total_messages}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['total_users']}</td><td>{total_users}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['chat_period']}</td><td>{chat_start} - {chat_end}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['most_active_day']}</td><td>{most_active_day}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['hour_of_day']} ({UI_STRINGS[current_lang]['most_active_day']} based)</td><td>{most_active_hour}:00</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['text_messages']}</td><td>{msg_counts['total_text_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['photos']}</td><td>{msg_counts['total_photo_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['videos']}</td><td>{msg_counts['total_video_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['voice_messages']}</td><td>{msg_counts['total_voice_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['stickers']}</td><td>{msg_counts['total_sticker_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['files']}</td><td>{msg_counts['total_file_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['animated_emojis']}</td><td>{msg_counts['total_animated_emojis']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['links']}</td><td>{msg_counts['total_link_messages']}</td></tr>
            <tr><td>{UI_STRINGS[current_lang]['questions']}</td><td>{msg_counts['total_question_messages']}</td></tr>
        </table>
    </div>
    """
    return html_content


def generate_user_summary_table_html(df, current_lang):
    """Generates an HTML table summarizing user statistics."""
    user_stats = df.groupby('author').agg(
        total_messages=('id', 'count'),
        avg_msg_length=('message_length', 'mean'),
        sentiment=('sentiment', 'mean'),
        questions_asked=('is_question', lambda x: x.sum()),
        caps_ratio_avg=('caps_ratio', 'mean'),
        voice_messages_sent=('media_type', lambda x: (x == 'voice_message').sum()),
        links_sent=('media_type', lambda x: (x == 'link').sum()),
    ).reset_index()

    user_stats.columns = [
        UI_STRINGS[current_lang]['author'],
        UI_STRINGS[current_lang]['total_messages'],
        'Avg. Msg Length',
        'Avg. Sentiment',
        'Questions Asked',
        'Avg. Caps Ratio',
        UI_STRINGS[current_lang]['voice_messages'],
        UI_STRINGS[current_lang]['links']
    ]

    # Format sentiment and caps ratio to 2 decimal places
    user_stats['Avg. Sentiment'] = user_stats['Avg. Sentiment'].round(2)
    user_stats['Avg. Caps Ratio'] = user_stats['Avg. Caps Ratio'].round(3)
    user_stats['Avg. Msg Length'] = user_stats['Avg. Msg Length'].round(1)

    user_stats_html = user_stats.to_html(index=False, classes='styled-table')
    return f"""
    <div class="section">
        <h2>{UI_STRINGS[current_lang]['user_summary_table']}</h2>
        {user_stats_html}
    </div>
    """


def create_report(df, service_messages, output_folder, current_lang):
    """Generates the full HTML report and converts to PDF."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate charts
    generate_messages_per_author_chart(df, output_folder, current_lang)
    generate_activity_heatmap(df, output_folder, current_lang)
    generate_word_cloud(df, output_folder, current_lang)
    generate_top_emojis_chart(df, output_folder, current_lang)
    generate_top_domains_chart(df, output_folder, current_lang)
    generate_media_type_distribution_chart(df, output_folder, current_lang)
    generate_reply_heatmap(df, output_folder, current_lang)
    generate_night_owls_chart(df, output_folder, current_lang)
    generate_bigrams(df, output_folder, current_lang)
    generate_top_sticker_packs_chart(df, output_folder, current_lang)

    # Generate HTML components
    general_stats_html = generate_general_statistics_html(df, service_messages, current_lang)
    user_summary_table_html = generate_user_summary_table_html(df, current_lang)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="{current_lang}">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{UI_STRINGS[current_lang]['report_title']}</title>
        <style>
            body {{ font-family: '{fm.FontProperties(fname=CONFIG['text_font_path']).get_name()}', sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 20px; }}
            .section {{ margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            table, th, td {{ border: 1px solid #ddd; }}
            th, td {{ padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .chart-img {{ max-width: 100%; height: auto; display: block; margin: 15px auto; border: 1px solid #ddd; border-radius: 4px; }}
            .footer {{ text-align: center; margin-top: 40px; font-size: 0.8em; color: #777; }}
            .styled-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 400px; border-radius: 5px 5px 0 0; overflow: hidden; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }}
            .styled-table thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }}
            .styled-table th, .styled-table td {{ padding: 12px 15px; }}
            .styled-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
            .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #009879; }}
            .styled-table tbody tr.active-row {{ font-weight: bold; color: #009879; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{UI_STRINGS[current_lang]['report_title']}</h1>
            <p class="footer">{UI_STRINGS[current_lang]['generated_at']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            {general_stats_html}

            <div class="section">
                <h2>{UI_STRINGS[current_lang]['user_activity']}</h2>
                <h3>{UI_STRINGS[current_lang]['total_messages_per_author']}</h3>
                <img src="messages_per_author.png" alt="Messages per Author" class="chart-img">
                <h3>{UI_STRINGS[current_lang]['activity_heatmap']}</h3>
                <img src="activity_heatmap.png" alt="Activity Heatmap" class="chart-img">
            </div>

            <div class="section">
                <h2>{UI_STRINGS[current_lang]['content_analysis']}</h2>
                <h3>{UI_STRINGS[current_lang]['word_cloud']}</h3>
                <img src="word_cloud.png" alt="Word Cloud" class="chart-img">
                <h3>Top Bigrams</h3>
                <img src="bigrams_chart.png" alt="Bigrams Chart" class="chart-img">
                <h3>{UI_STRINGS[current_lang]['top_emojis']}</h3>
                <img src="top_emojis.png" alt="Top Emojis" class="chart-img">
                <h3>{UI_STRINGS[current_lang]['top_domains']}</h3>
                <img src="top_domains.png" alt="Top Domains" class="chart-img">
            </div>

            <div class="section">
                <h2>{UI_STRINGS[current_lang]['media_analysis']}</h2>
                <h3>{UI_STRINGS[current_lang]['media_type_distribution']}</h3>
                <img src="media_type_distribution.png" alt="Media Type Distribution" class="chart-img">
            </div>

            <div class="section">
                <h2>{UI_STRINGS[current_lang]['interaction_analysis']}</h2>
                <h3>{UI_STRINGS[current_lang]['reply_heatmap']}</h3>
                <img src="reply_heatmap.png" alt="Reply Heatmap" class="chart-img">
            </div>

            <div class="section">
                <h2>{UI_STRINGS[current_lang]['vocabulary_analysis']}</h2>
                <h3>{UI_STRINGS[current_lang]['night_activity']}</h3>
                <img src="night_owls.png" alt="Night Owls" class="chart-img">
                <h3>{UI_STRINGS[current_lang]['top_sticker_packs']}</h3>
                <img src="top_sticker_packs.png" alt="Top Sticker Packs" class="chart-img">
            </div>

            {user_summary_table_html}

        </div>
    </body>
    </html>
    """

    html_file_path = os.path.join(output_folder, 'report.html')
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"HTML report generated at: {html_file_path}")

    if CONFIG['generate_pdf']:
        try:
            path_wkhtmltopdf = CONFIG['wkhtmltopdf_path']
            if os.path.exists(path_wkhtmltopdf):
                config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
                pdf_file_path = os.path.join(output_folder, 'report.pdf')
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.5in',
                    'margin-right': '0.5in',
                    'margin-bottom': '0.5in',
                    'margin-left': '0.5in',
                    'encoding': "UTF-8",
                    'enable-local-file-access': None,
                    'no-stop-slow-scripts': None,
                    'javascript-delay': 2000
                }
                pdfkit.from_file(html_file_path, pdf_file_path, configuration=config, options=options)
                print(f"PDF report generated at: {pdf_file_path}")
            else:
                print(f"Warning: wkhtmltopdf not found at {path_wkhtmltopdf}. PDF report will not be generated.")
        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("Please ensure wkhtmltopdf is installed and its path is correctly configured in CONFIG.")


def main():
    # Check if font files exist
    if not os.path.exists(CONFIG['text_font_path']):
        print(f"Warning: Text font file not found at {CONFIG['text_font_path']}. Report may not display correctly.")
    if not os.path.exists(CONFIG['emoji_font_path']):
        print(f"Warning: Emoji font file not found at {CONFIG['emoji_font_path']}. Emojis may not display correctly.")

    setup_nltk()
    json_path = os.path.join(CONFIG['export_folder'], CONFIG['json_file_name'])
    df, service_messages = load_and_prepare_data(json_path)

    if df is not None:
        create_report(df, service_messages, CONFIG['output_folder'], CONFIG['report_language'])
    else:
        print("Data loading failed. Report not generated.")


if __name__ == "__main__":
    main()