# -*- coding: utf-8 -*-
import json
import os
import re
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # pip install vaderSentiment
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# --- НАСТРОЙКИ СКРИПТА ---
CONFIG = {
    # 1. Путь к папке с экспортом чата
    "export_folder": r"C:\Users",
    # 2. Имя JSON-файла
    "json_file_name": "result.json",
    # 3. Папка для сохранения отчета
    "output_folder": "telegram_report_advanced",
    # 4. Язык для анализа стоп-слов ('russian', 'ukrainian', 'english')
    "language": "ukrainian",
    # 5. Язык отчета ('uk', 'en')
    "report_language": "uk",
    # 6. Путь к шрифту с поддержкой кириллицы и эмодзи
    "font_path": r'fonts/seguiemj.ttf',
    # 7. Настройки PDF
    "generate_pdf": True,
    "wkhtmltopdf_path": r"C:\Apps\wkhtmltopdf\bin\wkhtmltopdf.exe"
}

# --- СТРОКИ ДЛЯ ЛОКАЛИЗАЦИИ ---
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


# --- КОНЕЦ НАСТРОЕК ---

def add_readability(df):
    """Adds Flesch Reading-Ease score; falls back to NaN if textstat is missing."""
    try:
        import textstat
    except ImportError:
        df['flesch_score'] = np.nan
        return df
    df['flesch_score'] = df['text'].apply(
        lambda t: textstat.flesch_reading_ease(str(t))
    )
    return df


def get_sentiment(text, analyser):
    if not text.strip(): return np.nan
    return analyser.polarity_scores(text)['compound']

def add_sentiment(df):
    vader = SentimentIntensityAnalyzer()
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
    """Обработка реакций с проверкой данных"""
    if 'reactions' not in df.columns:
        return None

    def unpack(react_list):
        if not isinstance(react_list, list):
            return []
        return [(r['emoji'], r['count']) for r in react_list if isinstance(r, dict)]

    exploded = df.explode('reactions')
    reaction_data = exploded['reactions'].apply(unpack).explode().dropna()

    # Проверка наличия данных перед обработкой
    if reaction_data.empty:
        return None

    # Безопасное преобразование в DataFrame
    reaction_df = reaction_data.apply(pd.Series)

    if reaction_df.empty or len(reaction_df.columns) < 2:
        return None

    return reaction_df.groupby(0)[1].sum().sort_values(ascending=False)


def setup_nltk():
    """Загрузка всех необходимых ресурсов NLTK"""
    print("Checking for NLTK packages...")
    packages_to_check = [
        'stopwords',  # We still need the main stopwords package for other languages
        'punkt',
        'punkt_tab',
        'omw-1.4',
    ]

    for package in packages_to_check:
        try:
            # A more generic way to check for data without causing an error
            nltk.data.find(f'corpora/{package}' if package == 'stopwords' else f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading NLTK package: {package}...")
            nltk.download(package, quiet=True)

    # We remove the check for Ukrainian stopwords here, as it will fail.
    # The script will rely on the hardcoded list later.
    print("NLTK check complete.")


def process_text_field(text_field):
    """Handles the complex 'text' field from the Telegram JSON export."""
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        return ''.join([part.get('text', '') for part in text_field if isinstance(part, dict)])
    return ''


def clean_author_name(name):
    """Removes unsupported characters from author names for plotting."""
    if not isinstance(name, str):
        return "Unknown"
    # ИСПРАВЛЕНО: убран ненужный эскейп-символ
    return re.sub(r'[^\w\s.-]', '', name).strip()


def load_and_prepare_data(json_path):
    """Loads and prepares data from the Telegram JSON export file."""
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {json_path}")
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
    df['message_length'] = df['text'].str.len().fillna(0)
    df['word_count'] = df['text'].str.split().str.len().fillna(0)

    # Time features
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.date

    # Media analysis
    df['media_type'] = df['media_type'].fillna('text')
    df['sticker_emoji'] = df.apply(lambda row: row.get('sticker_emoji'), axis=1)
    df['photo_count'] = df['media_type'].apply(lambda x: 1 if x == 'photo' else 0)
    df['video_count'] = df['media_type'].apply(lambda x: 1 if x in ['video_message', 'video_file'] else 0)
    df['sticker_count'] = df['media_type'].apply(lambda x: 1 if x == 'sticker' else 0)
    df['voice_count'] = df['media_type'].apply(lambda x: 1 if x == 'voice_message' else 0)

    # Reply analysis
    reply_map = df.set_index('id')['author'].to_dict()
    df['reply_to_author'] = df['reply_to_message_id'].apply(lambda x: reply_map.get(x))

    print(f"Loaded {len(df)} messages.")
    return df, service_messages


def analyze_chat(df, language):
    """Performs a comprehensive analysis of the chat data."""
    global stop_words
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
    stats['daily_activity'] = daily_counts
    stats['heatmap_data'] = df.groupby(['hour', 'weekday']).size().unstack().fillna(0)
    stats['heatmap_data'] = stats['heatmap_data'][
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

    # --- Content & Vocabulary ---
    emojis = sum(df['text'].apply(lambda t: [c for c in t if c in emoji.EMOJI_DATA]), [])
    stats['emoji_counts'] = Counter(emojis).most_common(20)

    try:
        stop_words = set(nltk.corpus.stopwords.words(language))
        extra_stop_words = ['та', 'шо', 'це', 'не', 'я', 'в', 'і', 'ти', 'то', 'а', 'ну', 'на', 'по', 'за', 'же', 'а',
                            'або', 'але', 'в', 'вам', 'вас', 'вже', 'вона', 'вони', 'воно', 'все', 'всім', 'він',
                            'вона', 'вони', 'все', 'всім', 'від', 'він', 'вона', 'воно', 'вони', 'все',
                            'всім', 'від', 'він', 'вона', 'воно', 'вони', 'все', 'всім']
        # Fallback Ukrainian stop-words in case NLTK has none
        ukrainian_stop = set("""
        а аж також без би бо була був була були було бути в вам ваш ваше вашого вашій вашою вашу ваші
        вас вже від вона вони воно вона вони воно всі всього всіх втім він вона воно вони вона вони воно
        да для до дуже є уже й ж за зі з як якби якщо їй її їм їх її їй їх і к кого коли кому краще
        ли м мене мені могти ми мною могти мої мій моя моє мої мій моя моє мої на над нам наша нашого
        нашій нашу наші не неї немає ним ними ні ніхто но ні ніхто об одна одного однієї одних один
        одна одне одні одного одній одну однієї одних ось от по під про по під про с сам сама саме самі
        самого самій самим самими самому свого своїм своїми свою своїх себе скрізь скільки так такий
        також твій твоя твоє твої твого твоїй твоїм твоїми твоєю твоїх тебе тобі тобою того тоді той
        ті тільки том тою тут тих ти у уже хто хоча що ще щоб як якби якщо я є""".split())

        with open('uk_stop_words.txt', "r", encoding='utf-8') as f:
            words = f.read()

        stop_words = stop_words.union(extra_stop_words).union(ukrainian_stop).union(words)
        all_text = ' '.join(df['text'])
        words = [word for word in nltk.word_tokenize(all_text.lower()) if
                 word.isalpha() and word not in stop_words and len(word) > 2]
        stats['top_words'] = Counter(words).most_common(50)
    except Exception as e:
        print(
            f"WARNING: Could not process words. This may be due to missing NLTK stopwords for '{language}'. Details: {e}")
        stats['top_words'] = []

    # --- Lexical Diversity (TTR) ---
    # ИСПРАВЛЕНО: обернуто в try-except для предотвращения падения скрипта
    try:
        def calculate_ttr(text):
            tokens = [word for word in nltk.word_tokenize(text.lower(), language='russian') if word.isalpha()]
            if not tokens: return 0
            return len(set(tokens)) / len(tokens)

        user_text = df.groupby('author')['text'].apply(' '.join)
        stats['lexical_diversity'] = user_text.apply(calculate_ttr).sort_values(ascending=False)
    except Exception as e:
        print(f"WARNING: Could not calculate lexical diversity. NLTK data might be missing. Details: {e}")
        stats['lexical_diversity'] = None

    # --- Night Activity ---
    night_messages = df[df['hour'].between(0, 5)]
    stats['night_activity'] = night_messages['author'].value_counts()

    # --- Media Analysis ---
    stats['media_distribution'] = df['media_type'].value_counts()

    # --- Sticker Analysis ---
    valid_stickers = df.dropna(subset=['sticker_emoji'])
    stats['top_sticker_packs'] = valid_stickers.groupby('sticker_emoji').size().sort_values(ascending=False).head(10)

    # --- Interaction Analysis ---
    stats['reply_matrix'] = df.groupby(['author', 'reply_to_author']).size().unstack().fillna(0)

    # --- Domain Analysis (Restored) ---
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    all_links = re.findall(url_pattern, ' '.join(df['text']))
    domains = [urlparse(link).netloc for link in all_links if urlparse(link).netloc]
    stats['top_domains'] = Counter(domains).most_common(15)

    # --- Sentiment & Readability ---
    df = add_sentiment(df)
    df = add_readability(df)
    df = extract_questions(df)
    df = extract_caps_ratio(df)

    stats['sentiment'] = df['sentiment']

    stats['avg_sentiment'] = df['sentiment'].mean()
    stats['most_positive'] = df.loc[df['sentiment'].idxmax(), ['author', 'text', 'sentiment']]
    stats['most_negative'] = df.loc[df['sentiment'].idxmin(), ['author', 'text', 'sentiment']]
    stats['question_ratio'] = df['is_question'].mean()
    stats['avg_flesch'] = df['flesch_score'].mean()

    # --- Longest Message Hall of Fame ---
    stats['longest_msg'] = extract_longest_msg(df)

    # --- Conversation Starters ---
    starters = df[df['reply_to_message_id'].isna()].groupby('author').size()
    stats['conversation_starters'] = starters.sort_values(ascending=False)

    # --- Average Response Time (minutes) ---
    df_with_reply = df.dropna(subset=['reply_to_message_id']).copy()
    df_with_reply['reply_to_id'] = df_with_reply['reply_to_message_id']
    merged = df_with_reply.merge(df[['id', 'date']], left_on='reply_to_id', right_on='id',
                                 suffixes=('', '_parent'))
    merged['response_time'] = (merged['date'] - merged['date_parent']).dt.total_seconds() / 60
    stats['avg_response_time'] = merged['response_time'].mean()
    stats['fastest_responder'] = merged.groupby('author')['response_time'].mean().sort_values()

    # --- Hour with highest traffic ---
    stats['rush_hour'] = extract_most_active_hour(df)

    # --- Heatmap of message length by hour & weekday ---
    stats['len_heatmap'] = df.groupby(['hour', 'weekday'])['message_length'].mean().unstack().fillna(0)

    # --- N-gram top phrases (bigrams) ---
    try:
        vectorizer = CountVectorizer(ngram_range=(2,2), stop_words=stop_words, min_df=2)
        bigram_counts = vectorizer.fit_transform(df['text'])
        bigram_sum = bigram_counts.sum(axis=0).A1
        bigram_vocab = vectorizer.get_feature_names_out()
        stats['top_bigrams'] = sorted(zip(bigram_vocab, bigram_sum), key=lambda x: -x[1])[:20]
    except Exception:
        stats['top_bigrams'] = []

    # --- Reaction statistics (if reactions exist) ---
    stats['reaction_stats'] = reaction_matrix(df)

    # --- Emoji Sentiment ---
    emoji_sent = {
        e: df[df['text'].str.contains(e, na=False)]['sentiment'].mean()
        for e in {e for e, _ in stats['emoji_counts']}
    }
    stats['emoji_sentiment'] = pd.Series(emoji_sent).sort_values(ascending=False).dropna()

    # --- Weekly Trend ---
    weekly = df.resample('W-MON', on='date').size()
    stats['weekly_trend'] = weekly

    # --- Active Days Streak ---
    daily_active = df.groupby('day').size().reset_index()
    daily_active['day'] = pd.to_datetime(daily_active['day'])
    daily_active['streak_id'] = (
        daily_active['day'] - daily_active['day'].shift(1)
         != pd.Timedelta(days=1)
         ).astype(int).cumsum()
    streaks = daily_active.groupby('streak_id').size()
    stats['longest_streak'] = streaks.max()

    # --- Summary Table ---
    summary = df.groupby('author').agg(
        messages=('id', 'count'),
        words=('word_count', 'sum'),
        avg_len=('message_length', 'mean'),
        photos=('photo_count', 'sum'),
        videos=('video_count', 'sum'),
        stickers=('sticker_count', 'sum')
    ).round(1)
    stats['user_summary_table'] = summary.sort_values('messages', ascending=False)

    print("Analysis complete.")
    return stats


def create_visualizations(stats, output_dir, font_path, loc):
    """Creates and saves all visualizations."""
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Segoe UI Emoji'

    print("Creating visualizations...")
    if not stats:
        print("Skipping visualizations due to no data.")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    # Plot 1: Messages per Author
    plt.figure(figsize=(12, 8))
    sns.barplot(y=[clean_author_name(author) for author in stats['author_counts'].index],
                x=stats['author_counts'].values)
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

    # Plot 3: Word Cloud
    if stats.get('top_words'):
        try:
            wordcloud = WordCloud(width=1200, height=600, background_color='white', font_path=font_path,
                                  colormap='magma').generate_from_frequencies(dict(stats['top_words']))
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
        try:
            emoji_df = pd.DataFrame(stats['emoji_counts'], columns=['emoji', 'count'])
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(y='emoji', x='count', data=emoji_df, ax=ax)
            ax.set_title(loc['top_emojis'], fontsize=16)
            ax.set_xlabel(loc['count'])
            ax.set_ylabel(loc['emoji'])
            plt.setp(ax.get_yticklabels(), fontname='Segoe UI Emoji', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "top_emojis.png"))
            plt.close()
        except Exception as e:
            print(f"WARNING: Could not create emoji plot. 'Segoe UI Emoji' font might be missing. Details: {e}")

    # Plot 5: Media Distribution Pie Chart
    if stats.get('media_distribution') is not None and not stats['media_distribution'].empty:
        plt.figure(figsize=(10, 10))
        stats['media_distribution'].plot(kind='pie', autopct='%1.1f%%', startangle=140, colormap='tab20c')
        plt.title(loc['media_type_distribution'], fontsize=16)
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "media_distribution.png"))
        plt.close()

    # Plot 6: Reply Matrix Heatmap
    if stats.get('reply_matrix') is not None and not stats['reply_matrix'].empty:
        plt.figure(figsize=(14, 10))
        sns.heatmap(stats['reply_matrix'], cmap="BuPu", annot=True, fmt=".0f")
        plt.title(loc['reply_heatmap'], fontsize=16)
        plt.xlabel(loc['replied_to'])
        plt.ylabel(loc['replier'])
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "reply_heatmap.png"))
        plt.close()

    # Plot 7: Lexical Diversity
    if stats.get('lexical_diversity') is not None and not stats['lexical_diversity'].empty:
        plt.figure(figsize=(12, 8))
        stats['lexical_diversity'].plot(kind='barh')
        plt.title(loc['lexical_diversity'], fontsize=16)
        plt.xlabel("TTR")
        plt.ylabel(loc['author'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "lexical_diversity.png"))
        plt.close()

    # Plot 8: Night Owls
    if stats.get('night_activity') is not None and not stats['night_activity'].empty:
        plt.figure(figsize=(12, 8))
        stats['night_activity'].head(15).sort_values().plot(kind='barh')
        plt.title(loc['night_owls'], fontsize=16)
        plt.xlabel(loc['message_count'])
        plt.ylabel(loc['author'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "night_activity.png"))
        plt.close()

    print("Visualizations saved.")

    # 9. Sentiment histogram
    plt.figure(figsize=(10,6))
    sns.histplot(stats['sentiment'], bins=30, kde=True, color='teal')
    plt.title("Sentiment Distribution")
    plt.xlabel("Compound VADER score")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sentiment_hist.png"))
    plt.close()

    # 10. Average response time bar
    if 'fastest_responder' in stats:
        plt.figure(figsize=(12,6))
        stats['fastest_responder'].head(15).sort_values().plot(kind='barh')
        plt.title("Average Response Time (minutes)")
        plt.xlabel("minutes")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "response_time.png"))
        plt.close()

    # 11. Weekly trend line
    if 'weekly_trend' in stats:
        plt.figure(figsize=(14,4))
        stats['weekly_trend'].plot(marker='o')
        plt.title("Weekly Message Volume")
        plt.ylabel("Messages")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "weekly_trend.png"))
        plt.close()

    # 12. Bigram bar
    if stats.get('top_bigrams'):
        bigram_df = pd.DataFrame(stats['top_bigrams'], columns=['phrase', 'count'])
        plt.figure(figsize=(12,6))
        sns.barplot(x='count', y='phrase', data=bigram_df, palette='mako')
        plt.title("Top 20 Bigrams")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "bigrams.png"))
        plt.close()

    # 13. Message length heatmap
    if 'len_heatmap' in stats:
        plt.figure(figsize=(12,8))
        sns.heatmap(stats['len_heatmap'], cmap="rocket_r", linewidths=.5, annot=True, fmt=".0f")
        plt.title("Average Message Length by Hour & Weekday")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "len_heatmap.png"))
        plt.close()


def generate_html_report(stats, output_dir, loc):
    """Generates a beautiful HTML report."""
    print("Creating HTML report...")
    if not stats: return None

    def series_to_html_table(data, table_headers):
        if data is None or not data: return "<p>Нет данных.</p>"
        # ИСПРАВЛЕНО: переменная html переименована в html_content
        html_content = '<table class="data-table"><tr>'
        for header in table_headers: html_content += f"<th>{header}</th>"
        html_content += "</tr>"
        for index, value in data:
            html_content += f"<tr><td>{index}</td><td>{value}</td></tr>"
        return html_content + "</table>"

    def df_to_html_table(df):
        if df is None or df.empty: return "<p>Нет данных.</p>"
        # ИСПРАВЛЕНО: убран неиспользуемый параметр headers
        return df.to_html(classes='data-table', index=True)

    html_template = f"""
    <html><head><meta charset="UTF-8"><title>{loc['report_title']}</title>
    <style>
    body {{
        font-family: 'Segoe UI', Roboto, sans-serif;
        background-color: #f4f7f6;
        color: #333;
        margin: 0;
        padding: 20px;
    }}

    .container {{
        max-width: 1200px;
        margin: auto;
        background: #fff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}

    h1, h2 {{
        color: #0056b3;
        border-bottom: 2px solid #0056b3;
        padding-bottom: 10px;
    }}

    h1 {{
        text-align: center;
        font-size: 2.5em;
    }}

    h2 {{
        font-size: 1.8em;
        margin-top: 40px;
    }}

    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }}

    .card {{
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }}

    .card h3 {{
        font-size: 1.2em;
        margin-top: 0;
        color: #333;
        border: none;
    }}

    .card p {{
        font-size: 1.5em;
        font-weight: bold;
        color: #0056b3;
        margin: 10px 0;
    }}

    .full-width {{
        grid-column: 1 / -1;
    }}

    img {{
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        margin-top: 15px;
    }}

    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 0.9em;
    }}

    .data-table th, .data-table td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }}

    .data-table th {{
        background-color: #0056b3;
        color: white;
    }}

    .data-table tr:nth-child(even) {{
        background-color: #f2f2f2;
    }}

    footer {{
        text-align: center;
        margin-top: 40px;
        color: #777;
        font-size: 0.9em;
    }}
</style>
</head><body><div class="container">
        <h1>{loc['report_title']}</h1>
        <h2>{loc['general_stats']}</h2>
        <div class="grid">
            <div class="card"><h3>{loc['total_messages']}</h3><p>{stats.get('total_messages', 0):,}</p></div>
            <div class="card"><h3>{loc['total_users']}</h3><p>{len(stats.get('authors', []))}</p></div>
            <div class="card"><h3>{loc['chat_period']}</h3><p style="font-size: 1.1em">{stats.get('chat_period', 'N/A')}</p></div>
            <div class="card"><h3>{loc['most_active_day']}</h3><p style="font-size: 1.1em">{stats.get('most_active_day', 'N/A')} ({stats.get('most_active_day_count', 0)} {loc['messages_abbr']})</p></div>
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
        
        
                <h2>Advanced Analytics</h2>

        <div class="grid">
            <div class="card"><h3>Sentiment Distribution</h3><img src="plots/sentiment_hist.png"></div>
            <div class="card"><h3>Top Bigrams</h3><img src="plots/bigrams.png"></div>
        </div>

        <div class="grid">
            <div class="card"><h3>Weekly Trend</h3><img src="plots/weekly_trend.png"></div>
            <div class="card"><h3>Response Time Leaders</h3><img src="plots/response_time.png"></div>
        </div>

        <div class="grid">
            <div class="card"><h3>Longest Message</h3>
                <p><strong>{stats['longest_msg']['author']}</strong></p>
                <p style="font-size:0.9em; color:#555; max-height:100px; overflow:auto">{stats['longest_msg']['text'][:200]}…</p>
                <p>{stats['longest_msg']['message_length']} chars</p>
            </div>
            <div class="card"><h3>Conversation Starters</h3>
                {series_to_html_table(stats.get('conversation_starters', {}).head(10).items(), ["User", "Starts"])}
            </div>
        </div>
        

        <h2>{loc['content_analysis']}</h2>
        <div class="card full-width"><h3>{loc['word_cloud']}</h3><img src="plots/wordcloud.png"></div>
        <div class="grid">
            <div class="card"><h3>{loc['top_emojis']}</h3><img src="plots/top_emojis.png"></div>
            <div class="card"><h3>{loc['top_sticker_packs']}</h3>{series_to_html_table(stats.get('top_sticker_packs', {}).items(), [loc['sticker_pack'], loc['count']])}</div>
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
        print("ERROR: 'pdfkit' library not installed. PDF report will not be created.")
        return

    pdf_path = os.path.join(output_dir, "report.pdf")
    try:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        options = {"enable-local-file-access": "", "encoding": "UTF-8", "page-size": "A4"}
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

    # ИСПРАВЛЕНО: переменная T переименована в loc для ясности
    loc = UI_STRINGS.get(CONFIG['report_language'], UI_STRINGS['en'])

    json_path = os.path.join(CONFIG['export_folder'], CONFIG['json_file_name'])
    df, service_messages = load_and_prepare_data(json_path)

    if df is None:
        print("Exiting due to data loading failure.")
        return

    chat_stats = analyze_chat(df, CONFIG['language'])
    create_visualizations(chat_stats, CONFIG['output_folder'], CONFIG['font_path'], loc)
    html_report_path = generate_html_report(chat_stats, CONFIG['output_folder'], loc)

    if CONFIG['generate_pdf']:
        generate_pdf_report(html_report_path, CONFIG['output_folder'], CONFIG['wkhtmltopdf_path'])

    print(f"\nDone! All files are in the folder: {os.path.abspath(CONFIG['output_folder'])}")


if __name__ == "__main__":
    main()
