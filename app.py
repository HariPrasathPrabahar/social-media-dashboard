import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
from io import BytesIO

st.set_page_config(layout="wide", page_title="Social Media Analytics Dashboard")

@st.cache_data(show_spinner=False)
def load_prepped_df(path="tweets_prepped.pkl"):
    try:
        return pd.read_pickle(path)
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_pipeline(path="engagement_pipe.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

def normalize_hashtags_col(df):
    if 'hashtags' not in df.columns:
        df['hashtags'] = df['content'].str.findall(r'#\w+').apply(lambda x: [h.lower() for h in x] if isinstance(x, list) else [])
    else:
        df['hashtags'] = df['hashtags'].apply(lambda x: [h.lower() for h in x] if isinstance(x, list) else [])

def recommend_hashtags(df, topic=None, emotion=None, k=10, min_uses=5):
    df2 = df.copy()
    if topic:
        df2 = df2[df2['topic'] == topic]
    if emotion:
        df2 = df2[df2['emotion'] == emotion]
    df2 = df2.explode('hashtags').dropna(subset=['hashtags'])
    if df2.empty:
        return pd.DataFrame(columns=['hashtags','count','mean_engagement'])
    df2['eng_norm'] = df2['engagement'] / (df2['text_len'] + 1)
    stats = (df2.groupby('hashtags')['eng_norm']
             .agg(['count','mean'])
             .query('count >= @min_uses')
             .reset_index()
             .rename(columns={'mean':'mean_engagement'})
             .sort_values('mean_engagement', ascending=False)
             .head(k))
    return stats

def recommend_hours(df, topic=None, k=6, min_posts=20):
    df2 = df.copy()
    if topic:
        df2 = df2[df2['topic'] == topic]
    by_hour = df2.groupby('hour')['engagement'].agg(['count','mean']).reset_index().sort_values('mean', ascending=False)
    if by_hour.empty:
        return by_hour.head(0)
    return by_hour.query('count >= @min_posts').head(k)

def recommend_influencers(df, topic=None, k=10, min_mentions=3):
    df2 = df.copy()
    if topic:
        df2 = df2[df2['topic'] == topic]
    rows = []
    for ents, eng in df2[['entities','engagement']].itertuples(index=False):
        if isinstance(ents, list):
            for text, label in ents:
                if label == 'PERSON':
                    rows.append((text.strip().lower(), eng))
    if not rows:
        return pd.DataFrame(columns=['person','count','avg_engagement'])
    tmp = pd.DataFrame(rows, columns=['person','eng'])
    stats = (tmp.groupby('person')['eng']
             .agg(['count','mean'])
             .query('count >= @min_mentions')
             .reset_index()
             .rename(columns={'mean':'avg_engagement'})
             .sort_values('avg_engagement', ascending=False)
             .head(k))
    return stats

def generate_templates_from_top(df, top_n=50, out_n=10):
    top_df = df.sort_values('engagement', ascending=False).head(top_n)
    templates = []
    for txt in top_df['content'].dropna().unique():
        s = re.sub(r'http\S+','', txt)
        s = re.sub(r'@\w+', '{mention}', s)
        s = re.sub(r'#\w+', '{hashtag}', s)
        s = s.strip()
        if 10 < len(s) < 280:  # skip extremely short/long
            templates.append(s)
        if len(templates) >= out_n:
            break
    return templates

def to_csv_bytes(df):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# --- App UI ---
st.title("TREND - E")
st.markdown("Social Media Analytics Dashboard")

df = load_prepped_df()
pipe = load_pipeline()

if df is None:
    st.warning("Could not find 'tweets_prepped.pkl' in the app folder. Upload it now (or add it to repo).")
    uploaded = st.file_uploader("Upload tweets_prepped.pkl (pickle)", type=['pkl','pickle'])
    if uploaded:
        df = pd.read_pickle(uploaded)
        st.success("Dataset loaded from upload.")
else:
    st.success(f"Loaded dataset with {len(df):,} rows.")

if df is not None:
    if 'engagement' not in df.columns:
        df['engagement'] = df['number_of_likes'].fillna(0) + df['number_of_shares'].fillna(0)
    if 'hour' not in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df['hour'] = df['date_time'].dt.hour.fillna(-1).astype(int)
    normalize_hashtags_col(df)

    st.sidebar.header("Filters")
    authors = ["(All)"] + sorted(df['author'].dropna().unique().tolist())
    topics = ["(All)"] + sorted(df['topic'].dropna().unique().tolist())
    emotions = ["(All)"] + sorted(df['emotion'].dropna().unique().tolist())

    sel_author = st.sidebar.selectbox("Author", authors)
    sel_topic  = st.sidebar.selectbox("Topic", topics)
    sel_emotion = st.sidebar.selectbox("Emotion", emotions)
    min_likes = st.sidebar.number_input("Min likes", value=0, step=1)

    view = df.copy()
    if sel_author != "(All)":
        view = view[view['author'] == sel_author]
    if sel_topic != "(All)":
        view = view[view['topic'] == sel_topic]
    if sel_emotion != "(All)":
        view = view[view['emotion'] == sel_emotion]
    view = view[view['number_of_likes'] >= min_likes]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tweets in view", f"{len(view):,}")
    col2.metric("Avg engagement", f"{int(view['engagement'].mean() if len(view) else 0):,}")
    top_hour = int(view.groupby(view['date_time'].dt.hour)['engagement'].mean().idxmax()) if len(view) and view['date_time'].notna().any() else "-"
    col3.metric("Top posting hour (avg eng)", top_hour)
    col4.metric("Top emotion", view['emotion'].mode().iloc[0] if len(view) else "-")

    st.subheader("Engagement over time")
    view['date'] = pd.to_datetime(view['date_time']).dt.date
    daily = view.groupby('date')['engagement'].sum().reset_index()
    if not daily.empty:
        fig = px.line(daily, x='date', y='engagement', title="Daily Engagement")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No date/time data to plot.")

    st.subheader("Top Hashtags (by avg engagement)")
    ht_stats = recommend_hashtags(view, topic=(None if sel_topic=="(All)" else sel_topic), emotion=(None if sel_emotion=="(All)" else sel_emotion), k=30)
    st.dataframe(ht_stats)

    st.subheader("Top Influencers (PERSON) by avg engagement")
    infl = recommend_influencers(view, topic=(None if sel_topic=="(All)" else sel_topic), k=30)
    st.dataframe(infl)

    st.subheader("Best Posting Hours (by avg engagement)")
    hours_df = recommend_hours(view, topic=(None if sel_topic=="(All)" else sel_topic), k=12)
    st.dataframe(hours_df)

    st.subheader("High-performing caption templates (from top tweets)")
    templates = generate_templates_from_top(view, top_n=200, out_n=25)
    for t in templates:
        st.write("- " + t)

    st.header("Recommendations (data-driven)")
    rec_topic = None if sel_topic=="(All)" else sel_topic
    recommended_hashtags = recommend_hashtags(df, topic=rec_topic, k=10)
    recommended_hours = recommend_hours(df, topic=rec_topic, k=6)
    recommended_influencers = recommend_influencers(df, topic=rec_topic, k=10)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Hashtags")
        st.dataframe(recommended_hashtags)
    with c2:
        st.subheader("Posting hours")
        st.dataframe(recommended_hours)
    with c3:
        st.subheader("Influencers")
        st.dataframe(recommended_influencers)

    if not recommended_hashtags.empty:
        csv_bytes = to_csv_bytes(recommended_hashtags)
        st.download_button("Download hashtag recommendations (CSV)", csv_bytes, file_name="hashtag_recommendations.csv")
    else:
        st.write("No hashtag recommendations available.")

    if pipe is not None:
        st.subheader("Estimate expected engagement for candidate hashtags (model-based)")
        candidate_input = st.text_input("Enter candidate hashtags separated by commas (e.g. #news,#sports):", "")
        if st.button("Score candidates") and candidate_input.strip():
            candidates = [c.strip().lower() for c in candidate_input.split(",") if c.strip()]
            base = df.sample(min(200, len(df))).copy()
            scores = []
            for h in candidates:
                base['hashtags_tmp'] = base['hashtags'].apply(lambda xs: list(set((xs or []) + [h])))
                base['num_hashtags'] = base['hashtags_tmp'].apply(len)
                Xcand = base[['polarity','subjectivity','text_len','num_hashtags','hour','dow','num_person','num_gpe','num_org','emotion','sentiment','topic','author','language']].copy()
                # fill and prepare similarly to training
                Xcand['emotion'] = Xcand['emotion'].fillna('NA').astype(str)
                Xcand['sentiment'] = Xcand['sentiment'].fillna('NA').astype(str)
                Xcand['topic'] = Xcand['topic'].fillna('NA').astype(str)
                Xcand['author'] = Xcand['author'].fillna('NA').astype(str)
                Xcand['language'] = Xcand['language'].fillna('NA').astype(str)
                Xcand = Xcand.fillna(0)
                y_hat = pipe.predict(Xcand)
                expected_eng = float(np.expm1(y_hat).mean())
                scores.append((h, expected_eng))
            score_df = pd.DataFrame(scores, columns=['hashtag','expected_engagement']).sort_values('expected_engagement', ascending=False)
            st.dataframe(score_df)
    else:
        st.info("Engagement prediction pipeline not found. Upload 'engagement_pipe.joblib' to enable model-driven scores.")
