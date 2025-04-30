import streamlit as st
import torch
import json
from model_definition import LightGCNModel
from torch_geometric.data.storage import GlobalStorage
import torch.serialization

# ---------------------- SETUP ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load graph
with torch.serialization.safe_globals([GlobalStorage]):
    data = torch.load("dataset/data_object.pt", weights_only=False).to(device)

# Load model
model = LightGCNModel(num_nodes=data.num_nodes).to(device)
model.load_state_dict(torch.load("model/model_weights.pth", map_location=device))
model.eval()

# Load metadata
with open("dataset/song_info_with_genres.json") as f:
    song_info = json.load(f)

# ---------------------- FUNCTIONS ----------------------
def recommend_similar_song(song_id, top_k=10):
    emb = model(data.edge_index)
    song_emb = emb[song_id]
    similarities = torch.matmul(emb, song_emb)
    top_indices = similarities.topk(top_k + 1).indices.tolist()
    filtered = [i for i in top_indices if i != song_id and str(i) in song_info][:top_k]
    return filtered

def recommend_by_embedding_average(song_ids, top_k=10):
    emb = model(data.edge_index)
    mean_emb = emb[song_ids].mean(dim=0)
    similarities = torch.matmul(emb, mean_emb)
    top_indices = similarities.topk(top_k).indices.tolist()
    filtered = [i for i in top_indices if str(i) in song_info]
    return filtered

def get_song_display(sid):
    s = song_info[str(sid)]
    return f"**{s['track_name']}** — *{s['artist_name']}*"

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="🎵 Music Recommender", layout="centered")
st.title("🎵 Music Recommender")

# Genre Dropdown
all_genres = sorted({genre for s in song_info.values() for genre in s.get('genre', [])})
genre_option = st.selectbox("🎧 Or explore by genre:", [""] + all_genres)

# Text Search
query = st.text_input("🔎 Or search by song, artist, or keyword:")

# Session state for "More Like This"
if "more_like_song_id" not in st.session_state:
    st.session_state.more_like_song_id = None

# ---------------------- RECOMMENDATION LOGIC ----------------------
recs = []

# Priority: More Like This button
if st.session_state.more_like_song_id is not None:
    sid = st.session_state.more_like_song_id
    st.subheader(f" More like **{song_info[str(sid)]['track_name']}**")
    recs = recommend_similar_song(sid)
    st.session_state.more_like_song_id = None

# Genre-based
elif genre_option:
    genre_matches = [int(sid) for sid, s in song_info.items()
                     if 'genre' in s and genre_option in s['genre']]
    st.subheader(f" Recommendations for {genre_option.title()}")
    recs = recommend_by_embedding_average(genre_matches)

# Text search
elif query:
    q = query.lower()
    exact_match = next((int(sid) for sid, info in song_info.items()
                        if info['track_name'].lower() == q), None)

    if exact_match is not None:
        st.subheader(" Exact match found! Recommendations:")
        recs = recommend_similar_song(exact_match)
    else:
        artist_matches = [int(sid) for sid, info in song_info.items()
                          if info['artist_name'].lower() == q]

        if artist_matches:
            st.subheader(" Artist match found! Recommendations:")
            recs = recommend_by_embedding_average(artist_matches)
        else:
            keyword_matches = [int(sid) for sid, info in song_info.items()
                               if q in info['track_name'].lower() or q in info['artist_name'].lower()]
            if keyword_matches:
                st.subheader(" Keyword match found! Recommendations:")
                recs = recommend_by_embedding_average(keyword_matches)
            else:
                st.error("No match found for your query.")

# ---------------------- DISPLAY RECOMMENDATIONS ----------------------
if recs:
    for sid in recs:
        s = song_info[str(sid)]
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(get_song_display(sid))
        with col2:
            if st.button(" More like this", key=f"more_{sid}"):
                st.session_state.more_like_song_id = sid
                st.rerun()
