import requests
import base64
import json
import time

# Replace these with your actual credentials
client_id = "a0bde1015577443b9a9f234c029994f0"
client_secret = "c3184cdf826f4244aea4f6ce899bd510"

def get_token(client_id, client_secret):
    auth = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth.encode()).decode()

    headers = {
        "Authorization": f"Basic {auth_b64}"
    }

    data = {
        "grant_type": "client_credentials"
    }

    response = requests.post("https://accounts.spotify.com/api/token", data=data, headers=headers)
    response.raise_for_status()
    return response.json()["access_token"]

def get_artist_genre(artist_uri, token):
    artist_id = artist_uri.split(":")[-1]
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("genres", [])
    else:
        print(f"Failed for {artist_uri}: {response.status_code}")
        return []

def update_song_info_with_genres(input_file, output_file, client_id, client_secret):
    with open(input_file, "r") as f:
        song_info = json.load(f)

    token = get_token(client_id, client_secret)
    updated = 0

    for sid, info in song_info.items():
        if 'genre' not in info:
            genres = get_artist_genre(info["artist_uri"], token)
            song_info[sid]["genre"] = genres
            updated += 1
            time.sleep(0.1)  # avoid rate limiting

    with open(output_file, "w") as f:
        json.dump(song_info, f, indent=2)

    print(f" Updated genres for {updated} songs")

# Usage:
update_song_info_with_genres("dataset/song_info.json", "dataset/song_info_with_genres.json", client_id, client_secret)
