import requests
from app.config import LASTFM_API_KEY

BASE_URL = "https://ws.audioscrobbler.com/2.0"


def _request(method: str, **params):
    params["method"] = method
    params["api_key"] = LASTFM_API_KEY
    params["format"] = "json"
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


"""Fetch listener count, playcount, and basic tags for a specific track.

Listener counts are collected and surfaced, but are not used to filter tracks.
"""
def get_track_info(artist: str, track: str) -> dict:
    data = _request("track.getInfo", artist=artist, track=track)
    track_data = data.get("track", {})

    listeners = int(track_data.get("listeners", 0))
    playcount = int(track_data.get("playcount", 0))

    toptags = track_data.get("toptags", {}).get("tag", [])
    tags = [t["name"].lower().strip() for t in toptags]

    return {
        "listeners": listeners,
        "playcount": playcount,
        "tags": tags
    }


"""
    core of the embedding pipeline, returns a tag_name : count dict where count is how many users applied that tag to the artist (our confidence score) this dict is what gets passed into embeddings.py to become a vector

"""
def get_artist_top_tags(artist: str) -> dict:
    data = _request("artist.getTopTags", artist=artist)
    toptags = data.get("toptags", {}).get("tag", [])

    tag_counts = {}
    for t in toptags:
        name = t["name"].lower().strip()
        count = int(t["count"])
        tag_counts[name] = count

    return tag_counts


"""
    asks last.fm for tracks similar to a given one, ideally we want to use our vectordb but at the start we'll probably be using this a lot since our db will be quite sparse
"""
def search_tracks(query: str, limit: int = 10) -> list:
    data = _request("track.search", track=query, limit=limit)
    tracks = data.get("results", {}).get("trackmatches", {}).get("track", [])

    results = []
    for t in tracks:
        images = t.get("image", [])
        image = next((img["#text"] for img in reversed(images) if img.get("#text")), None)
        results.append({
            "name": t["name"],
            "artist": t["artist"],
            "listeners": int(t.get("listeners", 0)),
            "image": image
        })
    return results


def get_track_top_tags(artist: str, track: str) -> dict:
    data = _request("track.getTopTags", artist=artist, track=track)
    toptags = data.get("toptags", {}).get("tag", [])

    tag_counts = {}
    for t in toptags:
        name = t["name"].lower().strip()
        count = int(t["count"])
        if count > 0:
            tag_counts[name] = count

    return tag_counts


def get_similar_artists(artist: str, limit: int = 5) -> list:
    data = _request("artist.getSimilar", artist=artist, limit=limit)
    similar = data.get("similarartists", {}).get("artist", [])
    return [
        {"artist": a["name"], "match": float(a["match"])}
        for a in similar
        if float(a.get("match", 0)) > 0.5
    ]


def blend_tags(
    artist_tags: dict,
    track_tags: dict,
    similar_artist_tags: list = None,
    artist_weight: float = 0.3,
    similar_weight: float = 0.1
) -> dict:
    blended = {tag: int(count * artist_weight) for tag, count in artist_tags.items()}

    if similar_artist_tags:
        for tags, match in similar_artist_tags:
            for tag, count in tags.items():
                contribution = int(count * similar_weight * match)
                blended[tag] = blended.get(tag, 0) + contribution

    for tag, count in track_tags.items():
        blended[tag] = blended.get(tag, 0) + count

    return {tag: count for tag, count in blended.items() if count > 0}


def get_artist_top_tracks(artist: str, limit: int = 10) -> list:
    """
    Most-played tracks for an artist. Used as a cold-start fallback when a seed
    has no track.getSimilar results: we mine the seed's similar artists' top
    tracks so even an obscure/instrumental song still grows a graph.
    """
    data = _request("artist.getTopTracks", artist=artist, limit=limit)
    tracks = data.get("toptracks", {}).get("track", [])

    results = []
    for t in tracks:
        a = t.get("artist")
        name = a["name"] if isinstance(a, dict) and a.get("name") else artist
        results.append({"name": t["name"], "artist": name})
    return results


def get_similar_tracks(artist: str, track: str, limit: int = 10) -> list:
    data = _request("track.getSimilar", artist=artist, track=track, limit=limit)
    similar = data.get("similartracks", {}).get("track", [])

    results = []
    for t in similar:
        results.append({
            "name": t["name"],
            "artist": t["artist"]["name"],
            "match": float(t.get("match", 0))
        })

    return results
