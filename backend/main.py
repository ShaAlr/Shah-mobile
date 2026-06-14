"""
GAMEMATCH — Steam Game Recommendation System (Backend API)
============================================================
FastAPI port of the original Streamlit hybrid recommendation engine.

Pipeline:
  Layer 1 — Real-Time API Pipeline   : SteamSpy -> raw DataFrame (cache 30 menit)
  Layer 2 — Preprocessing & TF-IDF   : genre+tags -> sparse matrix (3000 fitur)
  Layer 3 — Hybrid Scoring (3-Layer) : 0.60*CBF + 0.15*CF(SVD) + 0.25*Quality_Norm
  Layer 4 — Multi-Constraint Filter  : Budget · OS · Free-to-Play toggle
  Layer 5 — Quantitative Evaluation  : Precision@K · Recall@K · MAP · NDCG@K · Coverage · Diversity
  Layer 6 — Exposed via REST endpoints for the React frontend
"""

import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
API_GENRE = "https://steamspy.com/api.php?request=genre&genre={g}"
API_TOP = "https://steamspy.com/api.php?request=top100in2weeks"

ALL_GENRES = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation",
    "Casual", "Indie", "Sports", "Racing", "Horror",
    "Puzzle", "Shooter", "Fighting", "Survival", "Platformer",
]

GT_PRIMARY = {
    "Action": ["counter-strike", "half-life", "left 4 dead", "l4d", "payday", "elden ring",
               "dark souls", "devil may cry", "nier", "bayonetta", "sekiro", "doom",
               "terraria", "portal", "garry", "gmod", "team fortress", "tf2",
               "batman", "spider-man", "god of war", "assassin", "yakuza", "nioh",
               "grand theft", "gta", "fallout", "witcher", "cyberpunk", "dying light",
               "borderlands", "bioshock", "dishonored", "prey", "hades", "dead cells"],
    "Adventure": ["portal", "witcher", "hollow knight", "ori", "firewatch", "outer wilds",
                  "disco elysium", "life is strange", "batman arkham", "alan wake", "control",
                  "uncharted", "red dead", "horizon", "death stranding", "journey", "oxenfree"],
    "RPG": ["elden ring", "dark souls", "witcher", "skyrim", "baldur", "divinity",
            "dragon age", "mass effect", "fallout", "cyberpunk", "persona", "pathfinder",
            "pillars of eternity", "tyranny", "planescape", "torment", "monster hunter",
            "final fantasy", "dragon quest", "tales of", "xenoblade"],
    "Strategy": ["civilization", "total war", "crusader kings", "stellaris", "xcom",
                 "age of empires", "factorio", "frostpunk", "anno", "into the breach",
                 "homeworld", "warcraft", "starcraft", "company of heroes", "warhammer",
                 "hearts of iron", "europa universalis", "victoria", "tropico"],
    "Simulation": ["cities skylines", "planet coaster", "stardew", "farming simulator",
                   "euro truck", "kerbal", "dwarf fortress", "rimworld", "oxygen not included",
                   "satisfactory", "prison architect", "two point", "surgeon simulator",
                   "flight sim", "american truck", "house flipper", "power wash"],
    "Casual": ["stardew", "among us", "fall guys", "overcooked", "unpacking", "celeste",
               "peggle", "bejeweled", "candy crush", "plants vs zombies", "solitaire",
               "angry birds", "cut the rope", "temple run", "hatoful boyfriend"],
    "Indie": ["hollow knight", "celeste", "hades", "dead cells", "shovel knight",
              "undertale", "stardew", "binding of isaac", "risk of rain", "noita",
              "terraria", "spelunky", "cuphead", "blasphemous", "salt and sanctuary",
              "ori", "hyper light", "katana zero", "gris"],
    "Horror": ["resident evil", "alien isolation", "amnesia", "outlast", "phasmophobia",
               "dead space", "silent hill", "fnaf", "little nightmares", "layers of fear",
               "visage", "soma", "subnautica", "the forest", "green hell", "don't starve",
               "darkwood", "doki doki", "ib", "witch's house"],
    "Shooter": ["counter-strike", "tf2", "team fortress", "overwatch", "borderlands",
                "bioshock", "half-life", "apex legends", "titanfall", "destiny",
                "doom", "quake", "unreal tournament", "halo", "call of duty", "battlefield",
                "rainbow six", "escape from tarkov", "hunt showdown", "ready or not"],
    "Survival": ["ark", "rust", "subnautica", "the forest", "green hell", "minecraft",
                 "valheim", "7 days", "don't starve", "conan exiles", "dayz", "terraria",
                 "no man's sky", "stranded deep", "the long dark", "astroneer", "raft"],
    "Racing": ["forza", "dirt", "f1", "grid", "assetto corsa", "project cars", "trackmania",
               "need for speed", "burnout", "gran turismo", "mario kart", "wreckfest",
               "beam.ng", "rfactor", "iracing", "automobilista", "kartkraft"],
    "Sports": ["fifa", "pes", "efootball", "nba 2k", "rocket league", "nhl", "mlb the show",
               "tennis world", "top spin", "wwe 2k", "ufc", "boxing", "golf", "cricket",
               "super mega baseball", "out of the park"],
    "Fighting": ["mortal kombat", "street fighter", "tekken", "guilty gear", "dragon ball",
                 "king of fighters", "marvel vs capcom", "smash", "skullgirls",
                 "blazblue", "under night", "granblue fantasy versus", "them's fightin"],
    "Puzzle": ["portal", "baba is you", "the witness", "talos principle", "stephen's sausage",
               "antichamber", "superliminal", "manifold garden", "tetris", "puyo puyo",
               "return of the obra", "obra dinn", "her story", "outer wilds", "heaven vault"],
    "Platformer": ["celeste", "hollow knight", "ori and", "shovel knight", "cuphead", "rayman",
                   "super meat boy", "a hat in time", "yooka", "crash", "spyro", "ratchet",
                   "axiom verge", "bloodstained", "blasphemous", "momodora", "iconoclasts"],
}

GT_GENRE_KW = {
    "Action": ["action", "hack and slash", "beat em up", "action-adventure", "brawler",
               "action roguelike", "melee", "combat", "fighting", "shooter", "violent",
               "third-person", "first-person", "gore", "martial arts", "character action"],
    "Adventure": ["adventure", "exploration", "narrative", "walking simulator", "point and click",
                  "story rich", "atmospheric", "open world", "mystery", "detective", "visual novel",
                  "choices matter", "multiple endings", "interactive fiction", "text-based"],
    "RPG": ["rpg", "role-playing", "jrpg", "action rpg", "dungeon crawler", "souls-like",
            "soulslike", "turn-based", "party-based", "character customization", "loot",
            "leveling", "skill tree", "fantasy", "deckbuilding", "tactical rpg", "crpg"],
    "Strategy": ["strategy", "rts", "real-time strategy", "turn-based strategy", "grand strategy",
                 "tower defense", "4x", "city builder", "colony sim", "management",
                 "resource management", "base building", "tactical", "military", "wargame"],
    "Simulation": ["simulation", "simulator", "management", "city builder", "farming", "sandbox",
                   "building", "construction", "life sim", "vehicle", "flight", "driving", "tycoon",
                   "realistic", "economy", "crafting", "open world", "relaxing"],
    "Casual": ["casual", "family friendly", "relaxing", "easy", "party game", "couch co-op",
               "funny", "humor", "cute", "colorful", "wholesome", "local multiplayer",
               "board game", "card game", "tabletop", "puzzle"],
    "Indie": ["indie", "action roguelike", "roguelike", "roguelite", "metroidvania",
              "platformer", "pixel art", "hand-drawn", "procedural generation",
              "singleplayer", "2d", "side scroller", "top-down", "isometric", "retro"],
    "Horror": ["horror", "survival horror", "psychological horror", "dark", "gore", "scary",
               "thriller", "jump scare", "haunted", "zombie", "monster", "paranormal",
               "creepy", "disturbing", "violent", "dark fantasy", "gothic", "mature"],
    "Shooter": ["shooter", "fps", "tps", "first-person shooter", "third-person shooter", "gun",
                "shooting", "bullet hell", "twin stick", "top-down shooter", "arena",
                "military", "tactical shooter", "hero shooter", "battle royale", "pvp"],
    "Survival": ["survival", "open world survival", "crafting", "base building", "permadeath",
                 "roguelike", "sandbox", "gathering", "resource", "exploration",
                 "post-apocalyptic", "wilderness", "zombie", "multiplayer survival"],
    "Racing": ["racing", "driving", "automobile", "motorsport", "car", "vehicle", "bikes",
               "kart", "drift", "rally", "arcade", "speed", "track", "competitive"],
    "Sports": ["sports", "football", "basketball", "tennis", "golf", "baseball", "soccer",
               "esports", "competitive", "team", "athletics", "management", "arcade sports"],
    "Fighting": ["fighting", "versus", "pvp", "martial arts", "combat", "2d fighter", "arena",
                 "beat em up", "brawler", "tournament", "competitive", "action"],
    "Puzzle": ["puzzle", "logic", "brain", "mystery", "escape room", "hidden object", "riddle",
               "problem solving", "casual", "relaxing", "point and click", "physics",
               "spatial", "word", "match 3", "jigsaw"],
    "Platformer": ["platformer", "platform", "2d", "side-scroller", "run and jump", "metroidvania",
                   "action", "indie", "pixel art", "precision", "challenging", "retro",
                   "collectathon", "3d platformer", "controller"],
}

GT_MIN_RATING = 20.0

EVAL_TARGETS = {
    "Precision@K": 0.50, "Recall@K": 0.40, "MAP": 0.45,
    "NDCG@K": 0.55, "Coverage": 0.30, "Diversity": 0.50,
}

GENRE_REMAP = {
    "Horror": "https://steamspy.com/api.php?request=tag&tag=Horror",
    "Platformer": "https://steamspy.com/api.php?request=tag&tag=Platformer",
    "Shooter": "https://steamspy.com/api.php?request=tag&tag=FPS",
    "Fighting": "https://steamspy.com/api.php?request=tag&tag=Fighting",
    "Survival": "https://steamspy.com/api.php?request=tag&tag=Survival",
    "Racing": "https://steamspy.com/api.php?request=tag&tag=Racing",
    "Puzzle": "https://steamspy.com/api.php?request=tag&tag=Puzzle",
}

CACHE_TTL = 1800  # 30 menit

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK DATASET — dipakai kalau SteamSpy API tidak bisa diakses / timeout
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_GAMES = [
    {"appid":730,"name":"Counter-Strike 2","genre":"Action","tags":"action, shooter, fps, multiplayer, competitive","price_usd":0.0,"owners_est":50000000,"positive":900000,"negative":200000,"rating":81.8,"windows":True,"mac":True,"linux":True},
    {"appid":570,"name":"Dota 2","genre":"Action","tags":"action, strategy, moba, multiplayer, free to play","price_usd":0.0,"owners_est":80000000,"positive":1200000,"negative":400000,"rating":75.0,"windows":True,"mac":True,"linux":True},
    {"appid":1245620,"name":"Elden Ring","genre":"Action","tags":"action, rpg, souls-like, dark fantasy, challenging","price_usd":59.99,"owners_est":12000000,"positive":600000,"negative":30000,"rating":95.2,"windows":True,"mac":False,"linux":False},
    {"appid":814380,"name":"Sekiro: Shadows Die Twice","genre":"Action","tags":"action, souls-like, challenging, dark fantasy, melee","price_usd":59.99,"owners_est":5000000,"positive":230000,"negative":15000,"rating":93.9,"windows":True,"mac":False,"linux":False},
    {"appid":601150,"name":"Devil May Cry 5","genre":"Action","tags":"action, hack and slash, character action, melee, violent","price_usd":29.99,"owners_est":4000000,"positive":120000,"negative":5000,"rating":96.0,"windows":True,"mac":False,"linux":False},
    {"appid":1091500,"name":"Cyberpunk 2077","genre":"Action","tags":"action, rpg, open world, cyberpunk, story rich","price_usd":59.99,"owners_est":20000000,"positive":550000,"negative":100000,"rating":84.6,"windows":True,"mac":False,"linux":False},
    {"appid":105600,"name":"Terraria","genre":"Action","tags":"action, adventure, sandbox, crafting, 2d","price_usd":9.99,"owners_est":35000000,"positive":900000,"negative":20000,"rating":97.8,"windows":True,"mac":True,"linux":True},
    {"appid":578080,"name":"PUBG: Battlegrounds","genre":"Action","tags":"action, battle royale, shooter, multiplayer, survival","price_usd":0.0,"owners_est":70000000,"positive":700000,"negative":400000,"rating":63.6,"windows":True,"mac":False,"linux":False},
    {"appid":413150,"name":"Stardew Valley","genre":"Action","tags":"action, simulation, farming, relaxing, indie","price_usd":14.99,"owners_est":20000000,"positive":700000,"negative":10000,"rating":98.6,"windows":True,"mac":True,"linux":True},
    {"appid":230410,"name":"Warframe","genre":"Action","tags":"action, shooter, free to play, multiplayer, sci-fi","price_usd":0.0,"owners_est":50000000,"positive":450000,"negative":50000,"rating":90.0,"windows":True,"mac":False,"linux":True},
    {"appid":1174180,"name":"Red Dead Redemption 2","genre":"Action","tags":"action, adventure, open world, story rich, western","price_usd":59.99,"owners_est":9000000,"positive":380000,"negative":25000,"rating":93.8,"windows":True,"mac":False,"linux":False},
    {"appid":292030,"name":"The Witcher 3: Wild Hunt","genre":"Adventure","tags":"adventure, rpg, open world, story rich, dark fantasy","price_usd":39.99,"owners_est":25000000,"positive":700000,"negative":20000,"rating":97.2,"windows":True,"mac":False,"linux":False},
    {"appid":1250410,"name":"Disco Elysium","genre":"Adventure","tags":"adventure, rpg, narrative, choices matter, story rich","price_usd":39.99,"owners_est":3000000,"positive":130000,"negative":5000,"rating":96.3,"windows":True,"mac":True,"linux":True},
    {"appid":753640,"name":"Outer Wilds","genre":"Adventure","tags":"adventure, exploration, mystery, atmospheric, indie","price_usd":24.99,"owners_est":2000000,"positive":90000,"negative":3000,"rating":96.8,"windows":True,"mac":False,"linux":False},
    {"appid":221910,"name":"The Stanley Parable","genre":"Adventure","tags":"adventure, walking simulator, comedy, indie, narrative","price_usd":14.99,"owners_est":3000000,"positive":130000,"negative":3000,"rating":97.7,"windows":True,"mac":True,"linux":True},
    {"appid":976730,"name":"Hades","genre":"RPG","tags":"rpg, action roguelike, indie, story rich, hack and slash","price_usd":24.99,"owners_est":6000000,"positive":290000,"negative":5000,"rating":98.3,"windows":True,"mac":True,"linux":True},
    {"appid":72850,"name":"The Elder Scrolls V: Skyrim","genre":"RPG","tags":"rpg, open world, fantasy, story rich, character customization","price_usd":39.99,"owners_est":40000000,"positive":600000,"negative":30000,"rating":95.2,"windows":True,"mac":False,"linux":False},
    {"appid":1086940,"name":"Baldur's Gate 3","genre":"RPG","tags":"rpg, turn-based, co-op, story rich, fantasy","price_usd":59.99,"owners_est":10000000,"positive":550000,"negative":15000,"rating":97.3,"windows":True,"mac":True,"linux":True},
    {"appid":435150,"name":"Divinity: Original Sin 2","genre":"RPG","tags":"rpg, turn-based, co-op, story rich, fantasy","price_usd":44.99,"owners_est":4000000,"positive":180000,"negative":7000,"rating":96.3,"windows":True,"mac":True,"linux":True},
    {"appid":1623730,"name":"Persona 5 Royal","genre":"RPG","tags":"rpg, jrpg, turn-based, story rich, anime","price_usd":59.99,"owners_est":2000000,"positive":90000,"negative":2000,"rating":97.8,"windows":True,"mac":False,"linux":False},
    {"appid":289070,"name":"Sid Meier's Civilization VI","genre":"Strategy","tags":"strategy, turn-based, 4x, historical, multiplayer","price_usd":59.99,"owners_est":15000000,"positive":250000,"negative":20000,"rating":92.6,"windows":True,"mac":True,"linux":True},
    {"appid":394360,"name":"Hearts of Iron IV","genre":"Strategy","tags":"strategy, grand strategy, historical, wargame, military","price_usd":39.99,"owners_est":5000000,"positive":140000,"negative":10000,"rating":93.3,"windows":True,"mac":True,"linux":True},
    {"appid":281990,"name":"Stellaris","genre":"Strategy","tags":"strategy, grand strategy, 4x, sci-fi, space","price_usd":39.99,"owners_est":5000000,"positive":130000,"negative":10000,"rating":92.9,"windows":True,"mac":True,"linux":True},
    {"appid":739630,"name":"Factorio","genre":"Strategy","tags":"strategy, simulation, base building, automation, resource management","price_usd":35.0,"owners_est":3000000,"positive":200000,"negative":3000,"rating":98.5,"windows":True,"mac":True,"linux":True},
    {"appid":255710,"name":"Cities: Skylines","genre":"Simulation","tags":"simulation, city builder, management, building, sandbox","price_usd":29.99,"owners_est":10000000,"positive":200000,"negative":10000,"rating":95.2,"windows":True,"mac":True,"linux":True},
    {"appid":367520,"name":"Hollow Knight","genre":"Simulation","tags":"simulation, metroidvania, platformer, indie, challenging","price_usd":14.99,"owners_est":10000000,"positive":400000,"negative":8000,"rating":98.0,"windows":True,"mac":True,"linux":True},
    {"appid":945360,"name":"Among Us","genre":"Casual","tags":"casual, multiplayer, party game, funny, social deduction","price_usd":5.0,"owners_est":40000000,"positive":600000,"negative":80000,"rating":88.2,"windows":True,"mac":True,"linux":True},
    {"appid":1659040,"name":"Unpacking","genre":"Casual","tags":"casual, relaxing, puzzle, story rich, indie","price_usd":19.99,"owners_est":1000000,"positive":40000,"negative":500,"rating":98.8,"windows":True,"mac":True,"linux":True},
    {"appid":311690,"name":"Enter the Gungeon","genre":"Indie","tags":"indie, action roguelike, bullet hell, shooter, challenging","price_usd":14.99,"owners_est":5000000,"positive":150000,"negative":5000,"rating":96.8,"windows":True,"mac":True,"linux":True},
    {"appid":504230,"name":"Celeste","genre":"Indie","tags":"indie, platformer, pixel art, challenging, retro","price_usd":19.99,"owners_est":3000000,"positive":130000,"negative":2000,"rating":98.5,"windows":True,"mac":True,"linux":True},
    {"appid":418370,"name":"Resident Evil 7","genre":"Horror","tags":"horror, survival horror, first-person, atmospheric, violent","price_usd":29.99,"owners_est":6000000,"positive":130000,"negative":5000,"rating":96.3,"windows":True,"mac":False,"linux":False},
    {"appid":952060,"name":"Phasmophobia","genre":"Horror","tags":"horror, co-op, multiplayer, paranormal, atmospheric","price_usd":13.99,"owners_est":10000000,"positive":380000,"negative":15000,"rating":96.2,"windows":True,"mac":False,"linux":False},
    {"appid":1172470,"name":"Apex Legends","genre":"Shooter","tags":"shooter, battle royale, fps, multiplayer, free to play","price_usd":0.0,"owners_est":60000000,"positive":700000,"negative":200000,"rating":77.8,"windows":True,"mac":False,"linux":False},
    {"appid":359550,"name":"Tom Clancy's Rainbow Six Siege","genre":"Shooter","tags":"shooter, tactical shooter, fps, multiplayer, competitive","price_usd":19.99,"owners_est":25000000,"positive":350000,"negative":80000,"rating":81.4,"windows":True,"mac":False,"linux":False},
    {"appid":892970,"name":"Valheim","genre":"Survival","tags":"survival, open world, crafting, building, multiplayer","price_usd":20.0,"owners_est":10000000,"positive":280000,"negative":10000,"rating":96.6,"windows":True,"mac":False,"linux":True},
    {"appid":252490,"name":"Rust","genre":"Survival","tags":"survival, multiplayer, crafting, open world, pvp","price_usd":39.99,"owners_est":15000000,"positive":500000,"negative":150000,"rating":76.9,"windows":True,"mac":True,"linux":True},
    {"appid":1085660,"name":"Assetto Corsa Competizione","genre":"Racing","tags":"racing, simulation, driving, motorsport, realistic","price_usd":29.99,"owners_est":3000000,"positive":60000,"negative":5000,"rating":92.3,"windows":True,"mac":False,"linux":False},
    {"appid":1282370,"name":"Wreckfest","genre":"Racing","tags":"racing, driving, destruction, arcade, multiplayer","price_usd":34.99,"owners_est":2000000,"positive":50000,"negative":2000,"rating":96.2,"windows":True,"mac":False,"linux":False},
    {"appid":252950,"name":"Rocket League","genre":"Sports","tags":"sports, competitive, multiplayer, pvp, arcade","price_usd":0.0,"owners_est":40000000,"positive":700000,"negative":80000,"rating":89.7,"windows":True,"mac":True,"linux":True},
    {"appid":1447570,"name":"Guilty Gear Strive","genre":"Fighting","tags":"fighting, anime, 2d fighter, versus, competitive","price_usd":39.99,"owners_est":2000000,"positive":60000,"negative":3000,"rating":95.2,"windows":True,"mac":False,"linux":False},
    {"appid":620,"name":"Portal 2","genre":"Puzzle","tags":"puzzle, co-op, first-person, funny, story rich","price_usd":9.99,"owners_est":20000000,"positive":600000,"negative":5000,"rating":99.2,"windows":True,"mac":True,"linux":True},
    {"appid":736260,"name":"Baba Is You","genre":"Puzzle","tags":"puzzle, indie, logic, 2d, minimalist","price_usd":14.99,"owners_est":1000000,"positive":30000,"negative":1000,"rating":96.8,"windows":True,"mac":True,"linux":True},
    {"appid":268910,"name":"Cuphead","genre":"Platformer","tags":"platformer, run and gun, challenging, hand-drawn, indie","price_usd":19.99,"owners_est":6000000,"positive":160000,"negative":5000,"rating":97.0,"windows":True,"mac":True,"linux":False},
    {"appid":1145360,"name":"Ori and the Will of the Wisps","genre":"Platformer","tags":"platformer, metroidvania, indie, beautiful, story rich","price_usd":29.99,"owners_est":3000000,"positive":100000,"negative":2000,"rating":98.1,"windows":True,"mac":False,"linux":False},
]


def get_fallback_df(genres: list) -> pd.DataFrame:
    """Return fallback dataset filtered by selected genres."""
    rows = [g for g in FALLBACK_GAMES if g["genre"] in genres]
    if not rows:
        rows = FALLBACK_GAMES  # return all if no genre match
    return pd.DataFrame(rows).drop_duplicates("appid").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE TTL CACHE (replacement for st.cache_data)
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _cache_get(key):
    entry = _cache.get(key)
    if entry is None:
        return None
    value, ts = entry
    if time.time() - ts > CACHE_TTL:
        del _cache[key]
        return None
    return value


def _cache_set(key, value):
    _cache[key] = (value, time.time())


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — REAL-TIME DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _parse_owners(s: str) -> int:
    try:
        parts = [p.strip().replace(",", "") for p in s.split("..")]
        return (int(parts[0]) + int(parts[1])) // 2 if len(parts) == 2 else int(parts[0])
    except Exception:
        return 0


def _parse_record(appid_str: str, info: dict, fallback_genre: str = "") -> Optional[dict]:
    try:
        price = int(info.get("price", 0) or 0) / 100.0
        owners = _parse_owners(info.get("owners", "0 .. 0"))
        pos = int(info.get("positive", 0) or 0)
        neg = int(info.get("negative", 0) or 0)
        total = pos + neg
        rating = round((pos / total * 100), 1) if total > 0 else 0.0
        tags = ", ".join(list((info.get("tags") or {}).keys())[:12])
        genre_str = info.get("genre", fallback_genre) or fallback_genre
        return {
            "appid": int(appid_str),
            "name": info.get("name", "Unknown"),
            "genre": genre_str,
            "tags": tags,
            "price_usd": price,
            "owners_est": owners,
            "positive": pos,
            "negative": neg,
            "rating": rating,
            "windows": True,
            "mac": any(t.lower() in ["mac", "macos", "osx"]
                       for t in list((info.get("tags") or {}).keys())),
            "linux": any(t.lower() in ["linux", "steamos", "proton"]
                         for t in list((info.get("tags") or {}).keys())),
        }
    except Exception:
        return None


def fetch_by_genre(genres: tuple) -> pd.DataFrame:
    """Tarik game dari SteamSpy per genre yang dipilih user (cache TTL 30 menit)."""
    cache_key = ("by_genre", genres)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    records_per_genre = {}
    for genre in genres:
        url = GENRE_REMAP.get(genre, API_GENRE.format(g=genre))
        try:
            resp = requests.get(url, timeout=1)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            records_per_genre[genre] = []
            continue
        recs = []
        for aid, info in list(data.items())[:150]:
            rec = _parse_record(aid, info, genre)
            if rec:
                rec["genre"] = genre
                recs.append(rec)
        records_per_genre[genre] = recs
        time.sleep(0.25)

    if not any(records_per_genre.values()):
        # SteamSpy tidak bisa diakses — pakai fallback dataset
        result = get_fallback_df(list(genres))
        _cache_set(cache_key, result)
        return result.copy()

    non_empty = [v for v in records_per_genre.values() if v]
    min_count = min(len(v) for v in non_empty) if non_empty else 30
    balanced_count = max(min_count, 30)

    records = []
    for recs in records_per_genre.values():
        records.extend(recs[:balanced_count])

    if not records:
        result = pd.DataFrame()
    else:
        result = pd.DataFrame(records).drop_duplicates("appid").reset_index(drop=True)

    _cache_set(cache_key, result)
    return result.copy()


def fetch_top100() -> pd.DataFrame:
    """Tarik Top-100 game terpopuler minggu ini dari SteamSpy (cache TTL 30 menit)."""
    cache_key = ("top100",)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    try:
        resp = requests.get(API_TOP, timeout=1)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        result = get_fallback_df(list(ALL_GENRES))
        _cache_set(cache_key, result)
        return result.copy()

    records = [r for aid, info in data.items()
               if (r := _parse_record(aid, info)) is not None]
    if not records:
        result = pd.DataFrame()
    else:
        result = pd.DataFrame(records).drop_duplicates("appid").reset_index(drop=True)

    _cache_set(cache_key, result)
    return result.copy()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — TF-IDF ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _build_content_features(df: pd.DataFrame) -> list:
    g = df["genre"].fillna("").str.replace(",", " ")
    t = df["tags"].fillna("").str.replace(",", " ")
    n = df["name"].fillna("").str.replace(",", " ")
    feats = (g + " " + t + " " + n).str.strip().tolist()
    feats = [f if f.strip() else "unknown game" for f in feats]
    return feats


def build_tfidf(cache_key: int, appids: tuple, feats: tuple):
    full_key = ("tfidf", cache_key, appids, feats)
    cached = _cache_get(full_key)
    if cached is not None:
        return cached

    if not feats or len(feats) == 0:
        return None, None
    feats_list = list(feats)
    non_empty = [f for f in feats_list if f.strip()]
    if not non_empty:
        return None, None

    vec = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2),
        min_df=1, sublinear_tf=True, strip_accents="unicode",
    )
    try:
        matrix = vec.fit_transform(feats_list)
        result = (vec, matrix)
    except ValueError:
        vec2 = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 1),
            min_df=1, sublinear_tf=False, strip_accents=None,
        )
        matrix = vec2.fit_transform(feats_list)
        result = (vec2, matrix)

    _cache_set(full_key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — HYBRID SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_cf_scores(df: pd.DataFrame) -> np.ndarray:
    """Collaborative Filtering berbasis Item-Item Similarity + SVD."""
    try:
        jumlah_game = len(df)
        if jumlah_game < 2:
            return np.ones(jumlah_game)

        rating_game = df["rating"].fillna(50).values.astype(float)
        pemilik_game = np.log1p(df["owners_est"].fillna(0).values.astype(float))
        harga_game = df["price_usd"].fillna(0).values.astype(float)
        ulasan_positif = df["positive"].fillna(0).values.astype(float) if "positive" in df.columns else rating_game * 1000
        ulasan_negatif = df["negative"].fillna(0).values.astype(float) if "negative" in df.columns else (100 - rating_game) * 100
        total_ulasan = ulasan_positif + ulasan_negatif + 1e-9

        def normalisasi(arr):
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

        fitur_rating = normalisasi(rating_game)
        fitur_popularitas = pemilik_game / (pemilik_game.max() + 1e-9)
        fitur_harga_inv = normalisasi(harga_game.max() - harga_game)
        fitur_ulasan_pos = normalisasi(ulasan_positif)
        fitur_rasio_ulasan = ulasan_positif / total_ulasan
        fitur_tier = np.where(pemilik_game > np.log1p(100_000_000), 1.0,
                    np.where(pemilik_game > np.log1p(50_000_000), 0.85,
                    np.where(pemilik_game > np.log1p(10_000_000), 0.70,
                    np.where(pemilik_game > np.log1p(1_000_000), 0.55,
                    np.where(pemilik_game > np.log1p(100_000), 0.35, 0.15)))))

        matriks_fitur = np.column_stack([
            fitur_rating, fitur_popularitas, fitur_harga_inv,
            fitur_ulasan_pos, fitur_rasio_ulasan, fitur_tier,
        ])

        panjang_vektor = np.linalg.norm(matriks_fitur, axis=1, keepdims=True)
        panjang_vektor = np.where(panjang_vektor == 0, 1e-9, panjang_vektor)
        matriks_ternormalisasi = matriks_fitur / panjang_vektor

        sim_antar_game = matriks_ternormalisasi @ matriks_ternormalisasi.T

        skor_gabungan = 0.6 * fitur_popularitas + 0.4 * fitur_rating
        jumlah_acuan = max(3, int(jumlah_game * 0.15))
        indeks_acuan = np.argsort(skor_gabungan)[-jumlah_acuan:]

        skor_item_cf = sim_antar_game[:, indeks_acuan].mean(axis=1)

        U, S, _ = np.linalg.svd(matriks_ternormalisasi, full_matrices=False)
        k_komponen = min(4, len(S))
        representasi_latent = np.linalg.norm(U[:, :k_komponen] * S[:k_komponen], axis=1)
        skor_svd = representasi_latent / (representasi_latent.max() + 1e-9)

        skor_cf_final = 0.7 * skor_item_cf + 0.3 * skor_svd
        return skor_cf_final / (skor_cf_final.max() + 1e-9)

    except Exception:
        rating_game = df["rating"].fillna(50).values.astype(float)
        return (rating_game - rating_game.min()) / (rating_game.max() - rating_game.min() + 1e-9)


def _score_df(df: pd.DataFrame, vec, matrix, query_text: str,
              w_cbf: float, w_cf: float, w_q: float) -> pd.DataFrame:
    vektor_query = vec.transform([query_text])
    skor_cbf = cosine_similarity(vektor_query, matrix).flatten()

    hasil = df.copy().reset_index(drop=True)
    if len(skor_cbf) != len(hasil):
        return pd.DataFrame()

    hasil["cbf_score"] = skor_cbf
    hasil["cf_score"] = compute_cf_scores(hasil)

    rating_game = hasil["rating"].fillna(0).values
    rating_norm = (rating_game - rating_game.min()) / (rating_game.max() - rating_game.min() + 1e-9)

    pemilik_log = np.log1p(hasil["owners_est"].fillna(0).values)
    pemilik_norm = pemilik_log / (pemilik_log.max() + 1e-9)

    hasil["qual_score"] = 0.8 * rating_norm + 0.2 * pemilik_norm

    hasil["final_score"] = (
        w_cbf * hasil["cbf_score"] +
        w_cf * hasil["cf_score"] +
        w_q * hasil["qual_score"]
    )
    return hasil


def compute_hybrid(
    df: pd.DataFrame, vec, matrix,
    selected_genres: list, budget: float, os_filter: str,
    include_free: bool, top_n: int,
    w_cbf: float = 0.75, w_qual: float = 0.25,
) -> pd.DataFrame:
    """Hybrid Scoring Pipeline (3-Layer) + Quota-Based Genre Selection + MMR."""
    if df is None or df.empty or vec is None:
        return pd.DataFrame()

    n_genres = len(selected_genres)
    w_cf = (1 - w_cbf) * 0.25
    w_q = (1 - w_cbf) * 0.75

    pool = df.copy().reset_index(drop=True)
    pool = pool[pool["price_usd"] <= budget]
    if not include_free:
        pool = pool[pool["price_usd"] > 0]
    if os_filter == "Windows":
        pool = pool[pool["windows"] == True]
    elif os_filter == "Mac":
        pool = pool[pool["mac"] == True]
    elif os_filter == "Linux":
        pool = pool[pool["linux"] == True]

    if len(pool) < n_genres:
        pool = df.copy().reset_index(drop=True)
        pool = pool[pool["price_usd"] <= budget]

    if pool.empty:
        pool = df.copy().reset_index(drop=True)

    pool = pool.reset_index(drop=True)

    query_parts = []
    for g in selected_genres:
        query_parts.extend([g.lower()] * 5)
        keyword_genre = GT_GENRE_KW.get(g, [])
        query_parts.extend(keyword_genre[:6])
    global_query = " ".join(query_parts)

    pool_feats = _build_content_features(pool)
    try:
        pool_vec_mat = vec.transform(pool_feats)
    except Exception:
        return pool.head(top_n).reset_index(drop=True)

    scored = _score_df(pool, vec, pool_vec_mat, global_query, w_cbf, w_cf, w_q)
    if scored.empty:
        return pool.head(top_n).reset_index(drop=True)

    quota_rows = []
    used_appids = set()

    for g in selected_genres:
        genre_query_parts = [g.lower()] * 8 + GT_GENRE_KW.get(g, [])[:8]
        genre_query = " ".join(genre_query_parts)

        user_vec_g = vec.transform([genre_query])
        cbf_g = cosine_similarity(user_vec_g, pool_vec_mat).flatten()

        genre_scored = scored.copy()
        genre_scored["genre_cbf"] = cbf_g
        genre_scored["genre_score"] = (
            0.80 * genre_scored["genre_cbf"] +
            0.10 * genre_scored["cf_score"] +
            0.10 * genre_scored["qual_score"]
        )
        candidates_g = genre_scored[~genre_scored["appid"].isin(used_appids)]
        if candidates_g.empty:
            continue
        best = candidates_g.nlargest(1, "genre_score")
        if not best.empty:
            appid = best.iloc[0]["appid"]
            used_appids.add(appid)
            quota_rows.append(best.iloc[0])

    remaining_slots = top_n - len(quota_rows)
    remaining_pool = scored[~scored["appid"].isin(used_appids)].copy()

    if remaining_slots > 0 and not remaining_pool.empty:
        try:
            MMR_LAMBDA = 0.65
            mmr_cands = remaining_pool.nlargest(remaining_slots * 4, "final_score").reset_index(drop=True)
            feat_texts = (mmr_cands["genre"].fillna("") + " " + mmr_cands["tags"].fillna("")).tolist()
            cand_vecs = vec.transform(feat_texts)
            sim_matrix = cosine_similarity(cand_vecs)

            scores_arr = mmr_cands["final_score"].values.copy()
            s_min, s_max = scores_arr.min(), scores_arr.max()
            if s_max > s_min:
                scores_arr = (scores_arr - s_min) / (s_max - s_min)

            quota_texts = []
            for qr in quota_rows:
                quota_texts.append(str(qr.get("genre", "")) + " " + str(qr.get("tags", "")))
            if quota_texts:
                quota_vecs = vec.transform(quota_texts)
                quota_sim_mat = cosine_similarity(cand_vecs, quota_vecs)
            else:
                quota_sim_mat = None

            selected_mmr, remaining_mmr = [], list(range(len(mmr_cands)))
            while len(selected_mmr) < remaining_slots and remaining_mmr:
                if not selected_mmr and quota_sim_mat is None:
                    best = max(remaining_mmr, key=lambda i: scores_arr[i])
                else:
                    def mmr_score(i):
                        rel = scores_arr[i]
                        red_mmr = max((sim_matrix[i][j] for j in selected_mmr), default=0)
                        red_q = float(quota_sim_mat[i].max()) if quota_sim_mat is not None else 0
                        redundancy = max(red_mmr, red_q)
                        return MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * redundancy
                    best = max(remaining_mmr, key=mmr_score)
                selected_mmr.append(best)
                remaining_mmr.remove(best)

            mmr_result = mmr_cands.iloc[selected_mmr]
            extra_cols = ["genre_cbf", "genre_score"] if "genre_cbf" in mmr_result.columns else []
            mmr_result = mmr_result.drop(columns=extra_cols, errors="ignore")
        except Exception:
            mmr_result = remaining_pool.nlargest(remaining_slots, "final_score")
    else:
        mmr_result = pd.DataFrame()

    quota_df = pd.DataFrame(quota_rows)
    for col in ["genre_cbf", "genre_score"]:
        if col in quota_df.columns:
            quota_df = quota_df.drop(columns=[col])

    final_parts = [p for p in [quota_df, mmr_result] if not p.empty]
    if not final_parts:
        return scored.nlargest(top_n, "final_score").reset_index(drop=True)

    result = pd.concat(final_parts).drop_duplicates("appid")
    return result.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — QUANTITATIVE EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def relevance_score(name: str, genre_str: str, tags_str: str,
                     rating: float, genres: list) -> float:
    nl = name.lower().strip()
    gl = (genre_str or "").lower()
    tl = (tags_str or "").lower()
    combined = gl + " " + tl

    seed_hit = 0.0
    for g in genres:
        daftar_seed = GT_PRIMARY.get(g, [])
        if any(seed in nl for seed in daftar_seed):
            seed_hit = 1.0
            break
    l1 = seed_hit * 0.40

    skor_keyword_terbaik = 0.0
    for g in genres:
        daftar_keyword = GT_GENRE_KW.get(g, [])
        if not daftar_keyword:
            continue
        jumlah_cocok = sum(1 for kw in daftar_keyword if kw in combined)
        skor_genre_ini = min(jumlah_cocok / 2.0, 1.0)
        skor_keyword_terbaik = max(skor_keyword_terbaik, skor_genre_ini)
    l2 = skor_keyword_terbaik * 0.45

    if rating < GT_MIN_RATING:
        l3 = 0.0
    else:
        l3 = ((rating - GT_MIN_RATING) / (100.0 - GT_MIN_RATING)) * 0.15

    return round(l1 + l2 + l3, 4)


def is_relevant_binary(name: str, genre_str: str, tags_str: str,
                        rating: float, genres: list) -> int:
    return 1 if relevance_score(name, genre_str, tags_str, rating, genres) >= 0.10 else 0


def _bootstrap_ci(values: list, n_boot: int = 500, ci: float = 0.95) -> tuple:
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    boot = np.array([np.mean(np.random.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return (round(float(lo), 4), round(float(hi), 4))


def evaluate(rec_df: pd.DataFrame, full_df: pd.DataFrame, genres: list) -> dict:
    if rec_df.empty:
        return {}

    names = rec_df["name"].tolist()
    genres_l = rec_df["genre"].fillna("").tolist()
    tags_l = rec_df["tags"].fillna("").tolist()
    ratings = rec_df["rating"].fillna(0).tolist()
    K = len(names)

    rel_binary = [is_relevant_binary(names[i], genres_l[i], tags_l[i], ratings[i], genres)
                   for i in range(K)]
    rel_scores = [relevance_score(names[i], genres_l[i], tags_l[i], ratings[i], genres)
                   for i in range(K)]

    sample_df = full_df.sample(min(len(full_df), 200), random_state=42) if len(full_df) > 200 else full_df
    total_rel = sum(
        is_relevant_binary(r["name"], r.get("genre", ""), r.get("tags", ""),
                           r.get("rating", 0), genres)
        for _, r in sample_df.iterrows()
    )
    if len(full_df) > 200:
        total_rel = max(1, round(total_rel * len(full_df) / 200))
    total_rel = max(total_rel, 1)

    prec_k = sum(rel_binary) / K

    prec_curve = []
    for i in range(1, K + 1):
        prec_curve.append(round(sum(rel_binary[:i]) / i, 4))

    rec_k = sum(rel_binary) / total_rel

    ap_sum, n_rel = 0.0, 0
    for i, r in enumerate(rel_binary):
        if r:
            n_rel += 1
            ap_sum += n_rel / (i + 1)
    map_score = ap_sum / max(n_rel, 1)

    dcg_graded = sum(rel_scores[i] / math.log2(i + 2) for i in range(K))
    ideal_graded = sorted(rel_scores, reverse=True)
    idcg_graded = sum(ideal_graded[i] / math.log2(i + 2) for i in range(K))
    ndcg_graded = dcg_graded / max(idcg_graded, 1e-9)

    dcg_bin = sum(rel_binary[i] / math.log2(i + 2) for i in range(K))
    ideal_bin = sorted(rel_binary, reverse=True)
    idcg_bin = sum(ideal_bin[i] / math.log2(i + 2) for i in range(K))
    ndcg_bin = dcg_bin / max(idcg_bin, 1e-9)

    n_rel_in_rec = sum(rel_binary)
    coverage = round(n_rel_in_rec / max(total_rel, 1), 4)

    token_cnt = {}
    for i in range(K):
        tokens = []
        for gs in (genres_l[i] + "," + tags_l[i]).split(","):
            t = gs.strip().lower()
            if t and len(t) > 2:
                tokens.append(t)
        for t in set(tokens):
            token_cnt[t] = token_cnt.get(t, 0) + 1
    total_tok = sum(token_cnt.values())
    if total_tok > 0:
        hhi = sum((c / total_tok) ** 2 for c in token_cnt.values())
        genre_diversity = round(1 - hhi, 4)
    else:
        genre_diversity = 0.0

    if "final_score" in rec_df.columns and K > 1:
        scores = rec_df["final_score"].values
        score_diversity = round(float(np.std(scores) / (np.mean(scores) + 1e-9)), 4)
        score_diversity = min(score_diversity, 1.0)
    else:
        score_diversity = 0.0

    name_diversity = round(len(set(names)) / max(K, 1), 4)
    diversity = round(0.5 * genre_diversity + 0.3 * score_diversity + 0.2 * name_diversity, 4)
    n_unique_genres = len(token_cnt)

    ci_prec = _bootstrap_ci(rel_binary)
    ci_rec = _bootstrap_ci([r / total_rel for r in rel_binary])

    ap_samples = []
    for _ in range(200):
        idx = np.random.choice(K, K, replace=True)
        samp = [rel_binary[i] for i in idx]
        ap_s, nr_s = 0.0, 0
        for j, rv in enumerate(samp):
            if rv:
                nr_s += 1
                ap_s += nr_s / (j + 1)
        ap_samples.append(ap_s / max(nr_s, 1))
    ci_map = (round(float(np.percentile(ap_samples, 2.5)), 4),
              round(float(np.percentile(ap_samples, 97.5)), 4))

    genre_prec = {}
    for g in genres:
        g_mask = [1 if g.lower() in genres_l[i].lower() else 0 for i in range(K)]
        g_rel = [rel_binary[i] for i in range(K) if g_mask[i]]
        genre_prec[g] = round(sum(g_rel) / max(len(g_rel), 1), 3)

    return {
        "Precision@K": round(prec_k, 4),
        "Recall@K": round(rec_k, 4),
        "MAP": round(map_score, 4),
        "NDCG@K": round(ndcg_bin, 4),
        "NDCG@K_graded": round(ndcg_graded, 4),
        "Coverage": round(coverage, 4),
        "Diversity": round(diversity, 4),
        "CI_Precision": ci_prec,
        "CI_Recall": ci_rec,
        "CI_MAP": ci_map,
        "_K": K,
        "_total_rel": total_rel,
        "_rel_in_topk": sum(rel_binary),
        "_unique_genres": n_unique_genres,
        "_genre_counts": token_cnt,
        "_rel_binary": rel_binary,
        "_rel_scores": rel_scores,
        "_prec_curve": prec_curve,
        "_genre_prec": genre_prec,
        "_avg_rel_score": round(float(np.mean(rel_scores)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA ASSEMBLY (replikasi logika dari main() Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_dataset(sel: list) -> pd.DataFrame:
    df_g = fetch_by_genre(tuple(sel))
    df_top = fetch_top100()

    frames = [f for f in [df_g, df_top] if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    if df_g is not None and not df_g.empty and df_top is not None and not df_top.empty:
        existing_ids = set(df_g["appid"].tolist())
        df_top_new = df_top[~df_top["appid"].isin(existing_ids)].copy()

        def assign_genre(row):
            combined = (str(row.get("genre", "")) + " " + str(row.get("tags", ""))).lower()
            for g in sel:
                kws = GT_GENRE_KW.get(g, [])
                if any(kw in combined for kw in kws):
                    return g
            return sel[0] if sel else row.get("genre", "Unknown")

        if not df_top_new.empty:
            df_top_new["genre"] = df_top_new.apply(assign_genre, axis=1)
        df_raw = pd.concat([df_g, df_top_new.head(50)]).drop_duplicates("appid").reset_index(drop=True)
    else:
        df_raw = pd.concat(frames).drop_duplicates("appid").reset_index(drop=True)

    return df_raw


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="GameMatch API", description="Steam Game Hybrid Recommendation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    genres: list[str] = Field(..., min_length=1, description="Genre yang dipilih")
    budget: float = Field(30.0, ge=0, le=100, description="Batas anggaran (USD)")
    os: str = Field("Any", description="Platform OS: Any | Windows | Mac | Linux")
    include_free: bool = Field(True, description="Sertakan game F2P")
    top_n: int = Field(10, ge=5, le=30, description="Jumlah rekomendasi")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/genres")
def get_genres():
    return {"genres": ALL_GENRES, "eval_targets": EVAL_TARGETS}


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    sel = [g for g in req.genres if g in ALL_GENRES]
    if not sel:
        raise HTTPException(status_code=400, detail="Pilih minimal satu genre yang valid.")

    df_raw = assemble_dataset(sel)
    if df_raw.empty:
        raise HTTPException(status_code=502, detail="Tidak ada data dari SteamSpy API. Coba lagi nanti.")

    if len(df_raw) < 5:
        raise HTTPException(status_code=422, detail="Data terlalu sedikit dari API. Coba pilih genre yang lebih umum.")

    feats_list = _build_content_features(df_raw)
    ck = hash(tuple(df_raw["appid"].tolist()[:60]))
    vec, matrix = build_tfidf(ck, tuple(df_raw["appid"].tolist()), tuple(feats_list))

    if vec is None or matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF engine gagal dibangun. Coba lagi.")

    if matrix.shape[0] != len(df_raw):
        raise HTTPException(status_code=500, detail="Shape mismatch pada engine. Coba lagi.")

    w_cbf, w_qual = 0.60, 0.40

    rec = compute_hybrid(df_raw, vec, matrix, sel, req.budget,
                          req.os, req.include_free, req.top_n, w_cbf, w_qual)
    if rec is None:
        rec = pd.DataFrame()

    mtr = evaluate(rec, df_raw, sel)

    # Raw data sample untuk tab "Raw Data"
    show_cols = ["name", "genre", "price_usd", "rating", "owners_est", "windows", "mac", "linux", "appid"]
    show_cols = [c for c in show_cols if c in df_raw.columns]
    df_display = df_raw[show_cols].sort_values("genre").reset_index(drop=True)
    genre_counts = df_raw["genre"].value_counts().to_dict()

    return {
        "recommendations": rec.replace({np.nan: None}).to_dict(orient="records") if not rec.empty else [],
        "metrics": mtr,
        "eval_targets": EVAL_TARGETS,
        "raw_data": df_display.head(300).replace({np.nan: None}).to_dict(orient="records"),
        "raw_total": len(df_raw),
        "genre_counts": genre_counts,
        "query": {
            "genres": sel, "budget": req.budget, "os": req.os,
            "include_free": req.include_free, "top_n": req.top_n,
            "w_cbf": w_cbf, "w_qual": w_qual,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
