"""Extract the Fallout: New Vegas radio stations (playlists, DJ lines with subtitles and transmitter
ranges) from FalloutNV.esm and the game archives into <site>/radio for the map's radio tab.
"""
import argparse
import json
import os
import re
import shutil
import struct

import fnv_terrain as ft
from bsa import BSA
from build_cesium import Placement

# (quest editor ID, talking-activator editor ID, short frequency label)
STATIONS = [
    ("RadioNewVegas", "RadioNewVegasTA", "RNV"),
    ("vCountryRadioQuest", "vCountryRadio", "MMR"),
    ("RadioBlackMountain", "RadioBlackMountainTA", "BMR"),
    ("vGomorrahRadioQuest", "vCasinoGomorrahRadio", "GOM"),
    ("vUltraLuxeRadioQuest", "vCasinoUltraLuxeRadio", "ULX"),
    ("TopsRadioQuest", "vCasinoTopsRadio", "TOP"),
]
# Mr. New Vegas segments: dialogue topic -> role in the programme
DJ_TOPICS = {"RNVMusicIntroGeneric": "intro", "RNVNewsIntro": "news_intro", "RNVNewsStory": "news",
             "RNVNewsStory2": "news", "RNVNewsStory3": "news", "RNVNewsPost": "news_post",
             "RNVNewsTransition": "transition"}
TITLES = {"eddyarnold_rca_itsasin": "It's a Sin", "flower_duet_lakm_kpm": "Flower Duet (Lakmé)",
          "piano_concerto_no_21__elvira_madigan": "Piano Concerto No. 21",
          "four_seasons_no_4_the_winter": "The Four Seasons: Winter",
          "concerto_for_2_vl_str_in_d_minor": "Concerto for 2 Violins in D Minor",
          "concerto_grosso_in_b_minor_allegro_01": "Concerto Grosso in B Minor I",
          "concerto_grosso_in_b_minor_allegro_02": "Concerto Grosso in B Minor II",
          "von_spanien_nach_s_damerika": "Von Spanien nach Südamerika", "i_m_movin_out": "I'm Movin' Out",
          "i_m_so_blue": "I'm So Blue", "goin_under": "Goin' Under", "its_a_sin_to_tell_a_lie": "It's a Sin to Tell a Lie", "let_s_ride_into_the_sunset_together": "Let's Ride into the Sunset Together",
          "aint_that_a_kick_in_the_head": "Ain't That a Kick in the Head", "homeonthewastes": "Home on the Wastes",
          "newvegasvalley": "New Vegas Valley", "streetsofnewreno": "Streets of New Reno",
          "somethings_gotta_give": "Something's Gotta Give", "why_dont_you_do_right": "Why Don't You Do Right",
          "jazz_blues_gt": "Jazz Blues", "jazz_club_blues_cas": "Jazz Club Blues", "joe_cool_cas": "Joe Cool",
          "sleepy_town_blues_cas": "Sleepy Town Blues", "slow_sax_kos": "Slow Sax"}


SMALL_WORDS = {"a", "an", "and", "as", "by", "for", "in", "into", "of", "the", "to"}


def song_title(path):
    stem = re.sub(r"^mus_", "", os.path.splitext(os.path.basename(path))[0].lower())
    if stem in TITLES:
        return TITLES[stem]
    words = stem.replace("_", " ").split()
    return " ".join(w if i and w in SMALL_WORDS else w.capitalize() for i, w in enumerate(words))


def scan(path):
    buf = memoryview(open(path, "rb").read())
    sounds, quests, tacts, dials, infos, radio_refs, wnames = {}, {}, {}, {}, [], [], {}

    def walk(start, end, world, dial):
        pos = start
        while pos < end:
            typ = bytes(buf[pos:pos + 4])
            if typ == b"GRUP":
                _, gsize, label, gtype, *_ = ft.GRP.unpack_from(buf, pos)
                lab = struct.unpack("<I", label)[0]
                walk(pos + 24, pos + gsize, lab if gtype == 1 else world, lab if gtype == 7 else dial)
                pos += gsize
                continue
            _, size, flags, fid, *_ = ft.REC.unpack_from(buf, pos)
            if typ in (b"SOUN", b"QUST", b"TACT", b"DIAL", b"INFO", b"REFR", b"WRLD"):
                sub = list(ft.subrecords(ft.record_data(buf, pos + 24, size, flags)))
                d = dict(sub)
                edid = ft.zstring(d.get(b"EDID", b""))
                if typ == b"SOUN" and b"FNAM" in d:
                    sounds[fid] = ft.zstring(d[b"FNAM"])
                elif typ == b"QUST":
                    quests[fid] = edid
                elif typ == b"WRLD":
                    wnames[fid] = edid
                elif typ == b"TACT":
                    tacts[fid] = (edid, ft.zstring(d.get(b"FULL", b"")))
                elif typ == b"DIAL":
                    dials[fid] = edid
                elif typ == b"INFO" and b"QSTI" in d:
                    infos.append((fid, dial, struct.unpack("<I", d[b"QSTI"])[0], sub))
                elif typ == b"REFR" and b"XRDO" in d:
                    radius, kind, static, _ = struct.unpack_from("<fIfI", d[b"XRDO"])
                    radio_refs.append((struct.unpack("<I", d[b"NAME"])[0], wnames.get(world),
                                       struct.unpack_from("<3f", d[b"DATA"]), radius, kind, static))
            pos += 24 + size

    walk(24 + ft.REC.unpack_from(buf, 0)[1], len(buf), None, None)
    return sounds, quests, tacts, dials, infos, radio_refs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="game Data directory")
    ap.add_argument("-o", "--out", default="site/radio")
    ap.add_argument("--lat", type=float, default=36.17)
    ap.add_argument("--lon", type=float, default=-115.14)
    a = ap.parse_args()

    place = Placement(a.lat, a.lon)
    sounds, quests, tacts, dials, infos, radio_refs = scan(os.path.join(a.data, "FalloutNV.esm"))
    voices = BSA(os.path.join(a.data, "Fallout - Voices1.bsa"))
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    copied = set()

    def song(path):
        name = os.path.basename(path).lower()
        if name not in copied:
            shutil.copyfile(os.path.join(a.data, "Sound", path), os.path.join(a.out, "audio", name))
            copied.add(name)
        return {"file": "audio/" + name, "title": song_title(name)}

    stations = []
    for quest, tact, freq in STATIONS:
        qid = next(k for k, v in quests.items() if v == quest)
        tid, (_, name) = next((k, v) for k, v in tacts.items() if v[0] == tact)
        songs, dj = {}, []
        for fid, dial, q, sub in infos:
            if q != qid:
                continue
            responses = []
            trdt = None
            for t, v in sub:
                if t == b"TRDT":
                    trdt = bytes(v)
                elif t == b"NAM1" and trdt is not None:
                    responses.append((trdt[12], struct.unpack_from("<I", trdt, 16)[0], ft.zstring(v)))
                    trdt = None
            for number, snd, text in responses:
                path = sounds.get(snd, "")
                if path.lower().startswith("songs\\") and os.path.exists(os.path.join(a.data, "Sound", path)):
                    songs[path.lower()] = song(path)
            role = DJ_TOPICS.get(dials.get(dial))
            if role:
                lines = []
                for number, _, text in responses:
                    key = (f"sound\\voice\\falloutnv.esm\\maleuniquemrnewvegas\\"
                           f"{quest.lower()}_{dials[dial].lower()}_{fid:08x}_{number}.ogg")
                    if key in voices.files:
                        name_ogg = os.path.basename(key)
                        with open(os.path.join(a.out, "audio", name_ogg), "wb") as f:
                            f.write(voices.read(key))
                        lines.append({"file": "audio/" + name_ogg, "text": text})
                if lines:
                    dj.append({"role": role, "lines": lines})
        st = {"id": freq.lower(), "freq": freq, "name": name, "songs": sorted(songs.values(), key=lambda s: s["title"]),
              "dj": dj, "range": None}
        for base, world, pos, radius, kind, static in radio_refs:
            if base == tid and world == "WastelandNV" and kind == 0 and radius > 0:
                lon, lat = place.to_lonlat(pos[0] * ft.UNIT_M, pos[1] * ft.UNIT_M)
                st["range"] = {"lon": round(float(lon), 7), "lat": round(float(lat), 7),
                               "radius": round(radius * ft.UNIT_M, 1), "static": static / 100}
        stations.append(st)
        print(f"{name:24s} songs={len(st['songs']):3d} dj segments={len(dj):3d} range={st['range']}")

    with open(os.path.join(a.out, "radio.json"), "w", encoding="utf-8") as f:
        json.dump({"stations": stations}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
