"""Extract the Fallout: New Vegas world-map icons, tinted like the Pip-Boy, into <site>/icons as PNGs
named after the marker types used in world.json (e.g. "Natural landmark" -> natural_landmark.png).
"""
import argparse
import os

import numpy as np
from osgeo import gdal

from bsa import BSA
from lod_texture import read_dds

gdal.UseExceptions()

FOLDER = "textures\\interface\\icons\\"
# marker type (world.json) -> game icon
ICONS = {"City": "world map\\icon_map_city", "Settlement": "world map\\icon_map_settlement",
         "Encampment": "world map\\icon_map_encampment", "Natural landmark": "world map\\icon_map_natural_landmark",
         "Cave": "world map\\icon_map_cave", "Factory": "world map\\icon_map_factory",
         "Monument": "world map\\icon_map_monument", "Military": "world map\\icon_map_military",
         "Office": "world map\\icon_map_office", "Town ruins": "world map\\icon_map_ruins_town",
         "Urban ruins": "world map\\icon_map_ruins_urban", "Sewer ruins": "world map\\icon_map_ruins_sewer",
         "Metro": "world map\\icon_map_metro", "Vault": "world map\\icon_map_vault",
         "Entrance": "local map\\icon_local_door", "Undiscovered": "world map\\icon_map_undiscovered"}
AMBER = np.array([255, 182, 66], np.float32)     # Pip-Boy HUD colour
INK = np.array([20, 12, 2], np.float32)          # the symbol cut out of the amber square


def slug(marker_type):
    return marker_type.lower().replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="game Data directory")
    ap.add_argument("--textures", default="Fallout - Textures2.bsa")
    ap.add_argument("-o", "--out", default="site/icons")
    a = ap.parse_args()

    tex = BSA(os.path.join(a.data, a.textures))
    os.makedirs(a.out, exist_ok=True)
    png = gdal.GetDriverByName("PNG")
    for marker_type, icon in ICONS.items():
        img = read_dds(tex.read(FOLDER + icon + ".dds")).astype(np.float32)
        lum = img[..., :3].mean(axis=2, keepdims=True) / 255
        if img.shape[2] == 4 and img[..., 3].min() < 255:
            # white square with the symbol in black: amber square, dark symbol
            rgba = np.concatenate([AMBER * lum + INK * (1 - lum), img[..., 3:4]], axis=2)
        else:
            # no alpha (city): drawn additively in game, so black is transparent and white is the symbol
            rgba = np.concatenate([np.broadcast_to(AMBER, lum.shape[:2] + (3,)), lum * 255], axis=2)
        rgba = np.clip(np.round(rgba), 0, 255).astype(np.uint8)
        mem = gdal.GetDriverByName("MEM").Create("", rgba.shape[1], rgba.shape[0], 4, gdal.GDT_Byte)
        for i in range(4):
            mem.GetRasterBand(i + 1).WriteArray(rgba[..., i])
        png.CreateCopy(os.path.join(a.out, slug(marker_type) + ".png"), mem)
    print(f"icons: {len(ICONS)} -> {a.out}")


if __name__ == "__main__":
    main()
