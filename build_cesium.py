"""Build a static CesiumJS site for a Fallout 3 / New Vegas worldspace:
terrain tiles (Cesium heightmap-1.0), imagery tiles (game LOD textures shaded with the DEM)
and map markers. Serve the output directory with any static web server.
"""
import argparse
import json
import math
import os
from types import SimpleNamespace

import numpy as np
from osgeo import gdal

from fnv_terrain import CELL_UNITS, STEP, UNIT_M, mosaic, parse_worlds
from lod_texture import lod_mosaic

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

R = 6378137.0
TERRAIN_SIZE = 65
IMAGE_SIZE = 256
WATER_MASK_SIZE = 256
WATER_RGB = np.array([44, 74, 80], np.float32)
MARKER_TYPES = ["None", "City", "Settlement", "Encampment", "Natural landmark", "Cave", "Factory",
                "Monument", "Military", "Office", "Town ruins", "Urban ruins", "Sewer ruins",
                "Metro", "Vault"]


class Placement:
    """Local tangent-plane placement of game metres around (lat0, lon0)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = R * math.cos(math.radians(lat0))

    def to_lonlat(self, xm, ym):
        return self.lon0 + np.degrees(xm / self.kx), self.lat0 + np.degrees(ym / R)

    def to_metres(self, lon, lat):
        return np.radians(lon - self.lon0) * self.kx, np.radians(lat - self.lat0) * R


def pyramid(a, levels):
    """Block-mean pyramid; works for (h, w) float or (h, w, c) uint8 arrays."""
    out = [a]
    for _ in range(levels):
        p = out[-1]
        if min(p.shape[:2]) < 4:
            break                         # small rasters (e.g. city worldspaces) need fewer levels
        h, w = p.shape[0] // 2 * 2, p.shape[1] // 2 * 2
        p = p[:h, :w]
        acc = p.astype(np.float32 if p.dtype != np.uint8 else np.uint16)
        q = (acc[0::2, 0::2] + acc[1::2, 0::2] + acc[0::2, 1::2] + acc[1::2, 1::2]) / 4
        out.append(q.astype(a.dtype))
    return out


def bilinear(img, fr, fc):
    """Sample img at fractional row/col (clamped to the edges)."""
    h, w = img.shape[:2]
    fr = np.clip(fr, 0, h - 1)
    fc = np.clip(fc, 0, w - 1)
    r0 = np.minimum(fr.astype(np.int64), h - 2)
    c0 = np.minimum(fc.astype(np.int64), w - 2)
    dr, dc = fr - r0, fc - c0
    if img.ndim == 3:
        dr, dc = dr[..., None], dc[..., None]
    a = img[r0, c0].astype(np.float32)
    b = img[r0, c0 + 1].astype(np.float32)
    c = img[r0 + 1, c0].astype(np.float32)
    d = img[r0 + 1, c0 + 1].astype(np.float32)
    return (a * (1 - dc) + b * dc) * (1 - dr) + (c * (1 - dc) + d * dc) * dr


class Grid:
    """A raster pyramid addressed in game metres; (left, top) is the centre of pixel (0, 0)."""

    def __init__(self, img, left, top, px, levels):
        self.pyr = pyramid(img, levels)
        self.left, self.top, self.px = left, top, px

    def sample(self, xm, ym, spacing_m):
        p = int(np.clip(math.floor(math.log2(max(spacing_m / self.px, 1.0))), 0, len(self.pyr) - 1))
        s = 2 ** p
        fc = ((xm - self.left) / self.px - (s - 1) / 2) / s
        fr = ((self.top - ym) / self.px - (s - 1) / 2) / s
        return bilinear(self.pyr[p], fr, fc)


def tile_range(rect, level):
    """Geographic-scheme tiles at `level` intersecting rect=(w,s,e,n); y counted from the south."""
    size = 180.0 / 2 ** level
    w, s, e, n = rect
    xs = range(int((w + 180) // size), int((e + 180) // size) + 1)
    ys = range(int((s + 90) // size), int((n + 90) // size) + 1)
    return size, {(x, y) for x in xs for y in ys}


def water_mask_bytes(water, west, north, size, spacing):
    """Cesium water mask for one tile: 1 byte (all land/all water) or 256x256, rows north->south."""
    k = (np.arange(WATER_MASK_SIZE) + 0.5) * size / WATER_MASK_SIZE
    lon, lat = np.meshgrid(west + k, north - k)
    m = water.grid.sample(*water.place.to_metres(lon, lat), spacing) > 0.5
    if not m.any():
        return bytes([0])
    if m.all():
        return bytes([255])
    return (m.astype(np.uint8) * 255).tobytes()


def build_terrain(dem, water, place, rect, out, max_level):
    tiles = {}
    for level in range(max_level + 1):
        size, t = tile_range(rect, level)
        if level == 0:
            t |= {(0, 0), (1, 0)}
        tiles[level] = t
    lat_mid = math.radians(place.lat0)
    count = 0
    for level in range(max_level + 1):
        size = 180.0 / 2 ** level
        spacing = size / (TERRAIN_SIZE - 1) * math.pi / 180 * R * math.cos(lat_mid)
        for tx, ty in tiles[level]:
            west, north = -180 + tx * size, -90 + (ty + 1) * size
            k = np.arange(TERRAIN_SIZE) * size / (TERRAIN_SIZE - 1)
            lon, lat = np.meshgrid(west + k, north - k)
            xm, ym = place.to_metres(lon, lat)
            h = dem.sample(xm, ym, spacing)
            enc = np.clip(np.round((h + 1000.0) * 5.0), 0, 65535).astype("<u2")
            mask = 0
            if level < max_level:
                for bit, (i, j) in zip((1, 2, 4, 8), ((0, 0), (1, 0), (0, 1), (1, 1))):
                    if (2 * tx + i, 2 * ty + j) in tiles[level + 1]:
                        mask |= bit
            d = os.path.join(out, str(level), str(tx))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{ty}.terrain"), "wb") as f:
                f.write(enc.tobytes() + bytes([mask]) + water_mask_bytes(water, west, north, size, spacing))
            count += 1
    with open(os.path.join(out, "layer.json"), "w") as f:
        json.dump({"tilejson": "2.1.0", "format": "heightmap-1.0", "version": "1.0.0", "scheme": "tms",
                   "projection": "EPSG:4326", "bounds": [-180, -90, 180, 90],
                   "tiles": ["{z}/{x}/{y}.terrain"]}, f)
    return count


def build_imagery(tex, shade, water, place, rect, out, max_level):
    mem = gdal.GetDriverByName("MEM")
    jpeg = gdal.GetDriverByName("JPEG")
    count = 0
    for level in range(max_level + 1):
        size, tiles = tile_range(rect, level)
        spacing = size / IMAGE_SIZE * math.pi / 180 * R * math.cos(math.radians(place.lat0))
        ny = 2 ** level
        for tx, ty in tiles:
            west, north = -180 + tx * size, -90 + (ty + 1) * size
            k = (np.arange(IMAGE_SIZE) + 0.5) * size / IMAGE_SIZE
            lon, lat = np.meshgrid(west + k, north - k)
            xm, ym = place.to_metres(lon, lat)
            rgb = tex.sample(xm, ym, spacing) * shade.sample(xm, ym, spacing)[..., None]
            wa = 0.7 * water.grid.sample(xm, ym, spacing)[..., None]
            rgb = rgb * (1 - wa) + WATER_RGB * wa
            img = np.clip(rgb, 0, 255).astype(np.uint8)
            ds = mem.Create("", IMAGE_SIZE, IMAGE_SIZE, 3, gdal.GDT_Byte)
            for i in range(3):
                ds.GetRasterBand(i + 1).WriteArray(img[:, :, i])
            d = os.path.join(out, str(level), str(tx))
            os.makedirs(d, exist_ok=True)
            # Cesium's {y} counts from the north
            jpeg.CreateCopy(os.path.join(d, f"{ny - 1 - ty}.jpg"), ds, options=["QUALITY=88"])
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="game Data directory")
    ap.add_argument("--plugin", default="FalloutNV.esm")
    ap.add_argument("--textures", default="Fallout - Textures2.bsa")
    ap.add_argument("--world", default="WastelandNV")
    ap.add_argument("-o", "--out", default="site")
    ap.add_argument("--lat", type=float, default=36.17)
    ap.add_argument("--lon", type=float, default=-115.14)
    ap.add_argument("--terrain-level", type=int, default=17)
    ap.add_argument("--imagery-level", type=int, default=16)
    ap.add_argument("--title", help="display name of the map")
    ap.add_argument("--no-water", action="store_true", help="ignore water levels (e.g. city worldspaces)")
    ap.add_argument("--extra-markers", help="JSON list of additional labels (e.g. from build_models.py)")
    ap.add_argument("--tileset", action="append", default=[], metavar="NAME=URL",
                    help="3D Tiles layer to show on this map, URL relative to the map directory")
    a = ap.parse_args()

    place = Placement(a.lat, a.lon)
    markers, waters = {}, {}
    cellmap = parse_worlds(os.path.join(a.data, a.plugin), markers, waters)[a.world]
    grid, x0, ytop = mosaic(cellmap)
    grid = np.where(np.isnan(grid), np.nanmedian(grid), grid) * UNIT_M
    vpx = CELL_UNITS / STEP * UNIT_M                       # metres between DEM vertices

    # water level per DEM vertex (cell value, or the worldspace default); fill lakes up to it
    default_level, cell_levels = waters[a.world]
    ncx, ncy = (grid.shape[1] - 1) // STEP, (grid.shape[0] - 1) // STEP
    levels = np.full((ncy, ncx), default_level, np.float32)
    for (cx, cy), lvl in cell_levels.items():
        r, c = ytop - cy, cx - x0
        if 0 <= r < ncy and 0 <= c < ncx:
            levels[r, c] = lvl
    rows = np.minimum(np.arange(grid.shape[0]) // STEP, ncy - 1)
    cols = np.minimum(np.arange(grid.shape[1]) // STEP, ncx - 1)
    wl = levels[rows][:, cols] * UNIT_M
    wet = (grid < wl) & (not a.no_water)
    grid = np.where(wet, wl, grid)
    print(f"water: {wet.mean() * 100:.1f}% of the map below water level")
    dem = Grid(grid, x0 * CELL_UNITS * UNIT_M, (ytop + 1) * CELL_UNITS * UNIT_M, vpx, 8)
    water = SimpleNamespace(grid=Grid(wet.astype(np.float32), dem.left, dem.top, vpx, 8), place=place)

    rgb, tx0, ty0, ppc = lod_mosaic(os.path.join(a.data, a.textures), a.world)
    tpx = CELL_UNITS * UNIT_M / ppc
    ncy, ncx = rgb.shape[0] // ppc, rgb.shape[1] // ppc
    tex = Grid(rgb, tx0 * CELL_UNITS * UNIT_M + tpx / 2, (ty0 + ncy) * CELL_UNITS * UNIT_M - tpx / 2, tpx, 8)

    # bake a soft hillshade into the imagery so relief reads well from any angle
    mem = gdal.GetDriverByName("MEM").Create("", grid.shape[1], grid.shape[0], 1, gdal.GDT_Float32)
    mem.SetGeoTransform((0, vpx, 0, 0, 0, -vpx))
    mem.GetRasterBand(1).WriteArray(grid)
    hs = gdal.DEMProcessing("", mem, "hillshade", format="MEM", multiDirectional=True,
                            computeEdges=True).ReadAsArray().astype(np.float32)
    flat = np.median(hs)
    shade = Grid(np.clip(0.35 + 0.65 * hs / flat, 0.3, 1.6), dem.left, dem.top, vpx, 8)

    xm0, ym0 = tx0 * CELL_UNITS * UNIT_M, ty0 * CELL_UNITS * UNIT_M
    xm1, ym1 = (tx0 + ncx) * CELL_UNITS * UNIT_M, (ty0 + ncy) * CELL_UNITS * UNIT_M
    (w, e), (s, n) = place.to_lonlat(np.array([xm0, xm1]), np.array([ym0, ym1]))
    rect = (float(w), float(s), float(e), float(n))

    os.makedirs(a.out, exist_ok=True)
    nt = build_terrain(dem, water, place, rect, os.path.join(a.out, "terrain"), a.terrain_level)
    print(f"terrain tiles: {nt}")
    ni = build_imagery(tex, shade, water, place, rect, os.path.join(a.out, "imagery"), a.imagery_level)
    print(f"imagery tiles: {ni}")

    marks = []
    for name, x, y, z, t in markers.get(a.world, []):
        lon, lat = place.to_lonlat(x * UNIT_M, y * UNIT_M)
        if w <= lon <= e and s <= lat <= n:
            marks.append({"name": name, "lon": round(float(lon), 7), "lat": round(float(lat), 7),
                          "height": round(z * UNIT_M, 1),
                          "type": MARKER_TYPES[t] if t < len(MARKER_TYPES) else str(t)})
    if a.extra_markers:
        with open(a.extra_markers, encoding="utf-8") as f:
            marks += json.load(f)
    marks.sort(key=lambda m: m["name"])
    zmin, zmax = float(grid.min()), float(grid.max())
    with open(os.path.join(a.out, "world.json"), "w", encoding="utf-8") as f:
        json.dump({"world": a.world, "title": a.title or a.world, "rectangle": rect,
                   "center": [(w + e) / 2, (s + n) / 2], "size": max(xm1 - xm0, ym1 - ym0),
                   "heightRange": [zmin, zmax], "imageryMaxLevel": a.imagery_level,
                   "tilesets": [dict(zip(("name", "url"), t.split("=", 1))) for t in a.tileset],
                   "markers": marks}, f, ensure_ascii=False, indent=1)
    print(f"markers: {len(marks)}  rectangle: {rect}  heights: {zmin:.0f}..{zmax:.0f} m")


if __name__ == "__main__":
    main()
