"""Extract worldspace terrain heightmaps from Fallout 3 / New Vegas / Skyrim SE plugins (.esm/.esp)
and write them as GeoTIFF DEMs readable by GIS software.

Format notes (TES4-family records as used by FO3/FNV):
  record header: type[4] size:u32 flags:u32 formid:u32 vc:u32 ver:u16 unk:u16  (24 bytes)
  group header:  'GRUP' size:u32(incl. header) label[4] gtype:i32 stamp:u16 unk:u16 ver:u16 unk:u16
  compressed record (flags & 0x40000): u32 decompressed size + zlib stream
  localized plugin (TES4 flags & 0x80, Skyrim): FULL holds a u32 id into the .STRINGS table
  CELL.XCLC: i32 grid x, i32 grid y
  LAND.VHGT: f32 offset, 33x33 i8 deltas, 3 pad bytes; heights in units of 8 game units
A cell spans 4096 game units; 1 game unit = 1/64 yard = 0.0142875 m.
"""
import argparse
import os
import struct
import zlib

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

UNIT_M = 0.9144 / 64          # metres per game unit
CELL_UNITS = 4096
VERTS = 33
STEP = VERTS - 1              # vertices shared between neighbouring cells

# Nominal placement: a metre-based transverse Mercator centred on Las Vegas, so GIS software
# has a valid projected CRS. Map coordinates are the game's own (0,0 = world origin).
DEFAULT_CRS = "+proj=tmerc +lat_0=36.17 +lon_0=-115.14 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

REC = struct.Struct("<4sIIIIHH")
GRP = struct.Struct("<4sI4siHHHH")


def subrecords(data):
    pos, big = 0, None
    while pos + 6 <= len(data):
        typ, size = struct.unpack_from("<4sH", data, pos)
        pos += 6
        if typ == b"XXXX":
            big = struct.unpack_from("<I", data, pos)[0]
            pos += size
            continue
        if big is not None:
            size, big = big, None
        yield typ, data[pos:pos + size]
        pos += size


def record_data(buf, pos, size, flags):
    raw = buf[pos:pos + size]
    if flags & 0x40000:
        # raw deflate past the 2-byte zlib header: some shipped records carry bad adler32 checksums
        return zlib.decompressobj(-15).decompress(raw[6:])
    return raw


def zstring(data):
    return bytes(data).split(b"\x00", 1)[0].decode("cp1252", "replace")


def decode_vhgt(v):
    offset = struct.unpack_from("<f", v, 0)[0]
    d = np.frombuffer(v, dtype=np.int8, count=VERTS * VERTS, offset=4).astype(np.float64).reshape(VERTS, VERTS)
    # first column accumulates down the rows, then each row accumulates along columns
    col0 = offset + np.cumsum(d[:, 0])
    h = col0[:, None] + np.cumsum(np.concatenate([np.zeros((VERTS, 1)), d[:, 1:]], axis=1), axis=1)
    return h * 8.0  # rows = south->north, cols = west->east


def load_strings(data):
    """Parse a Skyrim .STRINGS table into {id: text}."""
    count, _ = struct.unpack_from("<II", data, 0)
    base = 8 + 8 * count
    out = {}
    for i in range(count):
        sid, off = struct.unpack_from("<II", data, 8 + 8 * i)
        end = data.index(b"\x00", base + off)
        out[sid] = data[base + off:end].decode("utf-8", "replace")
    return out


def parse_worlds(path, markers=None, water=None, strings=None):
    """Return {world_editor_id: {(x, y): heights33x33}}.

    If a dict is passed as `markers`, it is filled with
    {world_editor_id: [(name, x, y, z, marker_type), ...]} from map-marker references.
    If a dict is passed as `water`, it is filled with
    {world_editor_id: (default_level, {(x, y): level})} in game units (WRLD.NAM4 / CELL.XCLW).
    For localized plugins, `strings` maps FULL ids to names (see load_strings)."""
    buf = memoryview(open(path, "rb").read())
    worlds = {}           # formid -> edid
    cells = {}            # cell formid -> (world formid, x, y)
    lands = {}            # world formid -> {(x,y): heights}
    marks = {}            # world formid -> [(name, x, y, z, type)]
    default_water = {}    # world formid -> level
    cell_water = {}       # world formid -> {(x,y): level}
    localized = bool(REC.unpack_from(buf, 0)[2] & 0x80) and strings is not None

    def walk(start, end, world, cell):
        pos = start
        while pos < end:
            typ = bytes(buf[pos:pos + 4])
            if typ == b"GRUP":
                _, gsize, label, gtype, *_ = GRP.unpack_from(buf, pos)
                lab = struct.unpack("<I", label)[0]
                gend = pos + gsize
                if gtype == 0 and label not in (b"WRLD",):
                    pos = gend          # skip unrelated top-level groups
                    continue
                if gtype == 1:
                    walk(pos + 24, gend, lab, None)
                elif gtype in (6, 8, 9, 10):
                    walk(pos + 24, gend, world, lab)
                else:
                    walk(pos + 24, gend, world, cell)
                pos = gend
                continue
            _, size, flags, formid, *_ = REC.unpack_from(buf, pos)
            dpos = pos + 24
            if typ == b"WRLD":
                for st, sd in subrecords(record_data(buf, dpos, size, flags)):
                    if st == b"EDID":
                        worlds[formid] = zstring(sd)
                    elif st == b"NAM4":
                        default_water[formid] = struct.unpack_from("<f", sd, 0)[0]
                    elif st == b"DNAM" and len(sd) == 8:
                        # Skyrim: default land height, default water height (NAM4 wins if present)
                        default_water.setdefault(formid, struct.unpack_from("<f", sd, 4)[0])
            elif typ == b"CELL" and world is not None:
                sub = dict(subrecords(record_data(buf, dpos, size, flags)))
                if b"XCLC" in sub:
                    x, y = struct.unpack_from("<ii", sub[b"XCLC"], 0)
                    cells[formid] = (world, x, y)
                    # XCLW of FLT_MAX means "use the worldspace default" (Skyrim also writes 4294953216);
                    # huge negative values (-FLT_MAX, -2^31) mean no water
                    lvl = struct.unpack_from("<f", sub[b"XCLW"], 0)[0] if b"XCLW" in sub else 1e38
                    if lvl < 1e7:
                        cell_water.setdefault(world, {})[(x, y)] = lvl
            elif typ == b"LAND" and cell in cells:
                w, x, y = cells[cell]
                for st, sd in subrecords(record_data(buf, dpos, size, flags)):
                    if st == b"VHGT":
                        lands.setdefault(w, {})[(x, y)] = decode_vhgt(sd)
                        break
            # initially disabled markers (0x800) are not on the map at the start of the game, e.g. Fallout 3's
            # internal downtown sectors "DC01a" or "Megaton Ruins"
            elif typ == b"REFR" and markers is not None and world is not None and not flags & 0x800:
                sub = dict(subrecords(record_data(buf, dpos, size, flags)))
                if b"XMRK" in sub and b"FULL" in sub and b"DATA" in sub:
                    px, py, pz = struct.unpack_from("<3f", sub[b"DATA"], 0)
                    mtype = struct.unpack_from("<H", sub[b"TNAM"], 0)[0] if b"TNAM" in sub else 0
                    full = sub[b"FULL"]
                    name = strings.get(struct.unpack_from("<I", full, 0)[0], "") if localized else zstring(full)
                    marks.setdefault(world, []).append((name, px, py, pz, mtype))
            pos = dpos + size

    walk(0, len(buf), None, None)
    if markers is not None:
        markers.update({worlds.get(w, f"{w:08X}"): v for w, v in marks.items()})
    if water is not None:
        water.update({worlds.get(w, f"{w:08X}"): (default_water.get(w, -1e9), cell_water.get(w, {}))
                      for w in lands})
    return {worlds.get(w, f"{w:08X}"): v for w, v in lands.items()}


def mosaic(cellmap):
    xs = [k[0] for k in cellmap]
    ys = [k[1] for k in cellmap]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w = (x1 - x0 + 1) * STEP + 1
    h = (y1 - y0 + 1) * STEP + 1
    grid = np.full((h, w), np.nan, dtype=np.float32)
    for (x, y), hm in cellmap.items():
        c = (x - x0) * STEP
        r = (y1 - y) * STEP              # north-up raster: flip rows
        grid[r:r + VERTS, c:c + VERTS] = hm[::-1, :]
    return grid, x0, y1


def write_geotiff(path, grid, x0, ytop, metres, crs):
    scale = UNIT_M if metres else 1.0
    px = CELL_UNITS / STEP * scale
    # vertices are sample points: shift origin by half a pixel so pixel centres sit on them
    left = x0 * CELL_UNITS * scale - px / 2
    top = (ytop + 1) * CELL_UNITS * scale + px / 2
    nodata = -32768.0
    data = np.where(np.isnan(grid), nodata, grid * scale).astype(np.float32)
    ds = gdal.GetDriverByName("GTiff").Create(
        path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32,
        ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"])
    ds.SetGeoTransform((left, px, 0, top, 0, -px))
    if crs:
        srs = osr.SpatialReference()
        srs.ImportFromProj4(crs)
        ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadata({"SOURCE": "Bethesda LAND/VHGT", "UNITS": "metre" if metres else "game unit",
                    "GAME_UNIT_METRES": str(UNIT_M)})
    b = ds.GetRasterBand(1)
    b.SetNoDataValue(nodata)
    b.WriteArray(data)
    b.SetUnitType("m" if metres else "")
    ds.FlushCache()
    ds = None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plugin", help="path to .esm/.esp")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("-w", "--world", action="append", help="only these worldspace editor IDs")
    ap.add_argument("--game-units", action="store_true", help="keep game units instead of metres")
    ap.add_argument("--crs", default=DEFAULT_CRS,
                    help="PROJ string for the output CRS (metres); empty string for none")
    ap.add_argument("--hillshade", action="store_true", help="also write <world>_hillshade.tif")
    ap.add_argument("--min-cells", type=int, default=4, help="skip worldspaces with fewer cells")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for name, cellmap in sorted(parse_worlds(a.plugin).items()):
        if a.world and name not in a.world:
            continue
        if len(cellmap) < a.min_cells:
            continue
        grid, x0, ytop = mosaic(cellmap)
        dst = os.path.join(a.out, f"{name}.tif")
        write_geotiff(dst, grid, x0, ytop, not a.game_units, a.crs if not a.game_units else "")
        if a.hillshade:
            gdal.DEMProcessing(os.path.join(a.out, f"{name}_hillshade.tif"), dst, "hillshade",
                               multiDirectional=True, computeEdges=True,
                               creationOptions=["COMPRESS=DEFLATE", "TILED=YES"]).Close()
        valid = grid[~np.isnan(grid)]
        print(f"{name:32s} cells={len(cellmap):5d} size={grid.shape[1]}x{grid.shape[0]} "
              f"z=[{valid.min() * UNIT_M:.0f}, {valid.max() * UNIT_M:.0f}] m -> {dst}")


if __name__ == "__main__":
    main()
