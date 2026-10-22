"""Convert the distant-LOD building blocks of a Fallout 3 / New Vegas worldspace (or the object LOD blocks
of a Skyrim SE worldspace, *.bto) into a 3D Tiles 1.1 tileset (one GLB per block, shared texture atlas) for CesiumJS.
"""
import argparse
import json
import math
import os
import re
import struct

import numpy as np
from osgeo import gdal

from bsa import BSA
from build_cesium import GAMES as CESIUM_GAMES, Placement
from fnv_terrain import UNIT_M
from lod_texture import read_dds
from nif import Nif, extract_meshes, read_av_object, read_tri_shape_data

gdal.UseExceptions()

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
SEP = "\\"


def geodetic_to_ecef(lon, lat, h):
    lon, lat = np.radians(lon), np.radians(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + h) * np.sin(lat)], axis=-1)


def enu_frame(lon, lat):
    """Rows: east, north, up unit vectors in ECEF."""
    lo, la = math.radians(lon), math.radians(lat)
    e = [-math.sin(lo), math.cos(lo), 0.0]
    n = [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)]
    u = [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]
    return np.array([e, n, u])


def read_block(nif):
    """Collect the geometry of a LOD block NIF in game units (node transforms applied)."""
    if nif.bs_version > 34:               # Skyrim .bto: regular scene graph of BSSubIndexTriShapes
        return [(p["positions"], p.get("normals"), p.get("uvs"), p["triangles"]) for p in extract_meshes(nif)]
    parts = []
    shapes = {}
    for i, (t, _, _) in enumerate(nif.blocks):
        if t in ("NiTriShape", "BSSegmentedTriShape"):
            r = nif.reader(i)
            _, tr, rot, scale = read_av_object(r, nif.bs_version)
            for _ in range(r.u32()):
                r.i32()                       # properties
            r.i32()                           # collision object
            shapes[r.i32()] = (tr, rot, scale)  # data ref
    for i, (tr, rot, scale) in shapes.items():
        g = read_tri_shape_data(nif.reader(i))
        if "triangles" not in g or "positions" not in g:
            continue
        pos = g["positions"].astype(np.float64) * scale @ rot.T + tr
        nrm = g["normals"] @ rot.T if "normals" in g else None
        parts.append((pos, nrm, g.get("uvs"), g["triangles"]))
    return parts


def glb(positions, normals, uvs, indices, image_uri):
    """Write a minimal single-mesh GLB (positions must already be glTF y-up)."""
    chunks, views, accessors = [], [], []
    offset = 0

    def add(arr, target, comp, typ, minmax=False):
        nonlocal offset
        data = arr.tobytes()
        pad = (-len(data)) % 4
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        chunks.append(data + b"\x00" * pad)
        offset += len(data) + pad
        return len(accessors) - 1

    attrs = {"POSITION": add(positions.astype("<f4"), 34962, 5126, "VEC3", True)}
    if normals is not None:
        attrs["NORMAL"] = add(normals.astype("<f4"), 34962, 5126, "VEC3")
    if uvs is not None:
        attrs["TEXCOORD_0"] = add(uvs.astype("<f4"), 34962, 5126, "VEC2")
    idx_type = (5123, "<u2") if positions.shape[0] < 65536 else (5125, "<u4")
    ind = add(indices.reshape(-1).astype(idx_type[1]), 34963, idx_type[0], "SCALAR")
    material = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
                "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True}
    gltf = {"asset": {"version": "2.0", "generator": "game-terrain-gis"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": attrs, "indices": ind, "material": 0}]}],
            "materials": [material], "buffers": [{"byteLength": offset}],
            "bufferViews": views, "accessors": accessors}
    if uvs is not None:
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        gltf.update({"textures": [{"source": 0, "sampler": 0}], "images": [{"uri": image_uri}],
                     "samplers": [{"magFilter": 9729, "minFilter": 9987}]})
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binary = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="game Data directory")
    ap.add_argument("--game", choices=("fnv", "skyrim"), default="fnv")
    ap.add_argument("--meshes", help="archive with the blocks (default: the game's)")
    ap.add_argument("--textures", help="archive with the atlas (default: the game's)")
    ap.add_argument("--world", default="WastelandNV")
    ap.add_argument("-o", "--out", default="site/buildings")
    ap.add_argument("--lat", type=float, help="nominal latitude of the world origin (as build_cesium.py)")
    ap.add_argument("--lon", type=float, help="nominal longitude of the world origin")
    ap.add_argument("--gain", type=float, help="atlas brightness factor (default 1.7 for Skyrim's dark textures)")
    ap.add_argument("--load-distance", type=float, default=60.0,
                    help="geometric error of each block; larger loads blocks from farther away")
    a = ap.parse_args()

    skyrim = a.game == "skyrim"
    g = CESIUM_GAMES[a.game]
    place = Placement(g["lat"] if a.lat is None else a.lat, g["lon"] if a.lon is None else a.lon)
    world = a.world.lower()
    os.makedirs(a.out, exist_ok=True)

    # texture atlas
    tex = BSA(os.path.join(a.data, a.textures or ("Skyrim - Textures7.bsa" if skyrim else "Fallout - Textures2.bsa")))
    if skyrim:
        atlas_key = SEP.join(["textures", "terrain", world, "objects", f"{world}.objects.dds"])
    else:
        atlas_key = SEP.join(["textures", "landscape", "lod", world, "blocks", f"{world}.buildings.dds"])
    rgba = read_dds(tex.read(atlas_key))
    gain = a.gain if a.gain is not None else (1.7 if skyrim else 1.0)
    if gain != 1.0:
        rgba[..., :3] = np.clip(rgba[..., :3] * gain, 0, 255).astype(np.uint8)
    mem = gdal.GetDriverByName("MEM").Create("", rgba.shape[1], rgba.shape[0], 4, gdal.GDT_Byte)
    for i in range(4):
        mem.GetRasterBand(i + 1).WriteArray(rgba[:, :, i])
    gdal.GetDriverByName("PNG").CreateCopy(os.path.join(a.out, "atlas.png"), mem)

    # block meshes; Update.bsa overrides the base archive
    archives = [BSA(os.path.join(a.data, a.meshes or ("Skyrim - Meshes1.bsa" if skyrim else "Fallout - Meshes.bsa")))]
    if os.path.exists(os.path.join(a.data, "Update.bsa")):
        archives.append(BSA(os.path.join(a.data, "Update.bsa")))
    if skyrim:   # the 4x4-cell level only; 8 and 16 are coarser versions for farther away
        pat = re.compile(rf"meshes\\terrain\\{world}\\objects\\{world}\.4(?P<high>)\.(?P<x>-?\d+)\.(?P<y>-?\d+)\.bto$")
    else:
        pat = re.compile(rf"meshes\\landscape\\lod\\{world}\\blocks\\{world}\.level\d+(?P<high>\.high)?"
                         rf"\.x(?P<x>-?\d+)\.y(?P<y>-?\d+)\.nif$")
    blocks = {}
    for arc in archives:
        for k in arc.files:
            m = pat.match(k)
            if m:
                xy = (int(m.group("x")), int(m.group("y")))
                # ".high" variants are more detailed versions of a few blocks; prefer them
                if m.group("high") or not blocks.get(xy, (None, ""))[1].count(".high."):
                    blocks[xy] = (arc, k)

    children, nverts, ntris = [], 0, 0
    rw = rs = rmin = math.inf
    re_ = rn = rmax = -math.inf
    for (bx, by), (arc, key) in sorted(blocks.items()):
        parts = read_block(Nif(arc.read(key)))
        if not parts:
            continue
        pos = np.concatenate([p[0] for p in parts])
        nrm = np.concatenate([p[1] for p in parts]) if all(p[1] is not None for p in parts) else None
        uvs = np.concatenate([p[2] for p in parts]) if all(p[2] is not None for p in parts) else None
        base = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        tris = np.concatenate([p[3].astype(np.int64) + b for p, b in zip(parts, base)])

        lon, lat = place.to_lonlat(pos[:, 0] * UNIT_M, pos[:, 1] * UNIT_M)
        h = pos[:, 2] * UNIT_M
        ecef = geodetic_to_ecef(lon, lat, h)
        clon, clat = float(lon.mean()), float(lat.mean())
        center = geodetic_to_ecef(np.array(clon), np.array(clat), np.array(float(h.min())))
        frame = enu_frame(clon, clat)
        local = (ecef - center) @ frame.T                  # east, north, up
        gl_pos = np.stack([local[:, 0], local[:, 2], -local[:, 1]], axis=1)
        gl_nrm = None
        if nrm is not None:
            n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            gl_nrm = np.stack([n[:, 0], n[:, 2], -n[:, 1]], axis=1)

        name = f"b_{bx}_{by}.glb"
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(glb(gl_pos, gl_nrm, uvs, tris, "atlas.png"))

        transform = np.eye(4)
        transform[:3, :3] = frame.T
        transform[:3, 3] = center
        region = [math.radians(float(lon.min())), math.radians(float(lat.min())),
                  math.radians(float(lon.max())), math.radians(float(lat.max())),
                  float(h.min()), float(h.max())]
        rw, rs, re_, rn = min(rw, region[0]), min(rs, region[1]), max(re_, region[2]), max(rn, region[3])
        rmin, rmax = min(rmin, region[4]), max(rmax, region[5])
        children.append({
            "boundingVolume": {"region": region}, "geometricError": a.load_distance, "refine": "ADD",
            "children": [{"boundingVolume": {"region": region}, "geometricError": 0,
                          "transform": transform.T.reshape(-1).tolist(),   # column-major
                          "content": {"uri": name}}]})
        nverts += len(pos)
        ntris += len(tris)

    tileset = {"asset": {"version": "1.1"}, "geometricError": 2000,
               "root": {"boundingVolume": {"region": [rw, rs, re_, rn, rmin, rmax]},
                        "geometricError": 1000, "refine": "ADD", "children": children}}
    with open(os.path.join(a.out, "tileset.json"), "w") as f:
        json.dump(tileset, f)
    print(f"blocks: {len(children)}  vertices: {nverts}  triangles: {ntris}")


if __name__ == "__main__":
    main()
