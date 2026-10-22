"""Convert the placed objects (full-detail models) of one or more Fallout 3 / New Vegas
worldspaces into a 3D Tiles 1.1 tileset for CesiumJS, plus labels for building entrances.
"""
import argparse
import collections
import json
import math
import os
import re
import struct

import numpy as np
from osgeo import gdal

import fnv_terrain as ft
from bsa import BSA
from build_buildings import enu_frame, geodetic_to_ecef
from build_cesium import GAMES as CESIUM_GAMES, Placement
from lod_texture import read_dds
from nif import Nif, extract_meshes

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

UNIT_M = ft.UNIT_M
FX = re.compile(r"(^|\\)fx[^\\]*$|^effects\\tests\\")   # particle / test effect meshes
MINOR = re.compile(r"^(landscape|trees|clutter|plants)\\")   # rocks, plants, roads, junk: smaller textures, game LODs
PLACE = None                       # Placement, set in main()
MODEL_TYPES = {b"STAT", b"MSTT", b"ACTI", b"DOOR", b"CONT", b"FURN", b"TACT", b"LIGH", b"TREE", b"FLOR"}
# plugins (load order) and archives per game; Skyrim SE ships the DLC assets inside its main archives
GAMES = {
    "fnv": dict(plugins=["FalloutNV.esm"], meshes=["Fallout - Meshes.bsa", "Update.bsa"], skip=None,
                textures=["Fallout - Textures.bsa", "Fallout - Textures2.bsa", "Update.bsa"], gain=1.0),
    "skyrim": dict(plugins=["Skyrim.esm", "Update.esm", "Dawnguard.esm", "HearthFires.esm", "Dragonborn.esm"],
                   meshes=["Skyrim - Meshes0.bsa", "Skyrim - Meshes1.bsa"],
                   # editor markers, sky domes, critter spawners and fog/mist/spray effects
                   skip=re.compile(r"^(markers|sky|critters|effects)\\"),
                   textures=[f"Skyrim - Textures{i}.bsa" for i in range(9)],
                   # Skyrim textures are dark and rely on strong in-game lighting
                   gain=1.7),
}
Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)   # (e,n,u) -> glTF (e,u,-n)


def load_order_remap(buf, plugin, order):
    """FormID mapper from a plugin's own numbering (high byte = index in its master list, or itself
    after the masters) to the load order `order` (lowercase plugin names)."""
    _, size, flags, *_ = ft.REC.unpack_from(buf, 0)
    masters = [ft.zstring(v).lower() for k, v in ft.subrecords(ft.record_data(buf, 24, size, flags)) if k == b"MAST"]
    table = [order.index(m) if m in order else 0xFF for m in masters] + [order.index(plugin.lower())]
    return lambda fid: (table[min(fid >> 24, len(masters))] << 24) | (fid & 0xFFFFFF)


def scan_plugins(data, plugins, worlds):
    """scan_esm over plugins in load order: later plugins override base records and references
    (a deleted or disabled override removes the reference)."""
    order = [p.lower() for p in plugins]
    base, refs, dest_names = {}, collections.defaultdict(dict), {}
    for p in plugins:
        b, r, d = scan_esm(os.path.join(data, p), worlds, order)
        base.update(b)
        dest_names.update(d)
        for w, items in r.items():
            refs[w].update(items)
    return base, {w: [v for v in items.values() if v is not None] for w, items in refs.items()}, dest_names


def scan_esm(path, worlds, order=None):
    """Collect base models, placed references in `worlds` ({ref formid: ref, or None when removed}) and
    door destinations for labels. `order`: load order, to translate FormIDs (see scan_plugins)."""
    buf = memoryview(open(path, "rb").read())
    remap = load_order_remap(buf, os.path.basename(path), order) if order else (lambda fid: fid)
    base = {}                    # formid -> (type, edid, model, full name)
    wnames = {}
    refs = collections.defaultdict(dict)   # world -> {ref formid: (base, pos, rot, scale, xtel_dest) | None}
    cell_names = {}              # cell formid -> FULL
    ref_cell = {}                # ref formid -> cell formid (only for door destinations)
    wanted_dest = set()

    def walk(start, end, world, cell):
        pos = start
        while pos < end:
            typ = bytes(buf[pos:pos + 4])
            if typ == b"GRUP":
                _, gsize, label, gtype, *_ = ft.GRP.unpack_from(buf, pos)
                lab = remap(struct.unpack("<I", label)[0])
                if gtype == 1:
                    walk(pos + 24, pos + gsize, lab, None)
                elif gtype in (6, 8, 9, 10):
                    walk(pos + 24, pos + gsize, world, lab)
                else:
                    walk(pos + 24, pos + gsize, world, cell)
                pos += gsize
                continue
            _, size, flags, formid, *_ = ft.REC.unpack_from(buf, pos)
            formid = remap(formid)
            dpos = pos + 24
            if typ in MODEL_TYPES:
                sub = dict(ft.subrecords(ft.record_data(buf, dpos, size, flags)))
                if b"MODL" in sub:
                    # TREE.BNAM: SpeedTree billboard width, height (game units)
                    bnam = struct.unpack_from("<2f", sub[b"BNAM"]) if typ == b"TREE" and b"BNAM" in sub else None
                    base[formid] = (typ, ft.zstring(sub.get(b"EDID", b"")), ft.zstring(sub[b"MODL"]),
                                    ft.zstring(sub.get(b"FULL", b"")), bnam)
            elif typ == b"WRLD":
                wnames[formid] = ft.zstring(dict(ft.subrecords(ft.record_data(buf, dpos, size, flags))).get(b"EDID", b""))
            elif typ == b"CELL":
                sub = dict(ft.subrecords(ft.record_data(buf, dpos, size, flags)))
                if b"FULL" in sub:
                    cell_names[formid] = ft.zstring(sub[b"FULL"])
            elif typ == b"REFR":
                w = wnames.get(world) if world is not None else None
                if w in worlds:
                    sub = {} if flags & 0x20 else dict(ft.subrecords(ft.record_data(buf, dpos, size, flags)))
                    refs[w][formid] = None          # deleted, disabled or a marker: also cancels earlier plugins
                    if b"NAME" in sub and b"DATA" in sub and not flags & 0x800 and b"XMRK" not in sub:
                        d = struct.unpack("<6f", sub[b"DATA"])
                        scale = struct.unpack("<f", sub[b"XSCL"])[0] if b"XSCL" in sub else 1.0
                        dest = remap(struct.unpack("<I", sub[b"XTEL"][:4])[0]) if b"XTEL" in sub else None
                        if dest:
                            wanted_dest.add(dest)
                        refs[w][formid] = (remap(struct.unpack("<I", sub[b"NAME"])[0]), np.array(d[:3]),
                                           np.array(d[3:]), scale, dest)
                elif cell is not None:
                    ref_cell[formid] = cell
            pos = dpos + size

    walk(0, len(buf), None, None)
    dest_names = {d: cell_names.get(ref_cell.get(d)) for d in wanted_dest}
    return base, refs, dest_names


def rotation(rx, ry, rz):
    """Bethesda reference rotation (radians, clockwise) as a column-vector matrix."""
    def rot(axis, a):
        c, s = math.cos(a), math.sin(a)
        m = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        if axis == 1:
            m[i, j], m[j, i] = s, -s
        return m
    return rot(0, -rx) @ rot(1, -ry) @ rot(2, -rz)


def norm_texture(path):
    t = path.lower().replace("/", "\\").lstrip("\\")
    # absolute build paths baked into some Skyrim meshes, e.g. skyrimhd\build\pc\data\textures\...
    i = t.find("textures\\")
    if i > 0:
        t = t[i:]
    if t.startswith("data\\"):
        t = t[5:]
    return t if t.startswith("textures\\") else "textures\\" + t


class Archives:
    def __init__(self, data, names):
        self.arcs = [BSA(os.path.join(data, n)) for n in names if os.path.exists(os.path.join(data, n))]

    def read(self, key):
        for a in reversed(self.arcs):     # later archives override earlier ones
            if key in a.files:
                return a.read(key)
        return None


def billboard_prims(model, bnam):
    """SpeedTree (.spt) trees as two crossed quads with the game's own billboard texture."""
    w, h = bnam
    name = os.path.splitext(model.replace("\\", "/").rsplit("/", 1)[-1])[0]
    hw = w / 2
    pos = np.array([[-hw, 0, 0], [hw, 0, 0], [hw, 0, h], [-hw, 0, h],
                    [0, -hw, 0], [0, hw, 0], [0, hw, h], [0, -hw, h]], float)
    uvs = np.tile(np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32), (2, 1))
    nrm = np.tile(np.array([[0, 0, 1]], np.float32), (8, 1))    # light like the ground, not the card
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.uint32)
    mat = {"texture": f"trees\\billboards\\{name}.dds", "alpha_mode": "MASK", "cutoff": 0.5,
           "alpha": 1.0, "unlit": False}
    return [{"positions": pos, "normals": nrm, "uvs": uvs, "triangles": tris, "material": mat}]


def _cluster(p, lo, cell):
    """Vertex-clustering simplification of one primitive: vertices sharing a grid cell (and roughly the
    same UV, so texture seams stay apart) merge into one at their mean position."""
    key = np.floor((p["positions"] - lo) / cell).astype(np.int64)
    if "uvs" in p:
        key = np.concatenate([key, np.floor(np.nan_to_num(p["uvs"]) * 4).astype(np.int64)], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(first)
    count = np.bincount(inv, minlength=n)[:, None]
    out = {"material": p["material"],
           "positions": np.stack([np.bincount(inv, p["positions"][:, i], n) for i in range(3)], 1) / count}
    if "normals" in p:
        nrm = np.stack([np.bincount(inv, p["normals"][:, i], n) for i in range(3)], 1)
        out["normals"] = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    for k in ("uvs", "colors"):
        if k in p:
            out[k] = p[k][first]
    t = inv[p["triangles"].astype(np.int64)]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    out["triangles"] = np.unique(t, axis=0).astype(np.uint32) if len(t) else t.astype(np.uint32)
    return out


def simplify(prims, ratio=0.25, min_triangles=64):
    """Reduced copy of a model with about `ratio` of its triangles, or None if it is already cheap."""
    total = sum(len(p["triangles"]) for p in prims)
    if total < min_triangles:
        return None
    allp = np.concatenate([p["positions"] for p in prims])
    lo = allp.min(0)
    diag = float(np.linalg.norm(allp.max(0) - lo))
    cell, best, best_n = diag / 96, None, 0.5 * total      # not worth a LOD above half the triangles
    while cell < diag / 6:
        out = [q for q in (_cluster(p, lo, cell) for p in prims) if len(q["triangles"])]
        n = sum(len(q["triangles"]) for q in out)
        if out and n <= best_n:
            best, best_n = out, n
            if n <= ratio * total:
                break
        cell *= 1.35
    return best


def quaternion(m):
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[i] = s / 4
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


class MeshStore:
    """Writes each model's geometry once to meshes/<name>.bin. Tiles reference it as an external
    buffer, so it is downloaded (and cached by Cesium) once instead of being copied into every tile."""

    def __init__(self, out):
        self.out = out
        self.layouts = {}
        os.makedirs(os.path.join(out, "meshes"), exist_ok=True)

    def add(self, model, prims):
        buf = bytearray()

        def put(arr, comp, typ, target, minmax=False):
            data = arr.tobytes()
            view = {"byteOffset": len(buf), "byteLength": len(data), "target": target}
            buf.extend(data + b"\x00" * ((-len(data)) % 4))
            acc = {"componentType": comp, "count": len(arr), "type": typ}
            if minmax:
                acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
            return view, acc

        layout = []
        for p in prims:
            attrs = {"POSITION": put(p["positions"].astype("<f4"), 5126, "VEC3", 34962, True)}
            if "normals" in p:
                attrs["NORMAL"] = put(p["normals"].astype("<f4"), 5126, "VEC3", 34962)
            if "uvs" in p:
                attrs["TEXCOORD_0"] = put(p["uvs"].astype("<f4"), 5126, "VEC2", 34962)
            if "colors" in p:
                c = np.clip(p["colors"], 0, 1).astype("<f4")
                if p["material"]["alpha_mode"] == "OPAQUE":
                    attrs["COLOR_0"] = put(np.ascontiguousarray(c[:, :3]), 5126, "VEC3", 34962)
                else:
                    attrs["COLOR_0"] = put(c, 5126, "VEC4", 34962)
            idx = p["triangles"].reshape(-1)
            if len(p["positions"]) < 65536:
                ind = put(idx.astype("<u2"), 5123, "SCALAR", 34963)
            else:
                ind = put(idx.astype("<u4"), 5125, "SCALAR", 34963)
            layout.append({"attributes": attrs, "indices": ind, "material": p["material"]})
        stem = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(model.replace("\\", "/").rsplit("/", 1)[-1])[0])
        name = f"{len(self.layouts):04d}_{stem}.bin"
        with open(os.path.join(self.out, "meshes", name), "wb") as f:
            f.write(buf)
        self.layouts[model] = ("meshes/" + name, len(buf), layout)


class GltfBuilder:
    def __init__(self, store):
        self.store = store
        self.bin = bytearray()            # GLB chunk: per-tile instance transforms
        self.buffers = [None]
        self.views, self.accessors, self.meshes, self.nodes = [], [], [], []
        self.materials, self.material_index = [], {}
        self.images, self.image_index = [], {}
        self.extensions = set()

    def accessor(self, arr, comp, typ):
        data = arr.tobytes()
        self.views.append({"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)})
        self.bin += data + b"\x00" * ((-len(data)) % 4)
        self.accessors.append({"bufferView": len(self.views) - 1, "componentType": comp,
                               "count": len(arr), "type": typ})
        return len(self.accessors) - 1

    def material(self, m, image_uri):
        key = (image_uri, m["alpha_mode"], round(m["cutoff"], 3), round(m["alpha"], 3), m["unlit"])
        if key in self.material_index:
            return self.material_index[key]
        mat = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.9,
                                        "baseColorFactor": [1, 1, 1, float(m["alpha"])]},
               "alphaMode": m["alpha_mode"], "doubleSided": True}
        if m["alpha_mode"] == "MASK":
            mat["alphaCutoff"] = float(m["cutoff"])
        if image_uri:
            if image_uri not in self.image_index:
                self.image_index[image_uri] = len(self.images)
                self.images.append(image_uri)
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": self.image_index[image_uri]}
        if m["unlit"]:
            mat["extensions"] = {"KHR_materials_unlit": {}}
            self.extensions.add("KHR_materials_unlit")
        self.materials.append(mat)
        self.material_index[key] = len(self.materials) - 1
        return self.material_index[key]

    def mesh(self, model, image_for):
        uri, length, layout = self.store.layouts[model]
        self.buffers.append({"uri": uri, "byteLength": length})
        buffer = len(self.buffers) - 1

        def ref(spec):
            view, acc = spec
            self.views.append({"buffer": buffer, **view})
            self.accessors.append({"bufferView": len(self.views) - 1, **acc})
            return len(self.accessors) - 1

        prims = [{"attributes": {k: ref(v) for k, v in p["attributes"].items()}, "indices": ref(p["indices"]),
                  "material": self.material(p["material"], image_for(p["material"]))} for p in layout]
        self.meshes.append({"primitives": prims})
        return len(self.meshes) - 1

    def node(self, mesh, matrix):
        self.nodes.append({"mesh": mesh, "matrix": matrix.T.reshape(-1).tolist()})

    def instanced_node(self, mesh, matrix, translations, rotations, scales):
        """One node drawing `mesh` at many places (EXT_mesh_gpu_instancing, in the node's frame)."""
        attrs = {"TRANSLATION": self.accessor(translations.astype("<f4"), 5126, "VEC3"),
                 "ROTATION": self.accessor(rotations.astype("<f4"), 5126, "VEC4"),
                 "SCALE": self.accessor(scales.astype("<f4"), 5126, "VEC3")}
        self.nodes.append({"mesh": mesh, "matrix": matrix.T.reshape(-1).tolist(),
                           "extensions": {"EXT_mesh_gpu_instancing": {"attributes": attrs}}})
        self.extensions.add("EXT_mesh_gpu_instancing")

    def glb(self):
        if not self.bin:
            self.bin += b"\x00" * 4               # buffers must not be empty
        self.buffers[0] = {"byteLength": len(self.bin)}
        gltf = {"asset": {"version": "2.0", "generator": "game-terrain-gis"},
                "scene": 0, "scenes": [{"nodes": list(range(len(self.nodes)))}],
                "nodes": self.nodes, "meshes": self.meshes, "materials": self.materials,
                "buffers": self.buffers, "bufferViews": self.views, "accessors": self.accessors}
        if self.images:
            gltf["images"] = [{"uri": u} for u in self.images]
            gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987}]
            gltf["textures"] = [{"source": i, "sampler": 0} for i in range(len(self.images))]
        if self.extensions:
            gltf["extensionsUsed"] = sorted(self.extensions)
        if "EXT_mesh_gpu_instancing" in self.extensions:
            gltf["extensionsRequired"] = ["EXT_mesh_gpu_instancing"]
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * ((-len(js)) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
                + struct.pack("<I4s", len(self.bin), b"BIN\x00") + bytes(self.bin))


def save_texture(rgba, path, max_size, gain=1.0):
    if gain != 1.0:
        rgba = rgba.copy()
        rgba[..., :3] = np.clip(rgba[..., :3] * gain, 0, 255).astype(np.uint8)
    while max(rgba.shape[:2]) > max_size:
        h, w = rgba.shape[0] // 2 * 2, rgba.shape[1] // 2 * 2
        a = rgba[:h, :w].astype(np.uint16)
        rgba = ((a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) // 4).astype(np.uint8)
    png = path.endswith(".png")
    bands = 4 if png else 3
    mem = gdal.GetDriverByName("MEM").Create("", rgba.shape[1], rgba.shape[0], bands, gdal.GDT_Byte)
    for i in range(bands):
        mem.GetRasterBand(i + 1).WriteArray(rgba[:, :, i] if rgba.shape[2] > i else rgba[:, :, 0])
    gdal.GetDriverByName("PNG" if png else "JPEG").CreateCopy(path, mem, options=[] if png else ["QUALITY=85"])




def region_of(points):
    """3D Tiles bounding region of game-unit points."""
    lons, lats = PLACE.to_lonlat(points[:, 0] * UNIT_M, points[:, 1] * UNIT_M)
    return [math.radians(float(lons.min())), math.radians(float(lats.min())),
            math.radians(float(lons.max())), math.radians(float(lats.max())),
            float(points[:, 2].min() * UNIT_M), float(points[:, 2].max() * UNIT_M)]


def union(regions):
    r = np.array(regions)
    return [float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 2].max()), float(r[:, 3].max()),
            float(r[:, 4].min()), float(r[:, 5].max())]


def write_tile(path, items, models, store, image_for, center):
    """One GLB for `items`; models placed more than once are drawn with GPU instancing."""
    g = GltfBuilder(store)
    by_model = collections.defaultdict(list)
    for it in items:
        by_model[it[0]].append(it)
    for model, group in sorted(by_model.items()):
        mesh = g.mesh(model, image_for)
        if len(group) == 1:
            _, pos, rot, scale, _, _ = group[0]
            t = np.eye(4)
            t[:3, :3] = rotation(*rot) * scale * UNIT_M
            t[:3, 3] = (pos - center) * UNIT_M
            g.node(mesh, Y_UP @ t)
        else:
            tr = np.array([(p[1] - center) * UNIT_M for p in group])
            q = np.array([quaternion(rotation(*p[2])) for p in group])
            s = np.array([[p[3] * UNIT_M] * 3 for p in group])
            g.instanced_node(mesh, Y_UP, tr, q, s)
    with open(path, "wb") as f:
        f.write(g.glb())


def main():
    global PLACE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="game Data directory")
    ap.add_argument("--game", choices=GAMES, default="fnv", help="defaults for plugins and archives")
    ap.add_argument("--plugin", action="append", help="plugin(s) in load order (default: the game's masters)")
    ap.add_argument("--world", action="append", required=True,
                    help="worldspace editor ID; repeat to merge overlapping worldspaces")
    ap.add_argument("-o", "--out", required=True, help="output directory for the tileset")
    ap.add_argument("--lat", type=float, help="nominal latitude of the world origin (as build_cesium.py)")
    ap.add_argument("--lon", type=float, help="nominal longitude of the world origin")
    ap.add_argument("--cells", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="only references in this range of cells (inclusive), for quick tests")
    ap.add_argument("--texture-size", type=int, default=512)
    ap.add_argument("--minor-texture-size", type=int, default=256,
                    help="maximum size of textures used only by landscape / tree / clutter models")
    ap.add_argument("--simplify-ratio", type=float, default=0.25,
                    help="target triangle share of generated LODs for minor models without a game LOD")
    ap.add_argument("--swap-distance", type=float, default=12.0,
                    help="geometric error at which minor models switch from the game's _lod mesh to the full one")
    ap.add_argument("--tile-cells", type=int, default=2, help="cells per tile edge (use 4 with --lod-tileset)")
    ap.add_argument("--load-distance", type=float, default=40.0,
                    help="geometric error at which a tile's large objects load")
    ap.add_argument("--small-distance", type=float, default=10.0,
                    help="geometric error at which a tile's small objects load")
    ap.add_argument("--small-radius", type=float, default=4.0,
                    help="objects with a bounding radius below this (metres) count as small")
    ap.add_argument("--lod-tileset", help="tileset.json of build_buildings.py blocks, shown from afar and "
                                          "replaced by the full models up close")
    ap.add_argument("--lod-distance", type=float, default=60.0, help="geometric error of the LOD blocks")
    ap.add_argument("--labels", help="write building-entrance labels to this JSON file")
    a = ap.parse_args()

    g = GAMES[a.game]
    place = CESIUM_GAMES[a.game]
    PLACE = Placement(place["lat"] if a.lat is None else a.lat, place["lon"] if a.lon is None else a.lon)
    tile_units = a.tile_cells * ft.CELL_UNITS
    base, refs, dest_names = scan_plugins(a.data, a.plugin or g["plugins"], set(a.world))
    meshes = Archives(a.data, g["meshes"])
    textures = Archives(a.data, g["textures"])

    # merge worldspaces, dropping near-identical duplicates of the same model
    placed, seen, bnams = [], set(), {}
    for w in a.world:
        for b, pos, rot, scale, dest in refs.get(w, []):
            if b not in base:
                continue
            if a.cells:
                cx, cy = pos[0] // ft.CELL_UNITS, pos[1] // ft.CELL_UNITS
                if not (a.cells[0] <= cx <= a.cells[2] and a.cells[1] <= cy <= a.cells[3]):
                    continue
            typ, edid, model, full, bnam = base[b]
            model = model.lower().lstrip("\\")
            if "marker" in model or edid.lower().endswith("marker") or FX.search(model) or (g["skip"] and g["skip"].match(model)):
                continue
            if model.endswith(".spt"):
                if not bnam:
                    continue
                bnams[model] = bnam
            key = (model, *np.round(pos / 64).astype(int))
            if key in seen:
                continue
            seen.add(key)
            placed.append((model, pos, rot, scale, typ, dest))

    # load each model once and write its geometry to the shared mesh store
    store = MeshStore(a.out)
    models = {}
    game_lod = set()

    def add_model(key, prims):
        allp = np.concatenate([p["positions"] for p in prims])
        lo, hi = allp.min(0), allp.max(0)
        models[key] = (prims, lo, hi, float(np.linalg.norm(np.abs(np.stack([lo, hi])).max(0))))
        store.add(key, prims)

    for model in sorted({p[0] for p in placed}):
        if model in bnams:
            prims = billboard_prims(model, bnams[model])
        else:
            data = meshes.read("meshes\\" + model)
            try:
                prims = extract_meshes(Nif(data)) if data else []
            except (ValueError, IndexError, struct.error) as e:
                print(f"skipped {model}: {e}")
                prims = []
        if not prims:
            continue
        if all(p["material"]["alpha_mode"] == "BLEND" for p in prims):
            continue                      # glow domes, light beams, smoke cards: effect-only meshes
        add_model(model, prims)
        if not MINOR.match(model) or model in bnams:
            continue
        # mid-distance version: the game's own reduced mesh (e.g. 4900 -> 150 triangles for a cliff),
        # otherwise our own simplification
        # (Skyrim trees ship a crossed-quad "_lod_flat" version beside the full model)
        data = meshes.read("meshes\\" + model[:-4] + "_lod.nif") or meshes.read("meshes\\" + model[:-4] + "_lod_flat.nif")
        lod = extract_meshes(Nif(data)) if data else []
        if lod:
            game_lod.add(model)
        else:
            lod = simplify(models[model][0], a.simplify_ratio)
        if lod:
            add_model(model + "|lod", lod)
    print(f"references: {len(placed)}  models: {len(models)}  with game LOD: {len(game_lod)}  "
          f"simplified: {sum(k.endswith('|lod') for k in models) - len(game_lod)}")

    # textures: one image per texture path, PNG when any material needs alpha
    os.makedirs(os.path.join(a.out, "textures"), exist_ok=True)
    tex_alpha = collections.defaultdict(bool)
    tex_major = set()                     # textures of at least one non-minor model keep full size
    for key, (prims, *_) in models.items():
        for p in prims:
            if p["material"]["texture"]:
                t = norm_texture(p["material"]["texture"])
                tex_alpha[t] |= p["material"]["alpha_mode"] != "OPAQUE"
                if not MINOR.match(key):
                    tex_major.add(t)
    tex_uri = {}
    for t, alpha in sorted(tex_alpha.items()):
        data = textures.read(t)
        if data is None:
            continue
        try:
            rgba = read_dds(data)
        except ValueError:
            continue
        if rgba.shape[2] == 3:
            rgba = np.concatenate([rgba, np.full(rgba.shape[:2] + (1,), 255, np.uint8)], axis=2)
        name = re.sub(r"[^a-z0-9]+", "_", t[len("textures\\"):-4]) + (".png" if alpha else ".jpg")
        save_texture(rgba, os.path.join(a.out, "textures", name),
                     a.texture_size if t in tex_major else a.minor_texture_size, g["gain"])
        tex_uri[t] = "textures/" + name
    print(f"textures: {len(tex_uri)}")

    def image_for(m):
        return tex_uri.get(norm_texture(m["texture"])) if m["texture"] else None

    # group references into tiles; each tile has a GLB of large objects and one of small objects
    tiles = collections.defaultdict(list)
    for p in placed:
        if p[0] in models:
            tiles[(int(p[1][0] // tile_units), int(p[1][1] // tile_units))].append(p)
    tile_nodes = {}
    counts = collections.Counter()
    for (tx, ty), items in sorted(tiles.items()):
        cx, cy = (tx + 0.5) * tile_units, (ty + 0.5) * tile_units
        cz = min(float(p[1][2]) for p in items)
        center = np.array([cx, cy, cz])
        # "" large, "_s" small, "_l" game LOD of minor models, "_f" their full meshes (replace "_l" up close)
        parts = {"": [], "_s": [], "_l": [], "_f": []}
        for p in items:
            radius = models[p[0]][3] * p[3] * UNIT_M
            # small objects with only our own LOD keep loading up close, at full detail
            if p[0] + "|lod" in models and (p[0] in game_lod or radius >= a.small_radius):
                parts["_l"].append((p[0] + "|lod", *p[1:]))
                parts["_f"].append(p)
                continue
            parts["_s" if radius < a.small_radius else ""].append(p)
        regions = {}
        for suffix, group in parts.items():
            if not group:
                continue
            corners = []
            for model, pos, rot, scale, _, _ in group:
                _, lo, hi, _ = models[model]
                c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                corners.append(c @ (rotation(*rot) * scale).T + pos)
            regions[suffix] = region_of(np.concatenate(corners))
            write_tile(os.path.join(a.out, f"t_{tx}_{ty}{suffix}.glb"), group, models, store, image_for, center)
            counts[suffix] += len(group)

        lon0, lat0 = PLACE.to_lonlat(cx * UNIT_M, cy * UNIT_M)
        transform = np.eye(4)
        transform[:3, :3] = enu_frame(float(lon0), float(lat0)).T
        transform[:3, 3] = geodetic_to_ecef(np.array(lon0), np.array(lat0), np.array(cz * UNIT_M))
        def subtree(far, near, error, refine):
            """`far` content, refined into `near` content below `error`."""
            keys = [k for k in (far, near) if k in regions]
            if not keys:
                return None
            node = {"boundingVolume": {"region": union([regions[k] for k in keys])},
                    "geometricError": error, "refine": refine, "children": []}
            if far in regions:
                node["content"] = {"uri": f"t_{tx}_{ty}{far}.glb"}
            if near in regions:
                node["children"].append({"boundingVolume": {"region": regions[near]}, "geometricError": 0,
                                         "content": {"uri": f"t_{tx}_{ty}{near}.glb"}})
            return node

        node = subtree("", "_s", a.small_distance, "ADD")
        lod = subtree("_l", "_f", a.swap_distance, "REPLACE")
        if lod:
            # both subtrees become visible together, at the distance this tile is reached
            node = {"boundingVolume": {"region": union(list(regions.values()))},
                    "geometricError": a.load_distance, "refine": "ADD", "children": [n for n in (node, lod) if n]}
        tile_nodes[(tx, ty)] = (node, transform)
    print(f"tiles: {len(tile_nodes)}  large objects: {counts['']}  small objects: {counts['_s']}  "
          f"minor objects with a LOD: {counts['_l']}")

    def with_transform(node, transform):
        node["transform"] = transform.T.reshape(-1).tolist()      # column-major
        return node

    if a.lod_tileset:
        # LOD blocks (4x4 cells) refine into the full-model tiles they cover
        rel = os.path.relpath(os.path.dirname(os.path.abspath(a.lod_tileset)), os.path.abspath(a.out))
        rel = rel.replace("\\", "/")
        blocks = {}
        for c in json.load(open(a.lod_tileset))["root"]["children"]:
            g = c["children"][0]
            m = re.match(r"b_(-?\d+)_(-?\d+)\.glb$", g["content"]["uri"])
            blocks[(int(m.group(1)), int(m.group(2)))] = (f"{rel}/{g['content']['uri']}",
                                                         np.array(g["transform"]).reshape(4, 4).T,
                                                         g["boundingVolume"]["region"])
        members = collections.defaultdict(list)
        for tx, ty in tile_nodes:
            members[(tx * a.tile_cells // 4 * 4, ty * a.tile_cells // 4 * 4)].append((tx, ty))
        children = []
        for key in sorted(set(blocks) | set(members)):
            uri, lod_t, lod_region = blocks.get(key, (None, np.eye(4), None))
            kids = [with_transform(tile_nodes[t][0], np.linalg.inv(lod_t) @ tile_nodes[t][1]) for t in members[key]]
            regions = [k["boundingVolume"]["region"] for k in kids] + ([lod_region] if lod_region else [])
            block = {"boundingVolume": {"region": union(regions)}, "geometricError": a.load_distance,
                     "refine": "REPLACE", "children": kids}
            if uri:
                with_transform(block, lod_t)["content"] = {"uri": uri}
            children.append({"boundingVolume": {"region": union(regions)}, "geometricError": a.lod_distance,
                             "refine": "ADD", "children": [block]})
    else:
        children = [{"boundingVolume": node["boundingVolume"], "geometricError": a.load_distance,
                     "refine": "ADD", "children": [with_transform(node, transform)]}
                    for node, transform in tile_nodes.values()]

    root_region = union([c["boundingVolume"]["region"] for c in children])
    with open(os.path.join(a.out, "tileset.json"), "w") as f:
        json.dump({"asset": {"version": "1.1"}, "geometricError": 5000,
                   "root": {"boundingVolume": {"region": root_region},
                            "geometricError": 2000, "refine": "ADD", "children": children}}, f)

    if a.labels:
        labels, used = [], set()
        for model, pos, rot, scale, typ, dest in placed:
            name = dest_names.get(dest) if dest else None
            if typ == b"DOOR" and name and name not in used:
                used.add(name)
                lon, lat = PLACE.to_lonlat(pos[0] * UNIT_M, pos[1] * UNIT_M)
                labels.append({"name": name, "lon": round(float(lon), 7), "lat": round(float(lat), 7),
                               "height": round(float(pos[2]) * UNIT_M, 1), "type": "Entrance"})
        with open(a.labels, "w", encoding="utf-8") as f:
            json.dump(sorted(labels, key=lambda l: l["name"]), f, ensure_ascii=False, indent=1)
        print(f"labels: {len(labels)}")


if __name__ == "__main__":
    main()
