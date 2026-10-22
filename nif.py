"""Minimal NIF reader: header, block table and the geometry blocks needed to export static meshes.
Gamebryo 20.2.0.7 as used by Fallout 3 / New Vegas (NiTriShape, NiTriStrips) and Skyrim Special Edition
(BSTriShape with packed vertex data, BS version 100)."""
import struct

import numpy as np


class Reader:
    def __init__(self, data, pos=0):
        self.d, self.p = data, pos

    def unpack(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v

    def u8(self): return self.unpack("B")[0]
    def u16(self): return self.unpack("H")[0]
    def u32(self): return self.unpack("I")[0]
    def i32(self): return self.unpack("i")[0]
    def f32(self): return self.unpack("f")[0]

    def array(self, dtype, count):
        a = np.frombuffer(self.d, dtype=dtype, count=count, offset=self.p)
        self.p += a.nbytes
        return a

    def sized_string(self):
        n = self.u32()
        s = self.d[self.p:self.p + n].decode("cp1252", "replace")
        self.p += n
        return s

    def short_string(self):
        n = self.u8()
        s = self.d[self.p:self.p + n].rstrip(b"\x00").decode("cp1252", "replace")
        self.p += n
        return s


class Nif:
    def __init__(self, data):
        nl = data.index(b"\n")
        self.header_string = data[:nl].decode()
        r = Reader(data, nl + 1)
        self.version = r.u32()
        r.u8()                                # endian
        self.user_version = r.u32()
        nblocks = r.u32()
        self.bs_version = r.u32() if self.user_version >= 10 else 0
        if self.bs_version:
            for _ in range(3):
                r.short_string()              # creator, export info 1, export info 2
        types = [r.sized_string() for _ in range(r.u16())]
        self.block_types = [types[r.u16() & 0x7FFF] for _ in range(nblocks)]
        sizes = [r.u32() for _ in range(nblocks)]
        nstrings = r.u32()
        r.u32()                               # max string length
        self.strings = [r.sized_string() for _ in range(nstrings)]
        for _ in range(r.u32()):
            r.u32()                           # groups
        self.blocks = []
        pos = r.p
        for t, s in zip(self.block_types, sizes):
            self.blocks.append((t, pos, s))
            pos += s
        self.data = data

    def reader(self, i):
        return Reader(self.data, self.blocks[i][1])

    def string(self, idx):
        return self.strings[idx] if 0 <= idx < len(self.strings) else ""


def read_object_net(r):
    """NiObjectNET: name (string index), extra data refs, controller. Returns the name index."""
    name = r.i32()
    r.array("<i4", r.u32())               # extra data refs
    r.i32()                               # controller
    return name


def read_av_object(r, bs_version):
    """NiObjectNET + NiAVObject transform fields; returns (name_index, translation, rotation 3x3, scale)."""
    name, _, t, rot, scale = read_av_object_flags(r, bs_version)
    return name, t, rot, scale


def read_av_object_flags(r, bs_version):
    name = read_object_net(r)
    flags = r.u32() if bs_version > 26 else r.u16()
    t = np.array(r.unpack("3f"))
    rot = np.array(r.unpack("9f")).reshape(3, 3)
    scale = r.f32()
    return name, flags, t, rot, scale


def read_geometry_data(r):
    """NiGeometryData fields shared by NiTriShapeData / NiTriStripsData (20.2.0.7, Bethesda)."""
    r.i32()                               # group id
    nv = r.u16()
    r.u8(); r.u8()                        # keep / compress flags
    out = {}
    if r.u8():
        out["positions"] = r.array("<f4", nv * 3).reshape(nv, 3)
    vflags = r.u16()
    if r.u8():
        out["normals"] = r.array("<f4", nv * 3).reshape(nv, 3)
        if vflags & 0x1000:
            r.array("<f4", nv * 6)        # tangents + bitangents
    r.unpack("4f")                        # bounding sphere
    if r.u8():
        out["colors"] = r.array("<f4", nv * 4).reshape(nv, 4)
    if vflags & 1:
        out["uvs"] = r.array("<f4", nv * 2).reshape(nv, 2)
    r.u16()                               # consistency flags
    r.i32()                               # additional data
    return out


def read_tri_shape_data(r):
    """NiTriShapeData. Returns dict of numpy arrays."""
    out = read_geometry_data(r)
    ntri = r.u16()
    r.u32()                               # num triangle points
    if r.u8():
        out["triangles"] = r.array("<u2", ntri * 3).reshape(ntri, 3)
    for _ in range(r.u16()):              # match groups
        r.array("<u2", r.u16())
    return out


def read_tri_strips_data(r):
    """NiTriStripsData, with the strips converted to a triangle list."""
    out = read_geometry_data(r)
    r.u16()                               # num triangles
    lengths = [r.u16() for _ in range(r.u16())]
    if r.u8():
        tris = []
        for n in lengths:
            s = r.array("<u2", n).astype(np.int64)
            if n < 3:
                continue
            a, b, c = s[:-2], s[1:-1], s[2:]
            odd = np.arange(n - 2) % 2 == 1
            t = np.stack([a, np.where(odd, c, b), np.where(odd, b, c)], axis=1)
            t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
            tris.append(t)
        if tris:
            out["triangles"] = np.concatenate(tris)
    return out


NODE_TYPES = {"NiNode", "BSFadeNode", "BSValueNode", "BSOrderedNode", "BSMultiBoundNode",
              "NiBillboardNode", "BSBlastNode", "BSDamageStage", "BSDebrisNode", "BSMasterParticleSystem",
              "NiSwitchNode", "NiLODNode", "NiBSPNode", "BSTreeNode", "BSLeafAnimNode"}
SHAPE_TYPES = {"NiTriShape", "NiTriStrips", "BSSegmentedTriShape"}
LIGHTING_SHADERS = {"BSShaderPPLightingProperty", "Lighting30ShaderProperty", "HairShaderProperty"}
UNLIT_SHADERS = {"BSShaderNoLightingProperty", "TileShaderProperty"}


def _material(nif, prop_refs):
    """Summarise the properties of a shape into a simple material description."""
    m = {"texture": None, "alpha_mode": "OPAQUE", "cutoff": 0.5, "alpha": 1.0, "unlit": False}
    for ref in prop_refs:
        if not 0 <= ref < len(nif.blocks):
            continue
        t, pos, size = nif.blocks[ref]
        r = nif.reader(ref)
        if t in LIGHTING_SHADERS or t in UNLIT_SHADERS:
            read_object_net(r)
            r.u16()                       # flags
            r.unpack("3I")                # shader type, flags, flags 2
            r.f32()                       # environment map scale
            r.u32()                       # texture clamp mode
            if t in UNLIT_SHADERS:
                m["texture"] = m["texture"] or r.sized_string() or None
                m["unlit"] = True
            else:
                ts = r.i32()
                if 0 <= ts < len(nif.blocks) and nif.blocks[ts][0] == "BSShaderTextureSet":
                    tr = nif.reader(ts)
                    names = [tr.sized_string() for _ in range(tr.i32())]
                    if names and names[0]:
                        m["texture"] = names[0]
        elif t == "NiTexturingProperty":
            read_object_net(r)
            r.u16()                       # flags
            r.u32()                       # texture count
            if r.u8():                    # has base texture
                src = r.i32()
                if 0 <= src < len(nif.blocks) and nif.blocks[src][0] == "NiSourceTexture":
                    sr = nif.reader(src)
                    read_object_net(sr)
                    if sr.u8():           # use external
                        name = nif.string(sr.i32())
                        m["texture"] = m["texture"] or name or None
        elif t == "NiAlphaProperty":
            read_object_net(r)
            flags = r.u16()
            threshold = r.u8()
            if flags & 0x200:
                m["alpha_mode"], m["cutoff"] = "MASK", max(threshold, 1) / 255.0
            elif flags & 1:
                m["alpha_mode"] = "BLEND"
        elif t == "NiMaterialProperty":
            read_object_net(r)
            rest = pos + size - r.p
            # FO3/NV omit ambient/diffuse: specular, emissive, glossiness, alpha, emissive multiple
            floats = r.unpack(f"{rest // 4}f")
            m["alpha"] = floats[-2] if len(floats) == 9 else floats[-1]
    return m


# ---------- Skyrim Special Edition (BS version 100): BSTriShape with packed vertices ----------
SSE_SHAPE_TYPES = {"BSTriShape", "BSMeshLODTriShape", "BSLODTriShape", "BSSubIndexTriShape"}
EFFECT_SHADERS = {"BSEffectShaderProperty", "BSSkyShaderProperty"}
SKIP_SHADERS = {"BSWaterShaderProperty"}          # water planes are drawn by the terrain


def decode_vertices(r, desc, nv, stride, bs_version):
    """Packed BSVertexData: positions, half-float UVs, byte normals, byte colours."""
    flags = desc >> 44
    raw = r.array("u1", nv * stride).reshape(nv, stride)
    out = {}
    if flags & 0x400 or bs_version < 130:  # Skyrim SE positions are always full precision
        out["positions"] = raw[:, 0:12].copy().view("<f4").reshape(nv, 3)
    else:
        out["positions"] = raw[:, 0:6].copy().view("<f2").reshape(nv, 3).astype(np.float32)
    if flags & 0x2:
        o = ((desc >> 8) & 0xF) * 4
        out["uvs"] = raw[:, o:o + 4].copy().view("<f2").reshape(nv, 2).astype(np.float32)
    if flags & 0x8:
        o = ((desc >> 16) & 0xF) * 4
        out["normals"] = raw[:, o:o + 3].astype(np.float32) / 127.5 - 1
    if flags & 0x20:
        o = ((desc >> 24) & 0xF) * 4
        out["colors"] = raw[:, o:o + 4].astype(np.float32) / 255
    return out


def read_skin_partition_sse(nif, skin):
    """Geometry of a skinned Skyrim SE shape (e.g. trees): the vertex buffer and the triangles of all
    partitions, which live in the NiSkinPartition referenced by the skin instance. Bind pose."""
    if not 0 <= skin < len(nif.blocks) or nif.blocks[skin][0] not in ("NiSkinInstance", "BSDismemberSkinInstance"):
        return None
    sr = nif.reader(skin)
    sr.i32()                              # skin data
    part = sr.i32()
    if not 0 <= part < len(nif.blocks) or nif.blocks[part][0] != "NiSkinPartition":
        return None
    r = nif.reader(part)
    nparts = r.u32()
    size, vsize = r.u32(), r.u32()
    desc = r.unpack("Q")[0]
    if not size or not vsize:
        return None
    out = decode_vertices(r, desc, size // vsize, vsize, nif.bs_version)
    tris = []
    for _ in range(nparts):
        nv, ntri, nbones, nstrips, nweights = r.unpack("5H")
        r.array("<u2", nbones)
        if r.u8():
            r.array("<u2", nv)            # vertex map
        if r.u8():
            r.array("<f4", nv * nweights)
        lengths = r.array("<u2", nstrips)
        if r.u8():
            if nstrips:
                for n in lengths:
                    r.array("<u2", n)
            else:
                r.array("<u2", ntri * 3)  # partition triangles
        if r.u8():
            r.array("u1", nv * nweights)  # bone indices
        r.u8(); r.u8()                    # LOD level, global VB
        r.unpack("Q")                     # vertex desc
        # the copy indexes the shared vertex buffer above
        tris.append(r.array("<u2", ntri * 3).reshape(ntri, 3))
    if not tris:
        return None
    out["triangles"] = np.concatenate(tris)
    return out


def read_bs_tri_shape(r, t, nif):
    """BSTriShape geometry after the NiAVObject fields: (shader ref, alpha ref, data dict) or None."""
    r.unpack("4f")                        # bounding sphere
    skin, shader, alpha = r.i32(), r.i32(), r.i32()
    desc = r.unpack("Q")[0]
    ntri, nv, size = r.u16(), r.u16(), r.u32()
    if skin != -1:
        out = read_skin_partition_sse(nif, skin)
        return (shader, alpha, out) if out else None
    if size == 0:
        return None
    out = decode_vertices(r, desc, nv, (desc & 0xF) * 4, nif.bs_version)
    tris = r.array("<u2", ntri * 3).reshape(ntri, 3)
    if t in ("BSMeshLODTriShape", "BSLODTriShape"):
        r.array("u1", r.u32())            # particle data size (SSE), normally 0
        lod0 = r.u32()                    # triangles of the most detailed level come first
        tris = tris[:lod0] if lod0 else tris
    out["triangles"] = tris
    return shader, alpha, out


def _material_sse(nif, shader, alpha):
    """Material of a Skyrim SE shape, or None for shapes not to draw (water)."""
    m = {"texture": None, "alpha_mode": "OPAQUE", "cutoff": 0.5, "alpha": 1.0, "unlit": False,
         "vertex_colors": False, "uv": None}
    if 0 <= shader < len(nif.blocks):
        t = nif.blocks[shader][0]
        r = nif.reader(shader)
        if t in SKIP_SHADERS:
            return None
        if t == "BSLightingShaderProperty":
            r.u32()                       # shader type (precedes the name in Skyrim)
            read_object_net(r)
            _, flags2 = r.u32(), r.u32()
            offset, scale = r.unpack("2f"), r.unpack("2f")
            ts = r.i32()
            r.unpack("4f")                # emissive colour, multiple
            r.u32()                       # texture clamp mode
            m["alpha"] = r.f32()
            m["vertex_colors"] = bool(flags2 & 0x20)
            if offset != (0, 0) or scale != (1, 1):
                m["uv"] = (offset, scale)
            if 0 <= ts < len(nif.blocks) and nif.blocks[ts][0] == "BSShaderTextureSet":
                tr = nif.reader(ts)
                names = [tr.sized_string() for _ in range(tr.i32())]
                m["texture"] = names[0] if names and names[0] else None
        elif t in EFFECT_SHADERS:
            read_object_net(r)
            r.unpack("2I")                # flags
            r.unpack("4f")                # uv offset, scale
            m["texture"] = r.sized_string() or None
            m["alpha_mode"], m["unlit"] = "BLEND", True
    if 0 <= alpha < len(nif.blocks) and nif.blocks[alpha][0] == "NiAlphaProperty":
        r = nif.reader(alpha)
        read_object_net(r)
        flags = r.u16()
        threshold = r.u8()
        if flags & 0x200:
            m["alpha_mode"], m["cutoff"] = "MASK", max(threshold, 1) / 255.0
        elif flags & 1:
            m["alpha_mode"] = "BLEND"
    if m["alpha"] < 1 and m["alpha_mode"] == "OPAQUE":
        m["alpha_mode"] = "BLEND"
    return m


def extract_meshes(nif):
    """Flatten a NIF scene into primitives in model space (game units).

    Returns a list of dicts with positions, normals, uvs, colors, triangles and material."""
    out = []
    seen = set()

    def visit(i, parent):
        if not 0 <= i < len(nif.blocks) or i in seen:
            return
        seen.add(i)
        t = nif.blocks[i][0]
        sse = nif.bs_version > 34         # Skyrim: no property list, shader and alpha refs on the shape
        if t not in NODE_TYPES and t not in (SSE_SHAPE_TYPES if sse else SHAPE_TYPES):
            return
        r = nif.reader(i)
        name, flags, tr, rot, scale = read_av_object_flags(r, nif.bs_version)
        if flags & 1 or "editormarker" in nif.string(name).lower():
            return                        # hidden or editor-only
        props = [] if sse else list(r.array("<i4", r.u32()))
        r.i32()                           # collision object
        # Gamebryo uses row vectors: v' = v * (s R) + t, i.e. column form s R^T v + t
        local = np.eye(4)
        local[:3, :3] = scale * rot.T
        local[:3, 3] = tr
        world = parent @ local
        if t in NODE_TYPES:
            children = list(r.array("<i4", r.u32()))
            if t in ("NiSwitchNode", "NiLODNode") and children:
                children = children[:1]   # first child is the active / most detailed one
            for c in children:
                visit(c, world)
            return
        if sse:
            shape = read_bs_tri_shape(r, t, nif)
            if shape is None:
                return
            shader, alpha, g = shape
            material = _material_sse(nif, shader, alpha)
            if material is None:
                return
            if "colors" in g and not material["vertex_colors"]:
                del g["colors"]           # colours present but not used by the shader
            elif "colors" in g:
                g["colors"][:, 3] = 1     # vertex alpha holds wind / animation weights, not opacity
            if "uvs" in g and material["uv"]:
                (ou, ov), (su, sv) = material["uv"]
                g["uvs"] = g["uvs"] * np.array([su, sv], np.float32) + np.array([ou, ov], np.float32)
        else:
            data = r.i32()
            if not 0 <= data < len(nif.blocks):
                return
            dt = nif.blocks[data][0]
            if dt == "NiTriShapeData":
                g = read_tri_shape_data(nif.reader(data))
            elif dt == "NiTriStripsData":
                g = read_tri_strips_data(nif.reader(data))
            else:
                return
            material = _material(nif, props)
        if "positions" not in g or "triangles" not in g or len(g["triangles"]) == 0:
            return
        pos = g["positions"].astype(np.float64) @ world[:3, :3].T + world[:3, 3]
        prim = {"positions": pos, "triangles": g["triangles"].astype(np.uint32), "material": material}
        if "normals" in g:
            n = g["normals"] @ world[:3, :3].T
            prim["normals"] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        if "uvs" in g:
            prim["uvs"] = g["uvs"]
        if "colors" in g:
            prim["colors"] = g["colors"]
        out.append(prim)

    visit(0, np.eye(4))
    return out
