"""Mosaic the terrain LOD diffuse textures of a worldspace into one RGB image."""
import re

import numpy as np

from bsa import BSA


def _bc1_colors(blocks, force_four):
    """Decode the 8-byte BC1 colour part of each block into (n, 16, 3) int32."""
    c0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    c1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    idx = blocks[:, 4:8].copy().view("<u4")[:, 0]

    def rgb565(c):
        r = ((c >> 11) & 31) * 255 // 31
        g = ((c >> 5) & 63) * 255 // 63
        b = (c & 31) * 255 // 31
        return np.stack([r, g, b], axis=1).astype(np.int32)

    p0, p1 = rgb565(c0), rgb565(c1)
    four = np.ones((len(blocks), 1), bool) if force_four else (c0 > c1)[:, None]
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    palette = np.stack([p0, p1, p2, p3], axis=1)            # (n, 4, 3)
    sel = (idx[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3  # (n, 16)
    return np.take_along_axis(palette, sel[:, :, None].astype(np.int64), axis=1)


def _unblock(px, width, height):
    c = px.shape[-1]
    return px.reshape(height // 4, width // 4, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(height, width, c)


def decode_dxt1(data, width, height):
    """Decode a DXT1 (BC1) top mip level into an (h, w, 3) uint8 array."""
    blocks = np.frombuffer(data, dtype=np.uint8, count=width * height // 2).reshape(-1, 8)
    return _unblock(_bc1_colors(blocks, False), width, height).astype(np.uint8)


def decode_dxt3(data, width, height):
    """Decode a DXT3 (BC2) top mip level into an (h, w, 4) uint8 array."""
    blocks = np.frombuffer(data, dtype=np.uint8, count=width * height).reshape(-1, 16)
    alpha = blocks[:, :8].copy().view("<u8")[:, 0]
    a = ((alpha[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 15).astype(np.int32) * 17
    rgba = np.concatenate([_bc1_colors(blocks[:, 8:], True), a[:, :, None]], axis=2)
    return _unblock(rgba, width, height).astype(np.uint8)


def decode_dxt5(data, width, height):
    """Decode a DXT5 (BC3) top mip level into an (h, w, 4) uint8 array."""
    blocks = np.frombuffer(data, dtype=np.uint8, count=width * height).reshape(-1, 16)
    a0 = blocks[:, 0].astype(np.int32)
    a1 = blocks[:, 1].astype(np.int32)
    bits = np.zeros(len(blocks), np.uint64)
    for i in range(6):
        bits |= blocks[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    sel = ((bits[:, None] >> (3 * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
    k = np.arange(8)[None, :]
    eight = (a0[:, None] * (7 - (k - 1)) + a1[:, None] * (k - 1)) // 7     # used for k >= 2
    six = (a0[:, None] * (5 - (k - 1)) + a1[:, None] * (k - 1)) // 5
    six = np.where(k == 6, 0, np.where(k == 7, 255, six))
    pal = np.where((a0 > a1)[:, None], eight, six)
    pal[:, 0], pal[:, 1] = a0, a1
    a = np.take_along_axis(pal, sel, axis=1)
    rgba = np.concatenate([_bc1_colors(blocks[:, 8:], True), a[:, :, None]], axis=2)
    return _unblock(rgba, width, height).astype(np.uint8)


def decode_uncompressed(data, width, height, bits, masks):
    """Decode an uncompressed RGB(A)/luminance DDS top level using its channel bit masks."""
    bpp = bits // 8
    raw = np.frombuffer(data, np.uint8, count=width * height * bpp).reshape(height, width, bpp)
    v = np.zeros((height, width), np.uint32)
    for i in range(bpp):
        v |= raw[:, :, i].astype(np.uint32) << (8 * i)
    out = []
    for m in masks:
        if m == 0:
            out.append(np.full((height, width), 255, np.uint8))
            continue
        shift = (m & -m).bit_length() - 1
        maxv = m >> shift
        out.append(((v & m) >> shift).astype(np.float32) * 255 / maxv)
    r, g, b, a = (np.asarray(c, np.uint8) for c in out)
    if masks[1] == 0 and masks[2] == 0:   # luminance
        g = b = r
    return np.stack([r, g, b, a], axis=2)


def read_dds(data):
    height, width = (int(v) for v in np.frombuffer(data, "<u4", 2, 12))
    pf_flags = int(np.frombuffer(data, "<u4", 1, 80)[0])
    fourcc = data[84:88]
    if pf_flags & 0x4:
        if fourcc == b"DXT1":
            return decode_dxt1(data[128:], width, height)
        if fourcc == b"DXT3":
            return decode_dxt3(data[128:], width, height)
        if fourcc == b"DXT5":
            return decode_dxt5(data[128:], width, height)
        raise ValueError(f"unsupported DDS format {fourcc!r}")
    bits = int(np.frombuffer(data, "<u4", 1, 88)[0])
    r, g, b, a = (int(v) for v in np.frombuffer(data, "<u4", 4, 92))
    if not pf_flags & 0x1:
        a = 0
    return decode_uncompressed(data[128:], width, height, bits, (r, g, b, a))


def lod_mosaic(bsa_paths, world, level=4, rotated=True):
    """Return (rgb, x0, y0, cells_per_px) where rgb row 0 is the north edge of cell row y0+ny-1
    and (x0, y0) is the south-west cell of the mosaic.

    Reads FO3/FNV tiles (textures\\landscape\\lod\\<w>\\diffuse\\<w>.n.level4.x0.y0.dds) and Skyrim
    tiles (textures\\terrain\\<w>\\<w>.4.0.0.dds) from one or more archives. `rotated`: the tiles are
    stored rotated 180 degrees (FNV)."""
    if isinstance(bsa_paths, str):
        bsa_paths = [bsa_paths]
    w = re.escape(world.lower())
    pat = re.compile(rf"textures\\(?:landscape\\lod\\{w}\\diffuse\\{w}\.n\.level{level}|terrain\\{w}\\{w}\.{level})"
                     rf"\.x?(-?\d+)\.y?(-?\d+)\.dds$")
    tiles = {}
    for path in bsa_paths:
        b = BSA(path)
        for k in b.files:
            m = pat.match(k)
            if m:
                tiles[(int(m.group(1)), int(m.group(2)))] = (b, k)
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + level, max(ys) + level
    b, k = next(iter(tiles.values()))
    size = read_dds(b.read(k)).shape[0]
    ppc = size // level                                     # pixels per cell
    out = np.zeros(((y1 - y0) * ppc, (x1 - x0) * ppc, 3), np.uint8)
    for (x, y), (b, k) in tiles.items():
        img = read_dds(b.read(k))[..., :3]
        if rotated:
            img = img[::-1, ::-1]
        r = (y1 - (y + level)) * ppc
        c = (x - x0) * ppc
        out[r:r + size, c:c + size] = img
    return out, x0, y0, ppc
