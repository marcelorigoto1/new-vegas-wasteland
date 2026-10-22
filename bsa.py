"""Minimal reader for Bethesda BSA archives: Fallout 3 / New Vegas (version 104, zlib) and
Skyrim Special Edition (version 105, LZ4 frames)."""
import struct
import zlib


def lz4_frame_decompress(data):
    """Decode an LZ4 frame (as written by Skyrim SE) without the lz4 module."""
    if data[:4] != b"\x04\x22\x4d\x18":
        raise ValueError("not an LZ4 frame")
    flg = data[4]
    pos = 7 + (8 if flg & 0x08 else 0) + (4 if flg & 0x01 else 0)   # descriptor, content size, dict id
    out = bytearray()
    while True:
        bsize = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if bsize == 0:
            break
        size = bsize & 0x7FFFFFFF
        if bsize & 0x80000000:
            out += data[pos:pos + size]
        else:
            _lz4_block(data, pos, pos + size, out)
        pos += size + (4 if flg & 0x10 else 0)                     # optional block checksum
    return bytes(out)


def _lz4_block(src, pos, end, out):
    while pos < end:
        token = src[pos]
        pos += 1
        n = token >> 4
        if n == 15:
            while True:
                b = src[pos]
                pos += 1
                n += b
                if b != 255:
                    break
        out += src[pos:pos + n]
        pos += n
        if pos >= end:
            break
        off = src[pos] | (src[pos + 1] << 8)
        pos += 2
        n = token & 15
        if n == 15:
            while True:
                b = src[pos]
                pos += 1
                n += b
                if b != 255:
                    break
        n += 4
        start = len(out) - off
        if off >= n:
            out += out[start:start + n]
        else:                                   # overlapping copy repeats the last `off` bytes
            out += (out[start:] * (n // off + 1))[:n]


class BSA:
    def __init__(self, path):
        self.path = path
        self.files = {}  # lowercase "folder\\name" -> (offset, size, compressed)
        with open(path, "rb") as f:
            magic, ver, foff, aflags, nfold, nfile, _, fnamelen, _ = struct.unpack("<4sIIIIIIII", f.read(36))
            if magic != b"BSA\0" or ver not in (104, 105):
                raise ValueError(f"unsupported BSA {magic!r} v{ver}")
            self.version = ver
            self.embed_names = bool(aflags & 0x100)
            default_comp = bool(aflags & 0x4)
            if ver == 105:
                folders = [struct.unpack("<QIIQ", f.read(24))[1:2] for _ in range(nfold)]
            else:
                folders = [struct.unpack("<QII", f.read(16))[1:2] for _ in range(nfold)]
            entries = []
            for (count,) in folders:
                n = f.read(1)[0]
                folder = f.read(n).rstrip(b"\0").decode("cp1252")
                for _ in range(count):
                    _, size, off = struct.unpack("<QII", f.read(16))
                    entries.append((folder, size, off))
            names = f.read(fnamelen).split(b"\0")
            for (folder, size, off), name in zip(entries, names):
                comp = default_comp ^ bool(size & 0x40000000)
                key = f"{folder}\\{name.decode('cp1252')}".lower()
                self.files[key] = (off, size & 0x3FFFFFFF, comp)

    def read(self, key):
        off, size, comp = self.files[key.lower()]
        with open(self.path, "rb") as f:
            f.seek(off)
            data = f.read(size)
        if self.embed_names:
            data = data[1 + data[0]:]
        if comp:
            data = lz4_frame_decompress(data[4:]) if self.version == 105 else zlib.decompress(data[4:])
        return data
