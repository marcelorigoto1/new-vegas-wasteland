"""Minimal reader for Fallout 3 / New Vegas BSA archives (version 104)."""
import struct
import zlib


class BSA:
    def __init__(self, path):
        self.path = path
        self.files = {}  # lowercase "folder\\name" -> (offset, size, compressed)
        with open(path, "rb") as f:
            magic, ver, foff, aflags, nfold, nfile, _, fnamelen, _ = struct.unpack("<4sIIIIIIII", f.read(36))
            if magic != b"BSA\0" or ver != 104:
                raise ValueError(f"unsupported BSA {magic!r} v{ver}")
            self.embed_names = bool(aflags & 0x100)
            default_comp = bool(aflags & 0x4)
            folders = [struct.unpack("<QII", f.read(16)) for _ in range(nfold)]
            entries = []
            for _, count, _ in folders:
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
            data = zlib.decompress(data[4:])
        return data
