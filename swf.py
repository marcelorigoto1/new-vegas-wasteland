"""Minimal SWF reader that renders the first frame of an exported sprite (vector shapes only) to SVG.
Enough for the map-marker icons of Skyrim's interface\\map.swf.

Shapes are flattened into absolute coordinates: every placed object's matrix and colour transform is applied
in Python, so the SVG holds plain paths. Fills are rebuilt from the SWF edge list (each edge carries a
left fill "fill0" and a right fill "fill1") by chaining the edges of each fill style into closed loops.
"""
import struct
import zlib

SHAPE_TAGS = {2: 1, 22: 2, 32: 3, 83: 4}       # DefineShape tag -> shape version


class Bits:
    def __init__(self, data, pos):
        self.data, self.pos, self.bit = data, pos, 0

    def ub(self, n):
        v = 0
        for _ in range(n):
            byte = self.data[self.pos]
            v = (v << 1) | ((byte >> (7 - self.bit)) & 1)
            self.bit += 1
            if self.bit == 8:
                self.bit, self.pos = 0, self.pos + 1
        return v

    def sb(self, n):
        v = self.ub(n)
        return v - (1 << n) if n and v & (1 << (n - 1)) else v

    def fb(self, n):
        return self.sb(n) / 65536.0

    def align(self):
        if self.bit:
            self.bit, self.pos = 0, self.pos + 1

    def u8(self):
        self.align()
        v = self.data[self.pos]
        self.pos += 1
        return v

    def u16(self):
        self.align()
        v = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def rect(self):
        self.align()
        n = self.ub(5)
        r = [self.sb(n) for _ in range(4)]
        self.align()
        return r

    def matrix(self):
        self.align()
        a = d = 1.0
        b = c = 0.0
        if self.ub(1):
            n = self.ub(5)
            a, d = self.fb(n), self.fb(n)
        if self.ub(1):
            n = self.ub(5)
            b, c = self.fb(n), self.fb(n)
        n = self.ub(5)
        tx, ty = self.sb(n), self.sb(n)
        self.align()
        return (a, b, c, d, tx, ty)

    def color(self, alpha):
        self.align()
        rgba = list(self.data[self.pos:self.pos + (4 if alpha else 3)])
        self.pos += 4 if alpha else 3
        return tuple(rgba + ([] if alpha else [255]))

    def cxform(self):
        self.align()
        has_add, has_mult = self.ub(1), self.ub(1)
        n = self.ub(4)
        mult = [self.sb(n) / 256.0 for _ in range(4)] if has_mult else [1.0] * 4
        add = [self.sb(n) for _ in range(4)] if has_add else [0] * 4
        self.align()
        return mult, add


def tags(data, pos, end):
    while pos < end:
        h = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        code, ln = h >> 6, h & 63
        if ln == 63:
            ln = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        yield code, pos, ln
        pos += ln
        if code == 0:
            break


def mat_mul(m, n):
    """m applied after n."""
    a, b, c, d, tx, ty = m
    a2, b2, c2, d2, tx2, ty2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2,
            a * tx2 + c * ty2 + tx, b * tx2 + d * ty2 + ty)


def cx_mul(outer, inner):
    (m1, a1), (m2, a2) = outer, inner
    return [x * y for x, y in zip(m1, m2)], [x * y2 + z for x, y2, z in zip(m1, a2, a1)]


IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
NOCX = ([1.0] * 4, [0] * 4)


class SWF:
    def __init__(self, data):
        if data[:3] == b"CWS":
            data = data[:8] + zlib.decompress(data[8:])
        elif data[:3] != b"FWS":
            raise ValueError("not an SWF file")
        self.data = data
        r = Bits(data, 8)
        r.rect()
        self.chars, self.exports = {}, {}
        for code, p, ln in tags(data, r.pos + 4, len(data)):
            if code in SHAPE_TAGS or code == 39:
                self.chars[struct.unpack_from("<H", data, p)[0]] = (code, p, ln)
            elif code == 56:
                n = struct.unpack_from("<H", data, p)[0]
                q = p + 2
                for _ in range(n):
                    cid = struct.unpack_from("<H", data, q)[0]
                    e = data.index(b"\0", q + 2)
                    self.exports[data[q + 2:e].decode("latin-1")] = cid
                    q = e + 1

    # ---------- shapes ----------
    def _styles(self, r, ver):
        fills, lines = [], []
        n = r.u8()
        if n == 0xFF and ver >= 2:
            n = r.u16()
        for _ in range(n):
            t = r.u8()
            if t == 0x00:
                fills.append(("solid", r.color(ver >= 3)))
            elif t in (0x10, 0x12, 0x13):
                r.matrix()
                r.align()
                spec = r.u8()
                stops = [(r.u8(), r.color(ver >= 3)) for _ in range(spec & 15)]
                if t == 0x13:
                    r.u16()
                # approximate gradients by their mean colour
                avg = tuple(sum(c[1][i] for c in stops) // max(len(stops), 1) for i in range(4))
                fills.append(("solid", avg))
            elif 0x40 <= t <= 0x43:
                r.u16()
                r.matrix()
                fills.append(("none", None))
            else:
                raise ValueError(f"fill style {t:#x}")
        n = r.u8()
        if n == 0xFF and ver >= 2:
            n = r.u16()
        for _ in range(n):
            w = r.u16()
            if ver == 4:
                r.ub(2)
                join = r.ub(2)
                has_fill = r.ub(1)
                r.ub(3)
                r.ub(5)
                r.ub(1)
                r.ub(2)
                if join == 2:
                    r.u16()
                if has_fill:
                    t = r.u8()
                    col = r.color(True) if t == 0 else (255, 255, 255, 255)
                else:
                    col = r.color(True)
            else:
                col = r.color(ver >= 3)
            lines.append((w, col))
        r.align()
        nfb, nlb = r.ub(4), r.ub(4)
        return fills, lines, nfb, nlb

    def shape(self, cid):
        """Return [(kind, style, [loops])] where loops are lists of segments ((x0,y0),(cx,cy)|None,(x1,y1))."""
        code, p, ln = self.chars[cid]
        ver = SHAPE_TAGS[code]
        r = Bits(self.data, p + 2)
        r.rect()
        if ver == 4:
            r.rect()
            r.u8()
        fills, lines, nfb, nlb = self._styles(r, ver)
        groups = []                                # one entry per style array: (fills, lines, edges by style)
        cur = {"fills": fills, "lines": lines, "f": {}, "l": {}}
        groups.append(cur)
        x = y = 0
        f0 = f1 = ln_ = 0
        while True:
            if r.ub(1) == 0:
                flags = r.ub(5)
                if flags == 0:
                    break
                new, sl, sf1, sf0, mv = (flags >> 4) & 1, (flags >> 3) & 1, (flags >> 2) & 1, (flags >> 1) & 1, flags & 1
                if mv:
                    n = r.ub(5)
                    x, y = r.sb(n), r.sb(n)
                if sf0:
                    f0 = r.ub(nfb)
                if sf1:
                    f1 = r.ub(nfb)
                if sl:
                    ln_ = r.ub(nlb)
                if new:
                    fills, lines, nfb, nlb = self._styles(r, ver)
                    cur = {"fills": fills, "lines": lines, "f": {}, "l": {}}
                    groups.append(cur)
                continue
            straight = r.ub(1)
            n = r.ub(4) + 2
            if straight:
                if r.ub(1):
                    dx, dy = r.sb(n), r.sb(n)
                elif r.ub(1):
                    dx, dy = 0, r.sb(n)
                else:
                    dx, dy = r.sb(n), 0
                seg = ((x, y), None, (x + dx, y + dy))
            else:
                cx, cy = x + r.sb(n), y + r.sb(n)
                ax, ay = cx + r.sb(n), cy + r.sb(n)
                seg = ((x, y), (cx, cy), (ax, ay))
            x, y = seg[2]
            if f1:
                cur["f"].setdefault(f1, []).append(seg)
            if f0:
                cur["f"].setdefault(f0, []).append((seg[2], seg[1], seg[0]))
            if ln_:
                cur["l"].setdefault(ln_, []).append(seg)
        out = []
        for g in groups:
            for idx, segs in sorted(g["f"].items()):
                out.append(("fill", g["fills"][idx - 1], _chain(segs)))
            for idx, segs in sorted(g["l"].items()):
                out.append(("line", g["lines"][idx - 1], _chain(segs, closed=False)))
        return out

    # ---------- sprites ----------
    def render(self, cid, m=IDENT, cx=NOCX, skip_names=("hitarea",)):
        """Flatten the first frame of character `cid` into [(kind, style, loops)] in absolute twips."""
        code, p, ln = self.chars[cid]
        if code in SHAPE_TAGS:
            out = []
            for kind, style, loops in self.shape(cid):
                pts = [[(_tx(m, a), None if c is None else _tx(m, c), _tx(m, b)) for a, c, b in lp] for lp in loops]
                if kind == "fill":
                    style = (style[0], _apply_cx(cx, style[1]) if style[1] else None)
                else:
                    scale = (abs(m[0] * m[3] - m[1] * m[2])) ** 0.5
                    style = (style[0] * scale, _apply_cx(cx, style[1]))
                out.append((kind, style, pts))
            return out
        display = {}
        for c2, p2, l2 in tags(self.data, p + 4, p + ln):
            if c2 == 1:
                break
            if c2 != 26:
                continue
            r = Bits(self.data, p2)
            flags = r.u8()
            depth = r.u16()
            obj = dict(display.get(depth, {})) if flags & 0x01 else {}
            if flags & 0x02:
                obj["char"] = r.u16()
            if flags & 0x04:
                obj["m"] = r.matrix()
            if flags & 0x08:
                obj["cx"] = r.cxform()
            if flags & 0x10:
                r.u16()
            if flags & 0x20:
                e = self.data.index(b"\0", r.pos)
                obj["name"] = self.data[r.pos:e].decode("latin-1")
                r.pos = e + 1
            if flags & 0x40:
                obj["clip"] = True
            display[depth] = obj
        out = []
        for depth in sorted(display):
            obj = display[depth]
            if "char" not in obj or obj.get("clip") or obj.get("name", "").lower() in skip_names:
                continue
            if obj["char"] not in self.chars:
                continue
            out += self.render(obj["char"], mat_mul(m, obj.get("m", IDENT)), cx_mul(cx, obj.get("cx", NOCX)),
                               skip_names)
        return out

    def svg(self, name, size=64, pad=0.06, opaque=True):
        """SVG of an exported symbol, cropped to its visible parts. `opaque`: scale the alpha so the most
        opaque part is fully opaque (map icons are drawn semi-transparent and brighten on hover)."""
        items = [it for it in self.render(self.exports[name]) if it[1][1] is not None and it[1][1][3] > 0]
        if opaque:
            k = 255 / max(it[1][1][3] for it in items)
            items = [(kind, (s[0], s[1][:3] + (min(255, round(s[1][3] * k)),)), loops)
                     for kind, s, loops in items]
        xs = [pt[0] for _, _, loops in items for lp in loops for seg in lp for pt in (seg[0], seg[2])]
        ys = [pt[1] for _, _, loops in items for lp in loops for seg in lp for pt in (seg[0], seg[2])]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
        vx, vy = (x0 + x1 - side) / 2, (y0 + y1 - side) / 2
        body = []
        for kind, style, loops in items:
            d = "".join(_path(lp, kind == "fill") for lp in loops)
            if kind == "fill":
                if style[0] != "solid":
                    continue
                r, g, b, a = style[1]
                body.append(f'<path d="{d}" fill="rgb({r},{g},{b})"{_op("fill", a)} fill-rule="evenodd"/>')
            else:
                w, (r, g, b, a) = style
                body.append(f'<path d="{d}" fill="none" stroke="rgb({r},{g},{b})"{_op("stroke", a)} '
                            f'stroke-width="{max(w, 20):.0f}" stroke-linecap="round" stroke-linejoin="round"/>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
                f'viewBox="{vx:.0f} {vy:.0f} {side:.0f} {side:.0f}">{"".join(body)}</svg>')


def _tx(m, pt):
    a, b, c, d, tx, ty = m
    return (a * pt[0] + c * pt[1] + tx, b * pt[0] + d * pt[1] + ty)


def _apply_cx(cx, col):
    mult, add = cx
    return tuple(int(max(0, min(255, round(v * k + s)))) for v, k, s in zip(col, mult, add))


def _op(attr, a):
    return "" if a >= 255 else f' {attr}-opacity="{a / 255:.3f}"'


def _chain(segs, closed=True):
    """Join segments end to start into paths."""
    by_start = {}
    for s in segs:
        by_start.setdefault(s[0], []).append(s)
    used, loops = set(), []
    for s in segs:
        if id(s) in used:
            continue
        loop, cur = [], s
        while cur is not None and id(cur) not in used:
            used.add(id(cur))
            loop.append(cur)
            nxt = [t for t in by_start.get(cur[2], []) if id(t) not in used]
            cur = nxt[0] if nxt else None
        loops.append(loop)
    return loops


def _path(loop, close):
    f = lambda p: f"{p[0]:.1f} {p[1]:.1f}"
    out = [f"M{f(loop[0][0])}"]
    for _, c, b in loop:
        out.append(f"L{f(b)}" if c is None else f"Q{f(c)} {f(b)}")
    return "".join(out) + ("Z" if close else "")
