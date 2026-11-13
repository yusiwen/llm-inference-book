#!/usr/bin/env python3
"""gguf-inspect.py - print a GGUF file's map, so the chapter's numbers are read from a file and not typed.

This is a measuring instrument, not the deliverable. The chapter's exercise is a Rust parser of your own;
this script exists for three reasons:

  * the tables in the chapter are generated from a real file (offsets, counts, the alignment, the blob size),
  * the reader has an oracle: if your parser disagrees with this output, one of the two is wrong and the file
    settles it,
  * "does the file's own arithmetic close?" is a question you can answer in one command - the tensor table
    must tile the data blob exactly, and the last tensor must end at or before the file size.

Usage:
    python3 scripts/gguf-inspect.py FILE [--tensors] [--meta] [--hex]

Only the container is parsed here: no tensor is decoded, and the block layouts behind those byte lengths are
Chapter 4's subject. The sizes below are the format's own (elements per block, bytes per block).
"""

import sys
import struct

GGUF_MAGIC = b"GGUF"

VALUE_TYPES = {
    0: ("UINT8", "<B", 1), 1: ("INT8", "<b", 1), 2: ("UINT16", "<H", 2), 3: ("INT16", "<h", 2),
    4: ("UINT32", "<I", 4), 5: ("INT32", "<i", 4), 6: ("FLOAT32", "<f", 4), 7: ("BOOL", "<B", 1),
    8: ("STRING", None, None), 9: ("ARRAY", None, None), 10: ("UINT64", "<Q", 8),
    11: ("INT64", "<q", 8), 12: ("FLOAT64", "<d", 8),
}

# ggml type -> (name, elements per block, bytes per block). The block sizes are the format's:
# a Q4_K block is 256 weights in 144 bytes, and that is the whole reason 4.5 bits per weight is real.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 30: ("BF16", 1, 2),
}


class Reader:
    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def take(self, n):
        if self.pos + n > len(self.buf):
            raise EOFError(f"wanted {n} bytes at {self.pos}, file ends at {len(self.buf)}")
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def value(self, vtype):
        name, fmt, size = VALUE_TYPES[vtype]
        if name == "STRING":
            return self.string()
        if name == "ARRAY":
            etype = self.u32()
            count = self.u64()
            if etype == 8:  # strings: the tokenizer's vocabulary lives here, and it is most of the metadata
                return [self.string() for _ in range(count)]
            f = VALUE_TYPES[etype][1]
            step = VALUE_TYPES[etype][2]
            return list(struct.unpack("<" + f[1] * count, self.take(step * count)))
        return struct.unpack(fmt, self.take(size))[0]


def align_up(n, a):
    return (n + a - 1) // a * a


def inspect(path, show_tensors, show_meta, show_hex):
    buf = open(path, "rb").read()
    r = Reader(buf)
    if r.take(4) != GGUF_MAGIC:
        raise SystemExit(f"{path}: not a GGUF file (magic is {buf[:4]!r})")
    version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
    print(f"{path}")
    print(f"  size on disk      {len(buf)} bytes ({len(buf)/1e9:.3f} GB)")
    print(f"  magic/version     GGUF v{version}")
    print(f"  tensor count      {n_tensors}")
    print(f"  metadata entries  {n_kv}")
    if show_hex:
        print("  first 64 bytes    " + " ".join(f"{b:02x}" for b in buf[:64]))

    meta, alignment = {}, 32
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        val = r.value(vtype)
        meta[key] = (VALUE_TYPES[vtype][0], val)
        if key == "general.alignment":
            alignment = val
    kv_end = r.pos
    print(f"  metadata ends at  byte {kv_end}")
    print(f"  alignment         {alignment}")

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        ttype, off = r.u32(), r.u64()
        tensors.append((name, dims, ttype, off))
    infos_end = r.pos
    blob = align_up(infos_end, alignment)
    print(f"  tensor table ends byte {infos_end}")
    print(f"  data blob starts  byte {blob}  (aligned up from {infos_end}, padding {blob - infos_end})")
    print(f"  data blob size    {len(buf) - blob} bytes")

    # the arithmetic that has to close: every tensor's range inside the blob, no gaps
    order = sorted(tensors, key=lambda t: t[3])
    cursor, gaps, bad = 0, [], []
    for name, dims, ttype, off in order:
        tn, nb, bb = GGML_TYPES.get(ttype, (f"type{ttype}", 1, 1))
        numel = 1
        for d in dims:
            numel *= d
        if numel % nb:
            bad.append(f"{name}: {numel} elements is not a multiple of {nb}")
        length = numel // nb * bb
        if off != cursor:
            gaps.append((name, off - cursor))
        cursor = off + length
    print(f"  tensors tile the   blob: {'yes' if not gaps and not bad else 'NO'}"
          f" (last ends at blob+{cursor}, blob is {len(buf) - blob} bytes)")
    if gaps:
        print(f"  gaps found:        {gaps[:4]}")
    for b in bad:
        print(f"  [E] {b}")

    if show_meta:
        print("\nmetadata")
        for k, (vt, v) in meta.items():
            shown = v if not isinstance(v, list) else f"[{len(v)} x {type(v[0]).__name__}] {v[:3]}..."
            print(f"  {k:44s} {vt:8s} {shown}")

    if show_tensors:
        print("\ntensor table (name, dtype, shape as stored, block length, offset, absolute)")
        total = 0
        for name, dims, ttype, off in order:
            tn, nb, bb = GGML_TYPES.get(ttype, (f"type{ttype}", 1, 1))
            numel = 1
            for d in dims:
                numel *= d
            length = numel // nb * bb
            total += length
            shape = "x".join(str(d) for d in dims)
            print(f"  {name:34s} {tn:6s} {shape:18s} {length:>10} {off:>12} {blob + off:>12}")
        print(f"  {'':34s} {'':6s} {'':18s} {total:>10} total tensor bytes")

    return meta, tensors, blob


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    inspect(args[0], "--tensors" in sys.argv, "--meta" in sys.argv, "--hex" in sys.argv)
