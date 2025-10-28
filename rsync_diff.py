#!/usr/bin/env python3
# rsync_diff.py  OLD.tar  NEW.tar  [--block 300]

import sys, mmap, hashlib

def weak_init(block):
    n = len(block)
    s1 = sum(block) & 0xFFFF
    s2 = sum((n - i) * b for i, b in enumerate(block)) & 0xFFFF
    return s1, s2

def weak_roll(s1, s2, out_b, in_b, n):
    s1 = (s1 - out_b + in_b) & 0xFFFF
    s2 = (s2 - n * out_b + s1) & 0xFFFF
    return s1, s2

def key(s1, s2): return (s2 << 16) | s1

def build_sigs(mm, n):
    L = len(mm)
    table = {}
    if L < n: return table
    # non-overlapping blocks on receiver
    for bi, pos in enumerate(range(0, L - n + 1, n)):
        blk = mm[pos:pos+n]
        s1, s2 = weak_init(blk)
        h = hashlib.md5(blk).hexdigest()
        table.setdefault(key(s1, s2), []).append((h, bi))
    return table

def rsync_diff(oldp, newp, n):
    with open(oldp, 'rb') as fo, open(newp, 'rb') as fn:
        oldm = mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ)
        newm = mmap.mmap(fn.fileno(), 0, access=mmap.ACCESS_READ)

        sigs = build_sigs(oldm, n)
        N = len(newm)
        if N < n: return N, 0, N

        s1, s2 = weak_init(newm[0:n])
        i = 0; matched = 0; lits = 0

        while True:
            wk = key(s1, s2)
            if wk in sigs:
                chunk = newm[i:i+n]
                h = hashlib.md5(chunk).hexdigest()
                if any(h == H for H, _ in sigs[wk]):
                    matched += n
                    i += n
                    if i <= N - n:
                        s1, s2 = weak_init(newm[i:i+n])
                        continue
                    break
            # no match → send 1 literal byte and roll
            i += 1; lits += 1
            if i <= N - n:
                s1, s2 = weak_roll(s1, s2, newm[i-1], newm[i+n-1], n)
            else:
                break
        if i < N: lits += (N - i)
        return N, matched, lits

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: rsync_diff.py OLD.tar NEW.tar [--block 300]"); sys.exit(1)
    oldf, newf = sys.argv[1], sys.argv[2]
    bsz = 300
    if len(sys.argv) >= 5 and sys.argv[3] == "--block": bsz = int(sys.argv[4])
    total, matched, literal = rsync_diff(oldf, newf, bsz)
    pct = 100.0 * literal / total if total else 0.0
    print(f"Total (new): {total}")
    print(f"Matched    : {matched}")
    print(f"Literals   : {literal}")
    print(f"Residual % : {pct:.2f}%  (block={bsz})")
