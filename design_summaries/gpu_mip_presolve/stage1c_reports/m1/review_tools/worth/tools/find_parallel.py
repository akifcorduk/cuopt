"""find_parallel.py <reduced.csr>: rows of a GPRSCSR1 dump that are parallel to another row
(same columns, proportional coefficients to 1e-9), with and without their singleton columns;
and the singleton columns left, by type and cost."""
import struct
import sys
from collections import defaultdict

data = open(sys.argv[1], "rb").read()
assert data[:8] == b"GPRSCSR1"
nrows, ncols, nnz = struct.unpack_from("<qqq", data, 8)
off = 32
row_start = struct.unpack_from("<%dq" % (nrows + 1), data, off)
off += 8 * (nrows + 1)
col_index = struct.unpack_from("<%di" % nnz, data, off)
off += 4 * nnz
value = struct.unpack_from("<%dd" % nnz, data, off)
off += 8 * nnz
row_lower = struct.unpack_from("<%dd" % nrows, data, off)
off += 8 * nrows
row_upper = struct.unpack_from("<%dd" % nrows, data, off)
off += 8 * nrows
col_lower = struct.unpack_from("<%dd" % ncols, data, off)
off += 8 * ncols
col_upper = struct.unpack_from("<%dd" % ncols, data, off)
off += 8 * ncols
cost = struct.unpack_from("<%dd" % ncols, data, off)
off += 8 * ncols
col_type = struct.unpack_from("<%dB" % ncols, data, off)

size = [0] * ncols
for c in col_index:
    size[c] += 1
print("%d rows %d cols %d nnz; singleton columns %d" % (nrows, ncols, nnz, sum(s == 1 for s in size)))


def key(r, core):
    ent = [(col_index[e], value[e]) for e in range(row_start[r], row_start[r + 1])
           if not core or size[col_index[e]] != 1]
    ent.sort()
    if not ent:
        return None
    a0 = ent[0][1]
    return tuple((c, round(v / a0, 9)) for c, v in ent)


for core in (False, True):
    groups = defaultdict(list)
    for r in range(nrows):
        k = key(r, core)
        if k is not None:
            groups[k].append(r)
    pairs = [g for g in groups.values() if len(g) > 1]
    print("%s: %d groups of parallel rows, %d rows in them"
          % ("cores (without singleton columns)" if core else "whole rows", len(pairs),
             sum(len(g) for g in pairs)))
    for g in pairs[:8]:
        for r in g:
            ent = [(col_index[e], value[e], size[col_index[e]], col_type[col_index[e]])
                   for e in range(row_start[r], row_start[r + 1])]
            print("   row %d [%g, %g]: %s" % (r, row_lower[r], row_upper[r],
                                               " ".join("%+g*x%d(n%d,t%d)" % (v, c, s, t) for c, v, s, t in ent[:12])))
        print("   --")
