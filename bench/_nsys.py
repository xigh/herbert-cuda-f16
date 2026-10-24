"""Kernel occupancy resources and gaps inside each request's CUDA timeline."""
import json
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
rows = list(c.execute('select k.start,k.end,s.value,k.registersPerThread,k.dynamicSharedMemory,k.localMemoryPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start'))
groups, group = [], []
for row in rows:
    group.append(row)
    if row[2].startswith('argmax_final'):
        groups.append(group)
        group = []
print('GPU timeline spans include gaps; kernel sums exclude gaps. Diagnostic trace, not benchmark.')
for i, kernels in enumerate(groups):
    span = (kernels[-1][1]-kernels[0][0])/1e6
    compute = sum(x[1]-x[0] for x in kernels)/1e6
    gemm = sum(x[1]-x[0] for x in kernels if x[2].startswith('gemm_'))/1e6
    print(json.dumps(dict(request=i, span_ms=span, kernels_ms=compute, gaps_ms=span-compute, gemm_ms=gemm)))
seen = set()
for x in rows:
    if x[2].startswith('gemm_') and x[2] not in seen:
        seen.add(x[2])
        print(json.dumps(dict(kernel=x[2], registers=x[3], dynamic_shared=x[4], local_bytes=x[5])))
