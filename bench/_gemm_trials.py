"""Extract probe rows and numerical evidence; compare only controls in the same file."""
import csv
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rx = re.compile(r'gemm\s+(\S+)\s+(\S+)\s+M\s+(\d+)\s+(\d+)×(\d+)\s+(\S+):\s+([\d.]+) µs.*?max ([\d.]+) ulp, (\d+) > 1 ulp, ([\d.]+) of.*?, (\d+) non-finite / (\d+) — (\d+) regs, (\d+) B spill, (\d+) B static shared, (\d+) B dynamic — (\S+) round (\d+) weight copies (\d+)')
fields = ['file','model','shape','m','k','n','tile','microseconds','max_ulp','over_one_ulp','scaled_error','nonfinite','samples','registers','spill_bytes','static_shared_bytes','dynamic_shared_bytes','out','round','weight_copies']
rows = []
for file in sorted(root.rglob('*.log')):
    for line in file.read_text().splitlines():
        match = rx.search(line)
        if match: rows.append(dict(zip(fields, [str(file.relative_to(root)), *match.groups()])))
with (root / 'trials.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
groups = defaultdict(list)
for row in rows: groups[row['file'],row['shape'],row['m'],row['tile']].append(row)
print('file shape M tile N median_us min_us max_us max_ulp nonfinite max_spill')
for key, group in sorted(groups.items()):
    values = [float(r['microseconds']) for r in group]
    print(*key, len(values), f'{st.median(values):.3f}', min(values), max(values),
          max(float(r['max_ulp']) for r in group), sum(int(r['nonfinite']) for r in group),
          max(int(r['spill_bytes']) for r in group))
print(f'{len(rows)} timed and f64-checked variant/M/round cases')
for check in ['layout-check','order-check']:
    count = sum(file.read_text().count(check) for file in root.rglob('*.log'))
    print(check, count)
