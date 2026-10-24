"""Summarize observed NVML clocks; do not rescale benchmark latencies.

Usage: python3 bench/_gemm_telemetry.py GPU.csv [START [END]]
Optional bounds are host-local ISO timestamps (Europe/Paris), inclusive START,
exclusive END. Summaries describe observations, not the clock of every kernel.
"""
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

path = Path(sys.argv[1])
content = path.read_bytes()
start = dt.datetime.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
end = dt.datetime.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
rows = []
for raw in csv.DictReader(content.decode().splitlines(), skipinitialspace=True):
    if raw.get('timestamp') == 'timestamp':
        continue
    time = dt.datetime.strptime(raw['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
    if start is not None and time < start or end is not None and time >= end:
        continue
    rows.append(dict(time=time, sm=int(raw['clocks.current.sm [MHz]'].split()[0]),
                     memory=int(raw['clocks.current.memory [MHz]'].split()[0]),
                     temperature=int(raw['temperature.gpu']),
                     power=float(raw['power.draw [W]'].split()[0]),
                     events=int(raw['clocks_event_reasons.active'], 16)))
if not rows:
    raise SystemExit('no observations in requested interval')
clocks = collections.Counter(r['sm'] for r in rows)
nonidle = [r for r in rows if not r['events'] & 0x1]
gaps = [(b['time']-a['time']).total_seconds() for a, b in zip(rows, rows[1:])]
result = dict(file=str(path), sha256=hashlib.sha256(content).hexdigest(),
              timezone='Europe/Paris (host-local timestamps)', observations=len(rows),
              first=str(rows[0]['time']), last=str(rows[-1]['time']),
              maximum_gap_seconds=max(gaps) if gaps else None,
              sm_mhz=dict(min=min(clocks), max=max(clocks),
                          median=st.median(r['sm'] for r in rows),
                          histogram=dict(sorted(clocks.items())),
                          fraction_at_2805=clocks[2805]/len(rows)),
              memory_mhz=dict(collections.Counter(r['memory'] for r in rows)),
              maximum_temperature_c=max(r['temperature'] for r in rows),
              maximum_power_w=max(r['power'] for r in rows),
              event_masks=dict(collections.Counter(hex(r['events']) for r in rows)),
              power_cap_observations=sum(bool(r['events'] & 0x4) for r in rows),
              sw_thermal_observations=sum(bool(r['events'] & 0x20) for r in rows),
              hw_thermal_observations=sum(bool(r['events'] & 0x40) for r in rows),
              note='NVML samples do not prove constant clocks between observations; timings are not rescaled.')
if nonidle:
    result['observations_without_idle_event'] = dict(
        n=len(nonidle), min_sm_mhz=min(r['sm'] for r in nonidle),
        median_sm_mhz=st.median(r['sm'] for r in nonidle),
        fraction_at_2805=sum(r['sm'] == 2805 for r in nonidle)/len(nonidle))
print(json.dumps(result, indent=2))
