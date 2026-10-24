"""Recompute counter comparisons and PC attribution from preserved CSVs."""
import collections
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
summary = json.loads((root / 'summary.json').read_text())
comparisons = []
for c in summary['captures']:
    for k in c['kernels']:
        match = re.fullmatch(r'gemm_(\d+)x(\d+)_(bf16|res)_(\d+)x(\d+)x(\d+)_w\d+x\d+_s\d+(_xor)?(?:_sched\d+)?(?:_pipe\d+)?', k['kernel'])
        kd, n, out, bm, bn, bk, _ = match.groups()
        kd, n, bm, bn = map(int, (kd, n, bm, bn))
        m = c['metadata']['first_block_m']
        metrics = k['metrics']
        get = lambda key: metrics.get(key, {}).get('median')
        # Invalid A rows in the final tile use cp.async source_size=0.
        # B remains a full tile for each row tile, including the final one.
        row_tiles = (m + bm - 1) // bm
        column_tiles = n // bn
        ideal = (m * column_tiles + row_tiles * n) * kd * 2 / 32
        sectors = get('lts__t_sectors_srcunit_tex_op_read.sum')
        comparisons.append(dict(capture=c['capture'], kernel=k['kernel'],
            role='gate_up' if n == 18432 else 'down' if kd == 9216 else 'dn_projection',
            m=m, n=n, k=kd, launches=k['launches'],
            duration_us=get('gpu__time_duration.sum') / 1000,
            gpc_mhz=get('gpc__cycles_elapsed.avg.per_second') / 1e6,
            occupancy_pct=get('sm__warps_active.avg.pct_of_peak_sustained_active'),
            math_throttle=get('smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'),
            math_throttle_unit=metrics.get('smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio', {}).get('unit'),
            barrier=get('smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'),
            barrier_unit=metrics.get('smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio', {}).get('unit'),
            long_scoreboard=get('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),
            long_scoreboard_unit=metrics.get('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio', {}).get('unit'),
            registers=get('launch__registers_per_thread'),
            dynamic_shared=get('launch__shared_mem_per_block_dynamic'),
            occupancy_limit_shared=get('launch__occupancy_limit_shared_mem'),
            occupancy_limit_registers=get('launch__occupancy_limit_registers'),
            tensor_pct=get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
            dram_read_bytes=get('dram__bytes_read.sum'), l2_read_sectors=sectors,
            ideal_ab_read_sectors=ideal if out == 'bf16' else None,
            ab_read_inflation=sectors / ideal if out == 'bf16' and sectors is not None else None))

pc = []
for path in sorted(root.glob('*/source.csv')):
    totals = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for row in csv.reader(path.open()):
        if not row:
            continue
        if row[0] == 'Kernel Name':
            kernel = row[1]
            continue
        if row[0] == 'Address':
            header = row
            continue
        if not row[0].startswith('0x'):
            continue
        words = row[1].strip().split()
        instruction = words[1] if words[0].startswith('@') else words[0]
        for name in ['stall_math', 'stall_barrier', 'stall_long_sb',
                     'L1 Wavefronts Shared Excessive', 'L2 Theoretical Sectors Global Excessive']:
            try:
                value = float(row[header.index(name)].replace(',', ''))
            except ValueError:
                continue
            if value:
                totals[kernel][name][instruction] += value
    for kernel, metrics in totals.items():
        pc.append(dict(capture=path.parent.name, kernel=kernel,
            instruction_totals={name: dict(values.most_common()) for name, values in metrics.items()}))

(root / 'analysis.json').write_text(json.dumps(dict(comparisons=comparisons, pc_attribution=pc), indent=2) + '\n')
with (root / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
for row in comparisons:
    if row['role'] == 'gate_up':
        print(row['capture'], 'us=', row['duration_us'], 'L2 read inflation=', row['ab_read_inflation'])
