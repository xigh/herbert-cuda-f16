"""Summarize exported Nsight Compute counters; never benchmark request JSONs."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def load_capture(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    units = next((r for r in rows if not r.get('ID')), {})
    samples = [r for r in rows if r.get('ID')]
    if not samples:
        raise ValueError(f'{path}: no profiled launches')
    groups = {}
    for r in samples:
        key = (r['Kernel Name'], r['Grid Size'], r['Block Size'])
        groups.setdefault(key, []).append(r)
    meta = json.loads((path.parent / 'capture.json').read_text())
    if len(samples) != meta['launch_count']:
        raise ValueError(f'{path}: {len(samples)} launches, expected {meta["launch_count"]}')
    result = []
    for (kernel, grid, block), launches in groups.items():
        metrics = {}
        for name in reader.fieldnames:
            if '__' not in name:
                continue
            values = []
            for r in launches:
                try:
                    v = float(r[name].replace(',', ''))
                    if math.isfinite(v):
                        values.append(v)
                except (ValueError, AttributeError):
                    pass
            if values:
                metrics[name] = dict(unit=units.get(name, ''), n=len(values),
                    median=statistics.median(values), min=min(values), max=max(values))
        result.append(dict(kernel=kernel, grid=grid, block=block,
            launches=len(launches), metrics=metrics))
    return dict(capture=path.parent.name, metadata=meta, kernels=result)


def main():
    root = Path(sys.argv[1])
    captures = [load_capture(p) for p in sorted(root.glob('*/metrics.csv'))
                if (p.parent / 'capture.json').exists()]
    if not captures:
        raise ValueError('No counter CSVs found')
    (root / 'summary.json').write_text(json.dumps(dict(schema=1,
        diagnostic_only=True, captures=captures), indent=2) + '\n')
    names = [
        'gpu__time_duration.sum',
        'gpc__cycles_elapsed.avg.per_second',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'smsp__warps_eligible.avg.per_cycle_active',
        'smsp__issue_active.avg.pct_of_peak_sustained_active',
        'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__bytes_read.sum',
        'lts__t_sector_op_read_hit_rate.pct',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
        'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio',
    ]
    lines = ['Diagnostic Nsight counters. Medians summarize distinct layer occurrences, not repeated benchmark trials.']
    for c in captures:
        lines.append(f'\n{c["capture"]}: {json.dumps(c["metadata"], sort_keys=True)}')
        for k in c['kernels']:
            lines.append(f'  {k["kernel"]} grid={k["grid"]} n={k["launches"]}')
            for name in names:
                m = k['metrics'].get(name)
                if m:
                    lines.append(f'    {name} = {m["median"]:.6g} {m["unit"]} [{m["min"]:.6g}, {m["max"]:.6g}]')
    report = '\n'.join(lines) + '\n'
    (root / 'summary.txt').write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
