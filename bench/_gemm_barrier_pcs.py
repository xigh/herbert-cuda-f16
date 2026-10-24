"""Locate sampled barrier PCs and nearby synchronizations in driver SASS."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = []
for path in sorted(root.glob('*/source.csv')):
    blocks = []
    current = None
    for row in csv.reader(path.open()):
        if not row: continue
        if row[0] == 'Kernel Name':
            current = dict(kernel=row[1], instructions=[])
            blocks.append(current)
        elif row[0] == 'Address': header = row
        elif row[0].startswith('0x') and current is not None:
            try: samples = float(row[header.index('stall_barrier')].replace(',', ''))
            except ValueError: samples = 0
            current['instructions'].append((int(row[0], 16), row[1].strip(), samples))
    for instance, block in enumerate(blocks):
        instructions = block['instructions']
        if not instructions: continue
        base = instructions[0][0]
        for i, (at, instruction, samples) in enumerate(instructions):
            if not samples: continue
            before = instructions[max(0, i-16):i+1]
            barriers = [(pc, text) for pc, text, _ in before if 'BAR.SYNC' in text]
            result.append(dict(capture=path.parent.name, kernel=block['kernel'], instance=instance,
                sampled_pc=hex(at), sampled_offset=hex(at-base), instruction=instruction,
                barrier_samples=samples,
                nearby_barrier_pc=hex(barriers[-1][0]) if barriers else None,
                nearby_barrier_offset=hex(barriers[-1][0]-base) if barriers else None,
                nearby_barrier_instruction=barriers[-1][1] if barriers else None,
                context=[dict(pc=hex(pc),offset=hex(pc-base),instruction=text)
                         for pc,text,_ in instructions[max(0,i-5):i+3]]))
(root / 'barrier-pcs.json').write_text(json.dumps(result, indent=2)+'\n')
print(len(result), 'sampled instruction locations with adjacent SASS preserved')
