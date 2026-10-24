"""Compare displayed driver SASS, normalizing absolute control-flow targets.

Usage: python3 bench/_gemm_sass_compare.py CAPTURE_A CAPTURE_B
Keeps physical registers, operands, predicates and instruction order. This is
not a semantic equivalence checker; unequal code requires inspection. Binary
instruction encodings and scheduling control bits are not part of this export.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def programs(root):
    blocks = []
    current = None
    for row in csv.reader((root/'source.csv').open()):
        if not row:
            continue
        if row[0] == 'Kernel Name':
            current = dict(kernel=row[1], instructions=[])
            blocks.append(current)
        elif row[0].startswith('0x') and current is not None:
            current['instructions'].append((int(row[0], 16), row[1].strip()))
    result = collections.defaultdict(list)
    for block in blocks:
        instructions = block['instructions']
        if not instructions:
            continue
        match = re.match(r'gemm_(\d+)x(\d+)_(bf16|res)_', block['kernel'])
        if not match:
            continue
        base = instructions[0][0]
        limit = instructions[-1][0]+16

        def target(m):
            value = int(m[0], 16)
            return f'PC+{value-base:#x}' if base <= value < limit else m[0]

        normalized = []
        for _, text in instructions:
            words = text.split()
            opcode = words[1] if words[0].startswith('@') else words[0]
            if opcode.split('.')[0] in {'BRA', 'BSSY', 'CALL', 'JMP', 'BRX'}:
                text = re.sub(r'0x[0-9a-fA-F]+', target, text)
            normalized.append(text)
        opcodes = []
        for text in normalized:
            words = text.split()
            opcodes.append(words[1] if words[0].startswith('@') else words[0])
        result['x'.join(match.groups())].append(dict(kernel=block['kernel'],
            instructions=normalized, opcodes=opcodes, opcode_counts=dict(collections.Counter(opcodes)),
            sha256=hashlib.sha256('\n'.join(normalized).encode()).hexdigest()))
    if not result:
        raise ValueError(f'{root}: no driver SASS programs')
    return result


a_root, b_root = map(Path, sys.argv[1:3])
a, b = programs(a_root), programs(b_root)
if a.keys() != b.keys():
    raise ValueError('captured matrix/output sets differ')
rows = []
for shape in sorted(a):
    a_hashes = {p['sha256'] for p in a[shape]}
    b_hashes = {p['sha256'] for p in b[shape]}
    if len(a_hashes) != 1 or len(b_hashes) != 1:
        raise ValueError(f'{shape}: different SASS within one capture')
    pa, pb = a[shape][0], b[shape][0]
    first = next((i for i, (x, y) in enumerate(zip(pa['instructions'], pb['instructions'])) if x != y), None)
    if first is None and len(pa['instructions']) != len(pb['instructions']):
        first = min(len(pa['instructions']), len(pb['instructions']))
    rows.append(dict(shape=shape, captures=[str(a_root), str(b_root)],
        kernels=[pa['kernel'], pb['kernel']], occurrences=[len(a[shape]), len(b[shape])],
        instruction_counts=[len(pa['instructions']), len(pb['instructions'])],
        normalized_sass_sha256=[pa['sha256'], pb['sha256']],
        scope='Displayed instructions/operands/predicates/order, absolute branch targets made relative; not binary control bits',
        identical=pa['instructions'] == pb['instructions'], first_difference=first,
        opcode_order_identical=pa['opcodes'] == pb['opcodes'],
        opcode_counts=[pa['opcode_counts'], pb['opcode_counts']]))
print(json.dumps(rows, indent=2))
