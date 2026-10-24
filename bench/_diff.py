# _diff.py OUTDIR — first divergence of each CUDA run against the CPU oracle.
import sys, glob, os, re
out = sys.argv[1]
invalid = False
def ids(f):
    for line in open(f, errors="replace"):
        if line.startswith("[ids]"):
            return [int(x) for x in line.split()[1:]]
    return None
def turn(f):
    for line in open(f, errors="replace"):
        if line.startswith("[turn"):
            m = re.search(r"(\d+) generated in ([\d.]+) s \(([\d.]+) tok/s\)", line)
            p = re.search(r"(\d+) prompt tokens in ([\d.]+) s \(([\d.]+) tok/s\)", line)
            if not m or not p:
                return "no [turn] line"
            return f"prompt {p.group(1)} @ {p.group(3)} tok/s · gen {m.group(1)} @ {m.group(3)} tok/s"
    return "no [turn] line"
cpus = sorted(glob.glob(os.path.join(out, "*-cpu.log")))
if not cpus:
    raise SystemExit("no CPU logs")
for cpu in cpus:
    name = os.path.basename(cpu)[:-len("-cpu.log")]
    ref = ids(cpu)
    if not ref or turn(cpu) == "no [turn] line":
        print(f"{name}: INVALID CPU reference (missing/empty IDs or turn)")
        invalid = True
        continue
    print(f"{name}: cpu {len(ref) if ref else None} tokens · {turn(cpu)}")
    for vk in sorted(glob.glob(os.path.join(out, f"{name}-cuda-*.log"))):
        got = ids(vk)
        dev = os.path.basename(vk)[len(name) + 6:-4]
        auto_file = os.path.join(out, f"{name}-cuda-{dev.split('-')[0]}-auto.log")
        auto_ids = ids(auto_file) if os.path.isfile(auto_file) else None
        if not got or turn(vk) == "no [turn] line":
            print(f"  {dev}: INVALID (missing/empty IDs or turn)")
            invalid = True
            continue
        n = min(len(ref), len(got))
        first = next((i for i in range(n) if ref[i] != got[i]), None)
        if first is None and len(ref) == len(got):
            verdict = f"identical ({len(got)} tokens)"
        elif first is None:
            verdict = f"identical prefix of {n}, lengths {len(ref)} vs {len(got)}"
        else:
            verdict = f"first divergence at token {first} of {len(ref)}"
        print(f"  {dev}: {verdict} · {turn(vk)}")
        if auto_ids and not vk.endswith('-auto.log'):
            divergence = next((i for i, (a, b) in enumerate(zip(auto_ids, got)) if a != b), None)
            match = 'identical' if auto_ids == got else f'first divergence {divergence}, lengths {len(auto_ids)} vs {len(got)}'
            print(f"    vs auto: {match}")
    if not glob.glob(os.path.join(out, f"{name}-cuda-*.log")):
        print(f"{name}: INVALID (no CUDA logs)")
        invalid = True
raise SystemExit(1 if invalid else 0)
