# _dec.py OUTDIR — the decode campaign of herbert-cuda against herbert-vk and vLLM:
# per prompt and model, the medians of every speed.sh variant (OUTDIR/<prompt>-<model>/),
# the distinct [ids] fingerprints across rounds, and vLLM from its latency bench
# (OUTDIR/vllm-<model>-in<len>-out{1,255}.json): prefill = len / l1, decode = 254 / (l255 − l1).
import sys, glob, os, re, json, statistics, hashlib
out = sys.argv[1]
lens = {"p3-long": 759, "p4-xlong": 2880}
for p, n in lens.items():
    for m in ["0.8B", "4B"]:
        d = os.path.join(out, f"{p}-{m}")
        if not os.path.isdir(d):
            continue
        rows = {}
        for f in sorted(glob.glob(os.path.join(d, "r*-*.log"))):
            eng = re.sub(r"^r\d+-cuda-\d+-", "", os.path.basename(f))[:-4]
            ids = None
            for l in open(f, errors="replace"):
                if l.startswith("[ids]"):
                    ids = hashlib.md5(l.encode()).hexdigest()[:8]
                if l.startswith("[turn"):
                    pp = re.search(r"(\d+) prompt tokens in ([\d.]+) s \(([\d.]+) tok/s\)", l)
                    g = re.search(r"(\d+) generated in ([\d.]+) s \(([\d.]+) tok/s\)", l)
                    rows.setdefault(eng, []).append((float(pp.group(3)), float(g.group(3)), int(g.group(1)), ids))
        print(f"=== {p} ({n} tokens) · {m}")
        for eng, v in rows.items():
            pre = [x[0] for x in v]; dec = [x[1] for x in v]
            print(f"  {eng:10} N={len(v)}  prefill {statistics.median(pre):9.1f} (min {min(pre):.1f} max {max(pre):.1f})"
                  f"  decode {statistics.median(dec):7.2f} (min {min(dec):.2f} max {max(dec):.2f})"
                  f"  generated {sorted(set(x[2] for x in v))}  ids {sorted(set(x[3] for x in v))}")
        try:
            l1 = json.load(open(os.path.join(out, f"vllm-{m}-in{n}-out1.json")))["avg_latency"]
            l255 = json.load(open(os.path.join(out, f"vllm-{m}-in{n}-out255.json")))["avg_latency"]
            print(f"  {'vllm':10} own bench, 5 iters  prefill {n / l1:9.1f}  decode {254 / (l255 - l1):7.2f}")
        except Exception as e:
            print("  vllm", e)
