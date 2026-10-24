# _vs.py OUTDIR [PROMPT_TOKENS] — medians of the vs.sh campaign: prefill of 759 tokens and decode
# of 255 tokens at depth 759, per engine and model, in tok/s.
# llama.cpp: avg_ts of llama-bench (pp: -p 759, tg: -n 255 -d 759).
# herbert-vk: the [turn] line (prefill of the 759-token prompt, decode of 255).
# vLLM: its latency bench, prefill = PROMPT_TOKENS (759) / L(out 1), decode = 254 / (L(out 255) − L(out 1)).
import sys, glob, os, re, json, statistics
out = sys.argv[1]
prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 759
rows = {}
def add(model, engine, pp, tg):
    rows.setdefault((model, engine), ([], []))
    if pp is not None: rows[(model, engine)][0].append(pp)
    if tg is not None: rows[(model, engine)][1].append(tg)
for f in glob.glob(os.path.join(out, "r*-*-pp.json")):
    m = re.match(r"r\d+-([\d.]+B)-(.*)-pp\.json", os.path.basename(f))
    try:
        pp = json.load(open(f))[0]["avg_ts"]
        t = f.replace("-pp.json", "-tg.json")
        tg = json.load(open(t))[0]["avg_ts"] if os.path.exists(t) else None
    except Exception as e:
        print("unreadable", f, e); continue
    add(m.group(1), m.group(2), pp, tg)
for f in glob.glob(os.path.join(out, "r*-vk-*.log")):
    m = re.match(r"r\d+-([\d.]+B)-(vk-.*)\.log", os.path.basename(f))
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            p = re.search(r"(\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l)
            g = re.search(r"(\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)", l)
            add(m.group(1), "herbert-" + m.group(2), float(p.group(2)), float(g.group(2)))
for model in ["0.8B", "4B"]:
    try:
        l1 = json.load(open(os.path.join(out, f"vllm-{model}-out1.json")))["avg_latency"]
        l255 = json.load(open(os.path.join(out, f"vllm-{model}-out255.json")))["avg_latency"]
        add(model, "vllm-cuda (own bench, N=5 not interleaved)", prompt / l1, 254 / (l255 - l1))
    except Exception as e:
        print("vllm", model, e)
for (model, engine), (pp, tg) in sorted(rows.items()):
    fmt = lambda v: f"{statistics.median(v):9.1f} (min {min(v):.1f} max {max(v):.1f})" if v else "        -"
    print(f"{model:5} {engine:44} N={max(len(pp), len(tg))}  prefill {fmt(pp)}  decode {fmt(tg)}")
