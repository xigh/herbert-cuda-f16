# _pp.py OUTDIR — prefill tok/s against the prompt length (pp1, pp2): herbert-cuda and
# herbert-vk (median of the rounds, their [turn] line), llama.cpp CUDA and Vulkan (llama-bench avg_ts per n_prompt),
# vLLM (its latency bench: prompt / L(out 1)).
import sys, glob, os, re, json, statistics
out = sys.argv[1]
# Label -> exact tokenizer-visible length. The 512/4000 fixtures include one
# additional template token, so vLLM must be benchmarked at 513/4001 too.
pps = [(128, 128), (256, 256), (512, 513), (1024, 1024), (2048, 2048), (4000, 4001)]
rows = {}
# r<round>-<model>-<engine>-pp<n>.log, engine "herbert" (herbert-vk, pp1), or
# "herbert-cuda" / "herbert-vk" (pp2).
names = {"herbert": "herbert-vk (bf16)", "herbert-vk": "herbert-vk (bf16)", "herbert-cuda": "herbert-cuda"}
for f in glob.glob(os.path.join(out, "r*-*-herbert*-pp*.log")):
    base = os.path.basename(f)
    m = re.match(r"r\d+-([\d.]+B)-herbert-cuda(?:-([\w-]+))?-pp(\d+)\.log", base)
    if m:
        model, variant = m.group(1), m.group(2)
        engine = "herbert-cuda" + (f"-{variant}" if variant else "")
    else:
        m = re.match(r"r\d+-([\d.]+B)-(herbert(?:-\w+)?)-pp(\d+)\.log", base)
        if not m:
            continue
        model, variant = m.group(1), m.group(2)
        engine = names[variant]
    for l in open(f, errors="replace"):
        if l.startswith("[turn"):
            p = re.search(r"(\d+) prompt tokens in [\d.]+ s \(([\d.]+) tok/s\)", l)
            rows.setdefault((model, engine), {}).setdefault(int(p.group(1)), []).append(float(p.group(2)))
for m in ["0.8B", "4B"]:
    for eng, tag in [("llama.cpp CUDA", "cuda"), ("llama.cpp Vulkan", "vk")]:
        if not os.path.exists(os.path.join(out, f"{m}-llama-{tag}.json")):
            continue
        try:
            for e in json.load(open(os.path.join(out, f"{m}-llama-{tag}.json"))):
                rows.setdefault((m, eng), {}).setdefault(e["n_prompt"], []).append(e["avg_ts"])
        except Exception as e:
            print("llama", m, tag, e)
    for label, prompt_tokens in pps:
        try:
            data = json.load(open(os.path.join(out, f"vllm-{m}-pp{label}.json")))
            latencies = data.get("latencies", [data["avg_latency"]])
            rows.setdefault((m, "vLLM (own bench)"), {}).setdefault(prompt_tokens, []).extend(prompt_tokens / x for x in latencies)
        except Exception as e:
            print("vllm", m, label, e)
print(f"{'model':5} {'engine':20} " + " ".join(f"{n:>9}" for _, n in pps))
for (m, eng), d in sorted(rows.items()):
    print(f"{m:5} {eng:20} " + " ".join(f"{statistics.median(d[n]):9.1f}" if n in d else f"{'-':>9}" for _, n in pps))
print("\nCUDA / vLLM median ratio (greater than 1.000 means CUDA wins):")
for m in ["0.8B", "4B"]:
    vllm = rows.get((m, "vLLM (own bench)"), {})
    for (model, eng), d in sorted(rows.items()):
        if model != m or not eng.startswith("herbert-cuda"):
            continue
        ratios = [statistics.median(d[n]) / statistics.median(vllm[n]) if n in d and n in vllm else None for _, n in pps]
        verdict = "MARGIN" if all(x is not None and x >= 1.05 for x in ratios) else "CHECK"
        print(f"{m:5} {eng:20} {verdict:8} " + " ".join(f"{x:9.3f}" if x is not None else f"{'-':>9}" for x in ratios))
