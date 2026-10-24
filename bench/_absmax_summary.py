"""_absmax_summary.py FILE.json… — the f16 range of herbert-cuda-absmax outputs: per file, the weights (largest
magnitude, tensors with elements above 65 504, share below the f16 normal range) and the activations per buffer
(largest magnitude over layers and prompts with where it is, elements above 65 504, share below 2^-14 and 2^-24)."""
import sys, json, collections
for fn in sys.argv[1:]:
    d = json.load(open(fn)); print(f"== {fn}  ({d['tuning'][:80]}…)")
    w = d["weights"]
    tot = sum(x["count"] for x in w); bn = sum(x["below_normal"] for x in w); bs = sum(x["below_subnormal"] for x in w)
    top = max(w, key=lambda x: x["max"])
    print(f"weights: {len(w)} tensors, max {top['max']:.4g} ({top['name']}), over 65504: {sum(x['over_f16'] for x in w)}, "
          f"below 2^-14: {bn / tot * 100:.3f} %, below 2^-24: {bs / tot * 100:.4f} % of {tot} values")
    for x in sorted(w, key=lambda x: -x["below_normal"] / max(x["count"], 1))[:3]:
        print(f"   most below 2^-14: {x['name']} {x['below_normal'] / x['count'] * 100:.2f} % (max {x['max']:.3g}, {x['dtype']})")
    by = collections.defaultdict(list)
    for a in d["activations"]: by[a["buffer"]].append(a)
    print(f"{'buffer':<14} {'max':>10} {'layer':>5} {'prompt':<11} {'kernel':<34} {'>65504':>7} {'<2^-14 %':>9} {'<2^-24 %':>9}")
    for b, rows in sorted(by.items(), key=lambda kv: -max(r["max"] for r in kv[1])):
        r = max(rows, key=lambda r: r["max"]); n = sum(x["count"] for x in rows)
        print(f"{b:<14} {r['max']:>10.4g} {str(r['layer']):>5} {r['prompt']:<11} {r['max_kernel'][:34]:<34} {sum(x['over_f16'] for x in rows):>7} "
              f"{sum(x['below_normal'] for x in rows) / n * 100:>9.3f} {sum(x['below_subnormal'] for x in rows) / n * 100:>9.4f}")
    nf = sum(a["non_finite"] for a in d["activations"])
    if nf: print("non-finite values:", nf)
