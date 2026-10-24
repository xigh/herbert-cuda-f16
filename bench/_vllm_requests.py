"""Same token IDs and greedy fixed output lengths as herbert-cuda-bench."""
import argparse
import json
import time
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--outputs', default='1')
    p.add_argument('--ctx', type=int, default=4096)
    a = p.parse_args()
    cases = json.loads(Path(a.cases).read_text())
    outputs = [int(x) for x in a.outputs.split(',')]
    if not cases or a.rounds < 1 or min(outputs) < 1:
        raise ValueError('nonempty cases and positive rounds/outputs required')
    if any(len(c['ids']) + max(outputs) > a.ctx for c in cases):
        raise ValueError('request exceeds context')
    from vllm import LLM, SamplingParams, __version__
    started = time.perf_counter()
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=a.ctx,
              max_num_seqs=1, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, seed=0)
    load_seconds = time.perf_counter() - started
    samples, first_calls = [], []
    params = {n: SamplingParams(temperature=0, max_tokens=n, ignore_eos=True,
                               detokenize=False) for n in outputs}
    for round_ in range(a.warmup + a.rounds):
        for ci in range(len(cases)):
            c = cases[(ci + round_) % len(cases)]
            for n in outputs:
                started = time.perf_counter()
                result = llm.generate([{'prompt_token_ids': c['ids']}], params[n], use_tqdm=False)
                ids = list(result[0].outputs[0].token_ids)
                latency = time.perf_counter() - started
                if len(ids) != n:
                    raise RuntimeError(f"{c['name']}: generated {len(ids)} != {n}")
                row = dict(case=c['name'], prompt_tokens=len(c['ids']), output_tokens=n,
                           round=max(0, round_ - a.warmup), latency_seconds=latency,
                           first_token_seconds=latency if n == 1 else None, output_ids=ids)
                if round_ >= a.warmup:
                    samples.append(row)
                elif round_ == 0:
                    first_calls.append(row)
    Path(a.out).write_text(json.dumps(dict(schema=1, backend='vllm', version=__version__,
        model=a.model, timing='token_ids_to_output_ids_greedy_ignore_eos', request_initialization_included=True, context=a.ctx,
        rounds=a.rounds, warmup=a.warmup, load_seconds=load_seconds,
        first_calls=first_calls, samples=samples), indent=2))

if __name__ == '__main__':
    main()
