import collections
import io
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
(root / "report").mkdir(parents=True, exist_ok=True)
base = root.parent
work = [json.loads(line) for line in (base / 'formal_workload.jsonl').open()]
assert len(work) == 100
answer_pattern = re.compile(r'ANSWER\s*:\s*([A-D])', re.I)
evidence = {}
answers = {}
timing = {}
for variant in 'AB':
    label = f'pd_pcp_{variant}1_c32'
    details, = (base / 'aisbench' / label).glob('*/performances/*/gpqa_diamond_details.jsonl')
    rows = sorted((json.loads(line) for line in details.open()), key=lambda row: row['id'])
    assert len(rows) == len(work)
    assert all(row['id'] == item['id'] and row['input'] == item['messages'] and row['success'] and not row['error_info']
               for row, item in zip(rows, work, strict=True))
    log = (base / 'runs' / label / 'toolbox.log').read_text()
    seconds, = re.findall(r'Api infer task time elapsed: ([\d.]+)s', log)
    assert re.search(r'POST=100 .*RECV=100 .*FAIL=0 .*FIN(?:ISH)?=100', log)
    seconds = float(seconds)
    lengths = [row['output_tokens'] for row in rows]
    db, = (details.parent / 'db_data').glob('*.db')
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
        arrays = {idx: np.load(io.BytesIO(blob), allow_pickle=False)
                  for idx, blob in conn.execute('select id, arr_blob from numpy_store')}
    refs = collections.Counter(row['time_points']['__db_ref__'] for row in rows if row['time_points'])
    missing, ambiguous = [], []
    variant_timing = {}
    variant_answers = {}
    for row, item in zip(rows, work, strict=True):
        qid, n = row['id'], row['output_tokens']
        pred = row['prediction']
        cut_before_final = n >= 20480 and '</think>' not in pred
        match = answer_pattern.search(pred.rsplit('</think>', 1)[-1]) if not cut_before_final else None
        guess = match[1].upper() if match else None
        variant_answers[qid] = {'id': qid, 'prediction': guess, 'answer': item['answer'],
                                'correct': guess == item['answer'], 'output_tokens': n,
                                'cap_without_final_separator': cut_before_final}
        if not row['time_points'] or row['time_points']['__db_ref__'] not in arrays:
            missing.append(qid)
            continue
        idx = row['time_points']['__db_ref__']
        if refs[idx] != 1:
            ambiguous.append(qid)
            continue
        t = arrays[idx]
        assert len(t) >= 2 and np.isfinite(t).all() and (np.diff(t) >= 0).all()
        variant_timing[qid] = {'ttft_ms': float((t[1] - t[0]) * 1000),
                               'tpot_ms': float((t[-1] - t[1]) * 1000 / (n - 1))}
    timing[variant] = variant_timing
    answers[variant] = variant_answers
    evidence[variant] = dict(
        requests=len(rows), output_tokens=sum(lengths), seconds=seconds,
        output_tok_s=sum(lengths)/seconds, req_s=len(rows)/seconds,
        correct=sum(row['correct'] for row in variant_answers.values()),
        unparsed=sum(row['prediction'] is None for row in variant_answers.values()),
        output_p50=float(np.percentile(lengths,50)), output_p90=float(np.percentile(lengths,90)),
        cap_count=sum(n >= 20480 for n in lengths), missing_timing=missing,
        duplicate_timing=ambiguous, timing_count=len(variant_timing), raw_details=str(details),
        original_perf=json.loads(details.with_name('gpqa_diamond.json').read_text()))
shared = sorted(set(timing['A']) & set(timing['B']))
matched = {variant: {key: statistics.mean(timing[variant][qid][key] for qid in shared)
                     for key in ('ttft_ms', 'tpot_ms')} for variant in 'AB'}
result = {'runs':evidence,'matched_timing_n':len(shared),'matched_timing_ids':shared,
          'matched_timing':matched,'answers':answers}
(root/'report/ab_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
for variant, row in evidence.items():
    print(variant,{k:val for k,val in row.items() if k not in ('original_perf','raw_details')})
print('Matched timings',len(shared),matched)
