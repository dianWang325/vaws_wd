import json
import os
from pathlib import Path
import sys

import requests

root = Path(os.environ['PD_OUTPUT_DIR']).resolve()
base = Path(os.environ['EXPERIMENT_ROOT']).resolve().parent
sys.path.insert(0, str(base))
from dsv4flash_pd_toolbox import run

label = sys.argv[1]
session = requests.Session()
session.trust_env = False
for phase in ('warmup','c32'):
    tag = f'pd_pcp_{label}_{phase}'
    for side,port in [('P',18766),('D',18767)]:
        response = session.get(f'http://127.0.0.1:{port}/metrics', timeout=10)
        response.raise_for_status()
        (root/f'{tag}.{side}.before.prom').write_text(response.text)
    dataset = 'warm16.csv' if phase == 'warmup' else 'formal.csv'
    metadata = run(tag,32,base/dataset,service_instance={
        'topology':{'P':'TP4/PCP2','D':'TP4/PCP1, EP disabled'},
        'logs':{'P':str(root/'P.log'),'D':str(root/'D.log'),'proxy':str(root/'proxy.log')},
        'processes':json.loads((root/'processes.json').read_text()),
        'adaptive_verification':True,
    },npu_num=12)
    performance, = Path(metadata['output_dir']).glob('*/performances/*/gpqa_diamond.json')
    values = json.loads(performance.read_text())
    expected = 16 if phase == 'warmup' else 100
    with performance.with_name('gpqa_diamond_details.jsonl').open() as handle:
        rows = [json.loads(line) for line in handle]
    if (len(rows) != expected or {row['id'] for row in rows} != set(range(expected))
            or any(not row['success'] or row['error_info'] for row in rows)):
        raise RuntimeError(f'Incomplete or failed responses in {performance}')
    metadata['successful_responses'] = len(rows)
    metadata['aisbench_timing_summary_requests'] = values['Success Requests']['total']

    for side,port in [('P',18766),('D',18767)]:
        response = session.get(f'http://127.0.0.1:{port}/metrics', timeout=10)
        response.raise_for_status()
        (root/f'{tag}.{side}.after.prom').write_text(response.text)
    metadata.update(performance=values,performance_path=str(performance),variant=label[0],instance=label)
    (root/f'{tag}.validated.json').write_text(json.dumps(metadata,indent=2))
