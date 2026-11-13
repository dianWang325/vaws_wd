import concurrent.futures
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import requests

experiment = Path(os.environ['EXPERIMENT_ROOT']).resolve()
root = Path(os.environ.get('PD_OUTPUT_DIR', experiment)).resolve()
variant = os.environ.get('PD_VARIANT')
scripts = Path(__file__).resolve().parent
children = {}
session = requests.Session()
session.trust_env = False

def write(name, value):
    (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

def status(stage, **details):
    write('status.json', {'stage': stage, 'time': time.time(), **details})
    print(stage, details, flush=True)

def launch(name, command, **settings):
    env = dict(os.environ, **settings)
    env['SUFFIX_AB_RUN_ID'] = f'pd_pcp_{name}'
    for key, folder in [('PROMETHEUS_MULTIPROC_DIR','metrics'),('TRITON_CACHE_DIR','triton-cache')]:
        path = root / folder / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    with (root / f'{name}.log').open('wb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children[name] = child
    write('processes.json', {key: value.pid for key, value in children.items()})

def ready(name, port, endpoint='health'):
    status('waiting_ready', role=name)
    for _ in range(600):
        failed = [(key, process.returncode) for key, process in children.items() if process.poll() is not None]
        if failed:
            raise RuntimeError(f'Process exited: {failed}')
        try:
            response = session.get(f'http://127.0.0.1:{port}/{endpoint}', timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f'{name} health timeout')

def smoke(row):
    with requests.Session() as client:
        client.trust_env = False
        response = client.post('http://127.0.0.1:18765/v1/chat/completions',
            json={'model':'suffix-ab','messages':row['messages'],'max_tokens':160,'temperature':1.0,
                  'top_p':1.0,'seed':1024,'chat_template_kwargs':{'thinking':True,'reasoning_effort':'high'}},
            timeout=(10,900))
        return {'status':response.status_code,'response':response.json()}

def stop(*_):
    raise SystemExit(130)

signal.signal(signal.SIGTERM, stop)
try:
    for port in (18765,18766,18767):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('127.0.0.1',port))
    launch('P',['bash',str(scripts/'dsv4flash_pd_role.sh')],PD_ROLE='P',ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    ready('P',18766)
    extra = json.dumps({'enable_cpu_binding':True,'multistream_overlap_shared_expert':True,
        'runtime_config_path':str(root/'runtime_D.json'),
        'runtime_config':{'ascend_log':{'modules':{'vllm.logger':'DEBUG'}}}})
    settings = {} if variant else {'SUFFIX_ADDITIONAL_CONFIG': extra}
    launch('D',['bash',str(scripts/'dsv4flash_pd_role.sh')],PD_ROLE='D',
           ASCEND_RT_VISIBLE_DEVICES='8,9,10,11', **settings)
    ready('D',18767)
    launch('proxy',['python',str(scripts/'load_balance_proxy_server_example.py'),'--host','127.0.0.1','--port','18765',
        '--prefiller-hosts','127.0.0.1','--prefiller-ports','18766',
        '--decoder-hosts','127.0.0.1','--decoder-ports','18767','--workers','1'])
    ready('proxy',18765,'healthcheck')
    if variant:
        status('benchmarking',variant=variant)
        subprocess.run(['python',str(scripts/'dsv4flash_pd_benchmark.py'),variant+'1'],check=True)
        status('benchmark_complete',variant=variant)
        raise SystemExit(0)
    status('smoke')
    for side, port in [('P',18766),('D',18767)]:
        (root/f'{side}.before.prom').write_text(session.get(f'http://127.0.0.1:{port}/metrics',timeout=10).text)
    rows=[json.loads(line) for line in (root.parent/'formal_workload.jsonl').read_text().splitlines()[:4]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results=list(executor.map(smoke,rows))
    write('smoke.json',results)
    for side, port in [('P',18766),('D',18767)]:
        (root/f'{side}.after.prom').write_text(session.get(f'http://127.0.0.1:{port}/metrics',timeout=10).text)
    assert all(row['status']==200 and row['response']['usage']['completion_tokens']>0 for row in results),results
    status('smoke_complete',requests=len(results))
    while all(child.poll() is None for child in children.values()):
        time.sleep(5)
    raise RuntimeError('PD service exited')
except Exception as exc:
    status('failed',error=repr(exc))
    raise
finally:
    for child in reversed(list(children.values())):
        try:
            os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    for child in reversed(list(children.values())):
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
