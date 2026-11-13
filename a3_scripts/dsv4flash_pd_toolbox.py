import pathlib,json,os,subprocess,time,shutil,hashlib,sys,csv
R=pathlib.Path(os.environ["EXPERIMENT_ROOT"]).resolve().parent
T=R/"toolbox"
W=R/"aisbench_runtime"
def run(label,c,dataset,service_instance=None,npu_num=8):
 folder=R/'runs'/label;folder.mkdir(parents=True,exist_ok=False)
 dataset=pathlib.Path(dataset).resolve();n=sum(1 for _ in csv.reader(dataset.open(newline="")))-1;output=R/'aisbench'/label
 config=(T/'config.py').read_text();config='\n'.join('OUTPUT_DIR='+repr(str(output)) if x.startswith('OUTPUT_DIR=') else x for x in config.splitlines())+'\n'
 (T/'config.py').write_text(config);(folder/'toolbox_config.py').write_text(config)
 cfg=W/'ais_bench/benchmark/configs/datasets/gpqa/gpqa_diamond_perf.py'
 dataset_dir=W/'ais_bench/datasets/gpqa'
 cfg_lines=cfg.read_text().splitlines()
 matches=[line for line in cfg_lines if line.strip().startswith('path=')]
 if len(matches)!=1:
  raise RuntimeError(f'Expected one GPQA dataset path in {cfg}')
 replacement='            path='+repr(str(dataset_dir))+','
 cfg.write_text('\n'.join(replacement if line==matches[0] else line for line in cfg_lines)+'\n')
 env=os.environ.copy();env['PYTHONPATH']=str(W)+':'+env.get('PYTHONPATH','');env['NO_PROXY']='127.0.0.1,localhost';env['no_proxy']=env['NO_PROXY'];env['OMP_NUM_THREADS']='1';env['TOKENIZERS_PARALLELISM']='false'
 for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:env.pop(k,None)
 origin=subprocess.check_output(['python','-c','import ais_bench;print(ais_bench.__file__)'],env=env,text=True).strip();assert str(W) in origin,origin
 cmd=['python',str(T/'aisbench_test.py'),'--dataset',str(dataset),'--data_num',str(n),'--output_len','20480','--concurrency',str(c),'--request_rate','0','--test_type','stream','--api_mode','chat','--npu_num',str(npu_num),'--repeat','1']
 meta={'label':label,'concurrency':c,'requests':n,'dataset':str(dataset),'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'command':cmd,'cwd':str(T),'started_at':time.time(),'output_dir':str(output),'aisbench_origin':origin,'environment':{k:env[k] for k in ['PYTHONPATH','NO_PROXY','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM']},'service_instance':service_instance if service_instance is not None else json.loads((R/'service_state.json').read_text())}
 (folder/'phase.json').write_text(json.dumps(meta,indent=2))
 with (folder/'toolbox.log').open('w') as f:p=subprocess.run(cmd,cwd=T,env=env,stdout=f,stderr=subprocess.STDOUT)
 meta.update(ended_at=time.time(),exit_code=p.returncode)
 for n in ['temp_api.py','aisbench.log','aisbench_all.log','aisbench_result.csv']:
  if (T/n).exists():shutil.copyfile(T/n,folder/n)
 (folder/'phase.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
 if p.returncode:raise RuntimeError('Toolbox phase failed: '+str(folder/'toolbox.log'))
 return meta
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]),sys.argv[3])
