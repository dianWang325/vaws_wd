import pathlib,sys,json,hashlib,traceback,random
import numpy as np
import torch,torch_npu,triton
from triton.runtime import driver
R=pathlib.Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P))
from kernel_A import _suffix_local_scan_kernel as A
from kernel_B import _suffix_local_scan_kernel_early_exit as B
torch.npu.set_device(0)
N=96;NB=5;STRIDE=1025;BLOCK=256;DEPTH=24;RADIX=1026
SEEDS=[11,23,37,53,71];COUNTS=[0,12,24,48,72,84,96]
CASES=[]
for count in COUNTS:
    layouts=[("control",None)] if count in [0,N] else [("interleaved",None),("contiguous",None)]+[("random",s) for s in SEEDS]
    for layout,seed in layouts:
        cid=f"long{count}_{layout}"+(str(seed) if seed is not None else "")
        if layout=="interleaved": selected=[(k*N)//count for k in range(count)]
        elif layout=="random": selected=sorted(random.Random(seed).sample(range(N),count))
        else:selected=list(range(count))
        assert len(set(selected))==count
        CASES.append({"id":cid,"long_requests":count,"long_fraction":count/N,"layout":layout,"seed":seed,"long_batch_indices":selected})
assert len(CASES)==37
def make(c):
    depths=np.full(N,4,dtype=np.int32);depths[c["long_batch_indices"]]=24
    tokens=np.full((N,STRIDE),2,dtype=np.int32)
    for r,d in enumerate(depths):
        for blk in range(4):
            end=blk*BLOCK+64;tokens[r,end-d:end]=1
        tokens[r,-(d+1):]=1
    # Identity request mapping keeps long_batch_indices explicit.
    return [tokens,np.arange(N,dtype=np.int32),np.full(N,STRIDE,dtype=np.int32),np.ones(N,dtype=np.int32)],depths
def reference(vals):
    tokens,mapping,lengths,sampled=vals
    expected=np.zeros((N,NB),dtype=np.int64);survival=np.zeros((N,NB),dtype=np.int32)
    for batch,req in enumerate(mapping):
        row=tokens[req];n=int(lengths[req])
        for blk in range(NB):
            for pos in range(max(1,blk*BLOCK),min(n,(blk+1)*BLOCK)):
                m=0
                for t in range(min(pos,n,DEPTH)):
                    if row[pos-1-t]!=row[n-1-t]:break
                    m+=1
                survival[batch,blk]=max(int(survival[batch,blk]),m)
                if m:expected[batch,blk]=max(int(expected[batch,blk]),m*RADIX+pos)
    return expected,survival
def gpu(vals):return [torch.from_numpy(v.copy()).to("npu:0") for v in vals]
def launch(fn,args,out):return fn[(N,NB)](args[0],STRIDE,args[1],args[2],args[3],out,NB,RADIX,MAX_DEPTH=DEPTH,BLOCK=BLOCK,num_warps=4)
if sys.argv[1]=="check":
    results=[]
    for c in CASES:
        try:
            vals,depths=make(c);expected,survival=reference(vals)
            assert np.array_equal(survival,np.repeat(depths[:,None],NB,axis=1))
            iterations=np.minimum(survival+1,DEPTH)
            np.savez(R/(c["id"]+".inputs_reference.npz"),tokens=vals[0],mapping=vals[1],lengths=vals[2],sampled=vals[3],expected=expected,max_match_per_program=survival,B_expected_iterations=iterations,request_depths=depths)
            args=gpu(vals)
            for label,fn in [("A",A),("B",B)]:
                out=torch.full((N,NB),-1,dtype=torch.int64,device="npu:0")
                k=launch(fn,args,out);torch.npu.synchronize();actual=out.cpu().numpy()
                np.save(R/(c["id"]+"."+label+".scratch.npy"),actual)
                assert np.array_equal(expected,actual)
                if c["id"]=="long0_control":
                    (R/(label+".metadata.txt")).write_text(repr(k.metadata))
                    for ext,asm in k.asm.items():
                        if isinstance(asm,str):(R/(label+"."+ext+".txt")).write_text(asm)
                results.append({"case":c["id"],"variant":label,"compile":"pass","execution":"pass","exact_match":True,"compared_elements":int(actual.size)})
                print(json.dumps(results[-1]),flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            (R/(c["id"]+".error.log")).write_text(error);results.append({"case":c["id"],"error":error})
        (R/"correctness.json").write_text(json.dumps(results,indent=2))
    info={"cases":CASES,"source_commit":"c8ba92bf77e499a1b3ed5da7b9d881757f871fd9","kernel_file_sha256":{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ["kernel_A.py","kernel_B.py"]},"requests":N,"grid":[N,NB],"logical_programs":N*NB,"target":repr(driver.active.get_current_target()),"device_properties":repr(torch.npu.get_device_properties(0)),"MAX_DEPTH":DEPTH,"BLOCK":BLOCK,"stride":STRIDE,"Lp1":RADIX,"num_warps":4,"short_match_depth":4,"short_B_iterations":5,"long_match_depth":24,"long_B_iterations":24,"warmup":51,"rounds":6,"launches_per_round_per_variant":50}
    (R/"manifest.json").write_text(json.dumps(info,indent=2))
    assert len(results)==74 and all(x.get("exact_match") for x in results)
elif sys.argv[1]=="profile":
    checked=json.loads((R/"correctness.json").read_text())
    assert len(checked)==74 and all(x.get("exact_match") for x in checked)
    args={c["id"]:gpu(make(c)[0]) for c in CASES}
    outputs={c["id"]:{label:torch.empty((N,NB),dtype=torch.int64,device="npu:0") for label in ["A","B"]} for c in CASES}
    fns={"A":A,"B":B};plan=[]
    def block(c,label,n,phase,rnd):
        plan.append({"case":c["id"],"variant":label,"launches":n,"phase":phase,"round":rnd})
        for _ in range(n):launch(fns[label],args[c["id"]],outputs[c["id"]][label])
        torch.npu.synchronize()
    for c in CASES:
        for label in ["A","B"]:block(c,label,51,"warmup",-1)
    for rnd in range(6):
        cases=CASES.copy();random.Random(2026111200+rnd).shuffle(cases)
        for c in cases:
            order=["A","B"] if rnd%2==0 else ["B","A"]
            for label in order:block(c,label,50,"measurement",rnd)
        print("ROUND COMPLETE",rnd,flush=True)
    for c in CASES:
        expected=np.load(R/(c["id"]+".inputs_reference.npz"))["expected"]
        for label in ["A","B"]:assert np.array_equal(expected,outputs[c["id"]][label].cpu().numpy())
    (R/"launch_plan.json").write_text(json.dumps({"blocks":plan,"expected_tasks":25974,"final_outputs_exact":True},indent=2))
    print("PROFILE COMPLETE 25974 tasks; final outputs exact",flush=True)
