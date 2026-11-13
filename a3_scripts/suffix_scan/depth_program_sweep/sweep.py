import pathlib,sys,json,hashlib,traceback,random
import numpy as np
import torch,torch_npu,triton
from triton.runtime import driver
R=pathlib.Path(__file__).resolve().parent
P=R.parent
sys.path.insert(0,str(P))
from kernel_A import _suffix_local_scan_kernel as A
from kernel_B import _suffix_local_scan_kernel_early_exit as B
torch.npu.set_device(0)
STRIDE=1025; BLOCK=256; NB=5; MAX_DEPTH=24; Lp1=1026
DEPTHS=[4,8,12,16,20]; REQUESTS=[9,10,32,96]
CASES=[{"id":f"p{n*NB}_d{d}","requests":n,"programs":n*NB,"depth":d,"B_expected_iterations":d+1} for n in REQUESTS for d in DEPTHS]
def make(c):
    n=c["requests"];d=c["depth"]
    tokens=np.full((n,STRIDE),2,dtype=np.int32)
    # Every full block has an earlier run of exactly d ones.
    for blk in range(4):
        end=blk*BLOCK+64
        tokens[:,end-d:end]=1
    # The final block contains only end_pos=1024. A run of d+1 tail ones
    # makes that candidate match exactly d; the preceding 2 ends it.
    tokens[:,-(d+1):]=1
    mapping=np.arange(n-1,-1,-1,dtype=np.int32)
    lengths=np.full(n,STRIDE,dtype=np.int32)
    sampled=np.ones(n,dtype=np.int32)
    return tokens,mapping,lengths,sampled
def reference(vals):
    tokens,mapping,lengths,sampled=vals
    out=np.zeros((len(mapping),NB),dtype=np.int64)
    survival=np.zeros_like(out,dtype=np.int32)
    for b,req in enumerate(mapping):
        row=tokens[req];length=int(lengths[req])
        for blk in range(NB):
            for pos in range(max(1,blk*BLOCK),min(length,(blk+1)*BLOCK)):
                m=0
                for t in range(min(pos,length,MAX_DEPTH)):
                    if row[pos-1-t]!=row[length-1-t]: break
                    m+=1
                survival[b,blk]=max(int(survival[b,blk]),m)
                if m:out[b,blk]=max(int(out[b,blk]),m*Lp1+pos)
    return out,survival
def gpu(vals):
    return [torch.from_numpy(x.copy()).to("npu:0") for x in vals]
def launch(fn,c,args,out):
    return fn[(c["requests"],NB)](args[0],STRIDE,args[1],args[2],args[3],out,NB,Lp1,MAX_DEPTH=MAX_DEPTH,BLOCK=BLOCK,num_warps=4)
if sys.argv[1]=="check":
    result=[]
    for c in CASES:
        try:
            vals=make(c);expected,survival=reference(vals)
            assert (survival==c["depth"]).all(),(c,survival)
            np.savez(R/(c["id"]+".inputs_reference.npz"),tokens=vals[0],mapping=vals[1],lengths=vals[2],sampled=vals[3],expected=expected,max_match_per_program=survival,B_expected_iterations=survival+1)
            args=gpu(vals)
            for label,fn in [("A",A),("B",B)]:
                out=torch.full((c["requests"],NB),-1,dtype=torch.int64,device="npu:0")
                k=launch(fn,c,args,out);torch.npu.synchronize()
                actual=out.cpu().numpy()
                np.save(R/(c["id"]+"."+label+".scratch.npy"),actual)
                assert np.array_equal(actual,expected),(c,label,np.argwhere(actual!=expected).tolist())
                (R/(c["id"]+"."+label+".metadata.txt")).write_text(repr(k.metadata))
                result.append({**c,"variant":label,"compile":"pass","execution":"pass","exact_match":True,"compared_elements":int(actual.size)})
                print(json.dumps(result[-1]),flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            (R/(c["id"]+".error.log")).write_text(error)
            result.append({**c,"error":error})
        (R/"correctness.json").write_text(json.dumps(result,indent=2))
    info={"cases":CASES,"source_commit":"c8ba92bf77e499a1b3ed5da7b9d881757f871fd9","source_kernel_fingerprints":{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ["kernel_A.py","kernel_B.py"]},"target":repr(driver.active.get_current_target()),"device_properties":repr(torch.npu.get_device_properties(0)),"MAX_DEPTH":MAX_DEPTH,"BLOCK":BLOCK,"stride":STRIDE,"Lp1":Lp1,"num_warps":4,"warmup":51,"rounds":6,"measured_launches_per_variant_per_round":100}
    (R/"manifest.json").write_text(json.dumps(info,indent=2))
    assert len(result)==40 and all(x.get("exact_match") for x in result)
elif sys.argv[1]=="profile":
    checked=json.loads((R/"correctness.json").read_text())
    assert len(checked)==40 and all(x.get("exact_match") for x in checked)
    allargs={c["id"]:gpu(make(c)) for c in CASES}
    outputs={c["id"]:{label:torch.empty((c["requests"],NB),dtype=torch.int64,device="npu:0") for label in ["A","B"]} for c in CASES}
    fns={"A":A,"B":B};plan=[]
    def block(c,label,count,phase,rnd):
        plan.append({"case":c["id"],"variant":label,"launches":count,"phase":phase,"round":rnd,"programs":c["programs"],"depth":c["depth"]})
        for _ in range(count):launch(fns[label],c,allargs[c["id"]],outputs[c["id"]][label])
        torch.npu.synchronize()
    # Both kernels and all shapes have already passed compilation; profiling
    # still discards 51 warmup tasks for every (case,variant).
    for c in CASES:
        for label in ["A","B"]:block(c,label,51,"warmup",-1)
    for rnd in range(6):
        cases=CASES.copy();random.Random(20261112+rnd).shuffle(cases)
        for c in cases:
            order=["A","B"] if rnd%2==0 else ["B","A"]
            for label in order:block(c,label,100,"measurement",rnd)
    for c in CASES:
        expected=np.load(R/(c["id"]+".inputs_reference.npz"))["expected"]
        for label in ["A","B"]:assert np.array_equal(expected,outputs[c["id"]][label].cpu().numpy())
    (R/"launch_plan.json").write_text(json.dumps({"blocks":plan,"final_outputs_exact":True,"expected_tasks":26040},indent=2))
    print("PROFILE COMPLETE: 26040 tasks, final outputs exact",flush=True)
