import ast, difflib, hashlib, json, pathlib, sys, traceback
import numpy as np
import torch, torch_npu, triton
from triton.runtime import driver
R=pathlib.Path(__file__).resolve().parent
P=R.parent
sys.path.insert(0,str(R))
source=(P/"speculator.original.py").read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="_suffix_local_scan_kernel")
a="".join(source.splitlines(keepends=True)[min(d.lineno for d in node.decorator_list)-1:node.end_lineno])
b=a.replace("def _suffix_local_scan_kernel(","def _suffix_local_scan_kernel_early_exit(")
assert b.count("    for t in range(MAX_DEPTH):")==1
b=b.replace("    for t in range(MAX_DEPTH):","    t = 0\n    while (t < MAX_DEPTH) & (tl.sum(active.to(tl.int32), axis=0) > 0):")
b=b.replace("        match_len += active.to(tl.int64)","        match_len += active.to(tl.int64)\n        t += 1")
for name,text in [("kernel_A.py",a),("kernel_B.py",b)]:
    (R/name).write_text("import triton\nimport triton.language as tl\n\n"+text)
(R/"A_to_B.diff").write_text("".join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile="A",tofile="B")))
assert a.count("load_mask = candidate_mask")==b.count("load_mask = candidate_mask")==1
torch.npu.set_device(0)
from kernel_A import _suffix_local_scan_kernel as A
from kernel_B import _suffix_local_scan_kernel_early_exit as B
STRIDE=1025; BLOCK=256; DEPTH=24; N=9; NB=5; RADIX=1026
def data(case):
    tokens=np.full((N,STRIDE),17,dtype=np.int32)
    lengths=np.full(N,STRIDE,dtype=np.int32)
    mapping=np.array([8,0,6,2,7,1,5,3,4],dtype=np.int32)
    sampled=np.ones(N,dtype=np.int32)
    if case=="all_mismatch":
        for r in range(N): tokens[r]=np.arange(STRIDE,dtype=np.int32)+r*2000
    if case=="unaligned":
        lengths=np.array([0,1,2,23,25,255,257,513,1025],dtype=np.int32)
        tokens.fill(-777)
        for r,n in enumerate(lengths): tokens[r,:n]=np.arange(n,dtype=np.int32)%7
        sampled[6]=0
    return tokens,mapping,lengths,sampled
def cpu(vals):
    tokens,mapping,lengths,sampled=vals
    out=np.zeros((N,NB),dtype=np.int64); survival=np.zeros((N,NB),dtype=np.int32)
    for row,req in enumerate(mapping):
        n=int(lengths[req])
        for blk in range(NB):
            for pos in range(max(1,blk*BLOCK),min(n,(blk+1)*BLOCK)):
                m=0
                if sampled[row]>0:
                    for t in range(min(DEPTH,pos,n)):
                        if tokens[req,pos-1-t]!=tokens[req,n-1-t]: break
                        m+=1
                survival[row,blk]=max(int(survival[row,blk]),m)
                if m: out[row,blk]=max(int(out[row,blk]),m*RADIX+pos)
    return out,survival
def to_device(vals):
    return [torch.from_numpy(v.copy()).to("npu:0") for v in vals]
def launch(fn,args,out):
    return fn[(N,NB)](args[0],STRIDE,args[1],args[2],args[3],out,NB,RADIX,MAX_DEPTH=DEPTH,BLOCK=BLOCK,num_warps=4)
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="check":
        results=[]
        for case in ["full_depth","all_mismatch","unaligned"]:
            vals=data(case); expected,survival=cpu(vals)
            if case=="full_depth": assert (survival==DEPTH).all()
            np.savez(R/(case+".inputs_reference.npz"),tokens=vals[0],mapping=vals[1],lengths=vals[2],sampled=vals[3],expected=expected,max_match_per_program=survival)
            args=to_device(vals)
            for label,fn in [("A",A),("B",B)]:
                try:
                    out=torch.full((N,NB),-1,dtype=torch.int64,device="npu:0")
                    k=launch(fn,args,out);torch.npu.synchronize()
                    actual=out.cpu().numpy()
                    np.save(R/(case+"."+label+".scratch.npy"),actual)
                    (R/(case+"."+label+".scratch.json")).write_text(json.dumps({"actual":actual.tolist(),"expected":expected.tolist()},indent=2))
                    assert np.array_equal(expected,actual)
                    (R/(label+".metadata.txt")).write_text(repr(k.metadata))
                    for ext,asm in k.asm.items():
                        if isinstance(asm,str): (R/(label+"."+ext+".txt")).write_text(asm)
                    result={"case":case,"variant":label,"compile":"pass","execution":"pass","exact_match":True,"elements":int(actual.size)}
                except Exception:
                    error=traceback.format_exc();print(error,flush=True)
                    (R/(case+"."+label+".error.log")).write_text(error)
                    result={"case":case,"variant":label,"status":"failed","error":error}
                results.append(result);print(json.dumps(result),flush=True)
                (R/"correctness.json").write_text(json.dumps(results,indent=2))
        manifest={"commit":"c8ba92bf77e499a1b3ed5da7b9d881757f871fd9","source_path":"vllm_ascend/ops/triton/spec_decode/suffix.py","A_sha256":hashlib.sha256(a.encode()).hexdigest(),"B_sha256":hashlib.sha256(b.encode()).hexdigest(),"target":repr(driver.active.get_current_target()),"B_change":"only for->while active-reduction condition plus induction variable; function renamed for profiler attribution","device":0,"MAX_DEPTH":DEPTH,"BLOCK":BLOCK,"grid":[N,NB],"stride":STRIDE,"Lp1":RADIX,"num_warps":4}
        (R/"manifest.json").write_text(json.dumps(manifest,indent=2))
        if not all(x.get("exact_match") for x in results): sys.exit(1)
    elif mode=="profile":
        results=json.loads((R/"correctness.json").read_text())
        assert len(results)==6 and all(x.get("exact_match") for x in results)
        vals=data("full_depth"); expected,survival=cpu(vals)
        assert (survival==DEPTH).all(), "every one of the 45 programs must survive to depth 24"
        args=to_device(vals)
        out={label:torch.empty((N,NB),dtype=torch.int64,device="npu:0") for label in ["A","B"]}
        fns={"A":A,"B":B}
        # 51 warmup launches per variant: initial cached compilation/launch + 50 warmups.
        for i in range(51):
            for label in ["A","B"]: launch(fns[label],args,out[label])
        torch.npu.synchronize()
        plan=[]
        for rnd in range(6):
            order=["A","B"] if rnd%2==0 else ["B","A"]
            for label in order:
                for i in range(200): launch(fns[label],args,out[label])
                torch.npu.synchronize()
                plan.append({"round":rnd,"variant":label,"launches":200})
        for label in ["A","B"]:
            assert np.array_equal(expected,out[label].cpu().numpy())
        (R/"launch_plan.json").write_text(json.dumps({"warmup_per_variant":51,"measured_per_variant":1200,"blocks":plan,"all_programs_depth_24":True,"performance_input":"full_depth.inputs_reference.npz","final_outputs_exact":True},indent=2))
        print("PROFILE COMPLETE",flush=True)
