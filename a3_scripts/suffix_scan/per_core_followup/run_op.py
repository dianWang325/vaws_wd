import sys,pathlib,json,time
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P/"mixed_depth_sweep"))
import mixed as M
caseid=sys.argv[1];label=sys.argv[2]
c=next(c for c in M.CASES if c["id"]==caseid)
vals,depths=M.make(c);args=M.gpu(vals)
out=torch.empty((M.N,M.NB),dtype=torch.int64,device="npu:0")
fn=M.A if label=="A" else M.B
for i in range(5):M.launch(fn,args,out)
torch.npu.synchronize()
for i in range(5):M.launch(fn,args,out)
torch.npu.synchronize()
expected=np.load(P/"mixed_depth_sweep"/(caseid+".inputs_reference.npz"))["expected"]
assert np.array_equal(expected,out.cpu().numpy())
(R/(caseid+"_"+label+".run.json")).write_text(json.dumps({"case":caseid,"variant":label,"warmup":5,"repetitions":5,"output_exact":True,"long_requests":c["long_requests"],"long_batch_indices":c["long_batch_indices"]},indent=2))
print("COMPLETE",caseid,label,flush=True)
