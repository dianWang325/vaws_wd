import csv,json,pathlib,statistics
R=pathlib.Path(__file__).resolve().parent
files=list((R/"msprof").glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
assert len(files)==1,files
rows=list(csv.DictReader(files[0].open()))
names={"_suffix_local_scan_kernel":"A","_suffix_local_scan_kernel_early_exit":"B"}
selected=[r for r in rows if r["Op Name"] in names]
selected.sort(key=lambda r:float(r["Task Start Time(us)"].strip()))
plan=json.loads((R/"launch_plan.json").read_text())
seen={"A":0,"B":0}; measured=[]
for r in selected:
    label=names[r["Op Name"]]; seen[label]+=1
    if seen[label]>plan["warmup_per_variant"]:
        measured.append({"variant":label,"task_id":r["Task ID"],"start_us":r["Task Start Time(us)"].strip(),"duration_us":float(r["Task Duration(us)"]),"aiv_time_us":float(r["aiv_time(us)"]),"block_num":int(r["Block Num"]),"device":r["Device_id"]})
assert seen=={"A":1251,"B":1251},seen
assert len(measured)==2400
offset=0; rounds=[]
def stats(v):
    return {"n":len(v),"mean":statistics.mean(v),"median":statistics.median(v),"stdev":statistics.stdev(v),"min":min(v),"max":max(v),"p10":float(sorted(v)[int(.1*(len(v)-1))]),"p90":float(sorted(v)[int(.9*(len(v)-1))])}
for block in plan["blocks"]:
    seg=measured[offset:offset+block["launches"]];offset+=block["launches"]
    assert all(r["variant"]==block["variant"] and r["block_num"]==45 and r["device"]=="0" for r in seg)
    for r in seg:r["round"]=block["round"]
    rounds.append({**block,"task_duration_us":stats([r["duration_us"] for r in seg]),"aiv_time_us":stats([r["aiv_time_us"] for r in seg])})
paired=[]
for rnd in range(6):
    a=next(x for x in rounds if x["round"]==rnd and x["variant"]=="A")
    b=next(x for x in rounds if x["round"]==rnd and x["variant"]=="B")
    av=a["task_duration_us"]["median"];bv=b["task_duration_us"]["median"]
    paired.append({"round":rnd,"A_median_us":av,"B_median_us":bv,"delta_us":bv-av,"delta_percent":(bv/av-1)*100})
overall={label:stats([r["duration_us"] for r in measured if r["variant"]==label]) for label in ["A","B"]}
med={label:statistics.median([x["task_duration_us"]["median"] for x in rounds if x["variant"]==label]) for label in ["A","B"]}
result={"source_csv":str(files[0]),"metric":"msprof Task Duration(us), device task only; host launch gaps and Task Wait Time excluded","raw_task_counts":seen,"warmup_removed_per_variant":51,"overall_us":overall,"rounds":rounds,"paired_round_medians":paired,"median_of_round_medians_us":med,"median_round_delta_percent":(med["B"]/med["A"]-1)*100,"median_of_paired_delta_percent":statistics.median([x["delta_percent"] for x in paired])}
(R/"performance.json").write_text(json.dumps(result,indent=2))
with (R/"measured_tasks.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(measured[0]));w.writeheader();w.writerows(measured)
print(json.dumps({k:v for k,v in result.items() if k!="rounds"},indent=2))
