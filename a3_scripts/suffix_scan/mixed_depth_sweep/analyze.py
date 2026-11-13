import pathlib,csv,json,statistics,collections
R=pathlib.Path(__file__).resolve().parent
paths=list((R/"msprof").glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"));assert len(paths)==1,paths
names={"_suffix_local_scan_kernel":"A","_suffix_local_scan_kernel_early_exit":"B"}
with paths[0].open() as f:rows=[r for r in csv.DictReader(f) if r["Op Name"] in names]
rows.sort(key=lambda r:float(r["Task Start Time(us)"].strip()))
plan=json.loads((R/"launch_plan.json").read_text())
assert len(rows)==plan["expected_tasks"]==25974
offset=0;data=[];counts=collections.Counter()
for block in plan["blocks"]:
    seg=rows[offset:offset+block["launches"]];offset+=block["launches"]
    assert len(seg)==block["launches"]
    for r in seg:
        assert names[r["Op Name"]]==block["variant"]
        assert r["Device_id"]=="0" and int(r["Block Num"])==48
        counts[(block["case"],block["variant"],block["phase"])]+=1
        if block["phase"]=="measurement":
            data.append({"case":block["case"],"variant":block["variant"],"round":block["round"],"task_id":r["Task ID"],"duration_us":float(r["Task Duration(us)"]),"aiv_time_us":float(r["aiv_time(us)"]),"block_num":int(r["Block Num"])})
assert offset==len(rows)
manifest=json.loads((R/"manifest.json").read_text())
results=[]
def stats(v):return {"n":len(v),"mean":statistics.mean(v),"median":statistics.median(v),"stdev":statistics.stdev(v),"min":min(v),"max":max(v)}
for c in manifest["cases"]:
    cid=c["id"];overall={};rounds=[]
    for label in ["A","B"]:
        assert counts[(cid,label,"warmup")]==51 and counts[(cid,label,"measurement")]==300
        overall[label]=stats([x["duration_us"] for x in data if x["case"]==cid and x["variant"]==label])
    for rnd in range(6):
        av={}
        for label in ["A","B"]:
            v=[x["duration_us"] for x in data if x["case"]==cid and x["variant"]==label and x["round"]==rnd]
            assert len(v)==50
            av[label]=statistics.median(v)
        rounds.append({"round":rnd,"A_us":av["A"],"B_us":av["B"],"speedup":av["A"]/av["B"],"B_time_reduction_percent":100*(1-av["B"]/av["A"])})
    av=statistics.median([x["A_us"] for x in rounds]);bv=statistics.median([x["B_us"] for x in rounds])
    results.append({**c,"A_us":av,"B_us":bv,"speedup_A_over_B":av/bv,"B_time_reduction_percent":100*(1-bv/av),"overall_us":overall,"rounds":rounds,"round_speedup_range":[min(x["speedup"] for x in rounds),max(x["speedup"] for x in rounds)]})
b0=next(x["B_us"] for x in results if x["long_requests"]==0)
b1=next(x["B_us"] for x in results if x["long_requests"]==96)
for x in results:
    prediction=(1-x["long_fraction"])*b0+x["long_fraction"]*b1
    x["balanced_endpoint_cost_model_B_us"]=prediction
    x["balanced_endpoint_cost_model_speedup"]=x["A_us"]/prediction
    x["B_excess_over_balanced_model_percent"]=100*(x["B_us"]/prediction-1)
groups=[]
for count in [0,12,24,48,72,84,96]:
    matches=[x for x in results if x["long_requests"]==count]
    grouped={"long_requests":count,"long_percent":100*count/96}
    for layout in ["control","interleaved","contiguous","random"]:
        v=[x for x in matches if x["layout"]==layout]
        if v: grouped[layout]={"layouts":len(v),"A_us_median":statistics.median([x["A_us"] for x in v]),"B_us_median":statistics.median([x["B_us"] for x in v]),"speedup_median":statistics.median([x["speedup_A_over_B"] for x in v]),"speedup_min":min(x["speedup_A_over_B"] for x in v),"speedup_max":max(x["speedup_A_over_B"] for x in v)}
    grouped["balanced_model_speedup_median"]=statistics.median([x["balanced_endpoint_cost_model_speedup"] for x in matches])
    groups.append(grouped)
out={"metric":"msprof Task Duration(us), device task only","aggregation":"median of six round medians, 50 samples per round per variant","raw_tasks":len(rows),"measured_tasks":len(data),"warmups":3774,"actual_block_num_all_tasks":48,"results":results,"grouped":groups,"source_csv":str(paths[0])}
(R/"performance.json").write_text(json.dumps(out,indent=2))
with (R/"measured_tasks.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print("count layout seed A_us B_us speedup B_time_reduction_percent")
for x in results:print(x["long_requests"],x["layout"],x["seed"],round(x["A_us"],3),round(x["B_us"],3),round(x["speedup_A_over_B"],4),round(x["B_time_reduction_percent"],3))
print("GROUPED",json.dumps(groups,indent=2))
