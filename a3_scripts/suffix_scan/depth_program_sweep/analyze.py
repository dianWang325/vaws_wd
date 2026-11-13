import csv,json,pathlib,statistics,collections
R=pathlib.Path(__file__).resolve().parent
paths=list((R/"msprof").glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
assert len(paths)==1,paths
names={"_suffix_local_scan_kernel":"A","_suffix_local_scan_kernel_early_exit":"B"}
with paths[0].open() as f: rows=[r for r in csv.DictReader(f) if r["Op Name"] in names]
rows.sort(key=lambda r:float(r["Task Start Time(us)"].strip()))
plan=json.loads((R/"launch_plan.json").read_text())
assert len(rows)==plan["expected_tasks"]==26040,(len(rows),plan["expected_tasks"])
offset=0;data=[];counts=collections.Counter();blocks=collections.defaultdict(set)
for part in plan["blocks"]:
    seg=rows[offset:offset+part["launches"]];offset+=part["launches"]
    assert len(seg)==part["launches"]
    for r in seg:
        assert names[r["Op Name"]]==part["variant"],(part,r)
        assert r["Device_id"]=="0"
        blocks[part["case"]].add(int(r["Block Num"]))
        counts[(part["case"],part["variant"],part["phase"])]+=1
        if part["phase"]=="measurement":
            data.append({"case":part["case"],"programs":part["programs"],"depth":part["depth"],"variant":part["variant"],"round":part["round"],"task_id":r["Task ID"],"duration_us":float(r["Task Duration(us)"]),"aiv_time_us":float(r["aiv_time(us)"]),"actual_block_num":int(r["Block Num"])})
assert offset==len(rows)
manifest=json.loads((R/"manifest.json").read_text())
results=[]
def stats(v):
    return {"n":len(v),"mean":statistics.mean(v),"median":statistics.median(v),"stdev":statistics.stdev(v),"min":min(v),"max":max(v)}
for c in manifest["cases"]:
    cid=c["id"];summary={};rstats=[]
    for label in ["A","B"]:
        assert counts[(cid,label,"warmup")]==51
        assert counts[(cid,label,"measurement")]==600
        v=[x for x in data if x["case"]==cid and x["variant"]==label]
        summary[label]=stats([x["duration_us"] for x in v])
    for rnd in range(6):
        vals={label:statistics.median([x["duration_us"] for x in data if x["case"]==cid and x["variant"]==label and x["round"]==rnd]) for label in ["A","B"]}
        assert all(sum(x["case"]==cid and x["variant"]==label and x["round"]==rnd for x in data)==100 for label in ["A","B"])
        rstats.append({"round":rnd,"A_us":vals["A"],"B_us":vals["B"],"speedup":vals["A"]/vals["B"],"reduction_percent":100*(1-vals["B"]/vals["A"])})
    av=statistics.median([x["A_us"] for x in rstats]);bv=statistics.median([x["B_us"] for x in rstats])
    results.append({**c,"actual_block_nums":sorted(blocks[cid]),"A_us":av,"B_us":bv,"speedup_A_over_B":av/bv,"B_time_reduction_percent":100*(1-bv/av),"B_minus_A_us":bv-av,"per_round_reduction_percent_range":[min(x["reduction_percent"] for x in rstats),max(x["reduction_percent"] for x in rstats)],"overall_us":summary,"rounds":rstats})
out={"metric":"msprof Task Duration(us), device task only, warmups and Task Wait Time excluded","aggregation":"median of six per-round medians; 100 samples per round per variant","raw_tasks":len(rows),"measured_tasks":len(data),"warmup_per_case_variant":51,"results":results,"source_csv":str(paths[0])}
(R/"performance.json").write_text(json.dumps(out,indent=2))
with (R/"measured_tasks.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print("programs depth blocks A_us B_us speedup B_time_reduction_percent")
for x in results:print(x["programs"],x["depth"],x["actual_block_nums"],round(x["A_us"],3),round(x["B_us"],3),round(x["speedup_A_over_B"],4),round(x["B_time_reduction_percent"],3))
