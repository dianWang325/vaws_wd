import pathlib,csv,json,statistics
R=pathlib.Path(__file__).resolve().parent
cases=[("op_baseline_A","A","long12_interleaved"),("op_contiguous","B","long12_contiguous"),("op_interleaved","B","long12_interleaved")]
mixed=json.loads((R.parent/"mixed_depth_sweep/manifest.json").read_text())
results=[]
for directory,variant,caseid in cases:
    roots=list((R/directory).glob("OPPROF_*"));assert len(roots)==1
    root=roots[0]
    with (root/"OpBasicInfo.csv").open() as f:basic=next(csv.DictReader(f))
    with (root/"PipeUtilization.csv").open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==48
    case=next(c for c in mixed["cases"] if c["id"]==caseid)
    selected=set(case["long_batch_indices"])
    coredata=[]
    for row in rows:
        block=int(row["block_id"]);assert row["sub_block_id"]=="vector0"
        nlong=int(block in selected)+int(block+48 in selected)
        elapsed=float(row["aiv_time(us)"]);cycles=int(row["aiv_total_cycles"])
        assert abs(elapsed-cycles/float(basic["Current Freq"]))<.002
        coredata.append({"block_id":block,"long_requests_on_block":nlong,"aiv_time_us":elapsed,"aiv_cycles":cycles,"scalar_time_us":float(row["aiv_scalar_time(us)"]),"vector_time_us":float(row["aiv_vec_time(us)"])})
    coredata.sort(key=lambda x:x["block_id"]);groups=[]
    for nlong in sorted(set(x["long_requests_on_block"] for x in coredata)):
        vals=[x for x in coredata if x["long_requests_on_block"]==nlong]
        ts=[x["aiv_time_us"] for x in vals]
        groups.append({"long_requests_on_block":nlong,"blocks":[x["block_id"] for x in vals],"count":len(vals),"min_us":min(ts),"median_us":statistics.median(ts),"max_us":max(ts)})
    result={"case":caseid,"variant":variant,"directory":str(root),"task_duration_us":float(basic["Task Duration(us)"]),"freq_MHz":float(basic["Current Freq"]),"groups":groups,"per_block":coredata,"slowest_block":max(coredata,key=lambda x:x["aiv_time_us"]),"fastest_block":min(coredata,key=lambda x:x["aiv_time_us"])}
    results.append(result)
    print(json.dumps({k:v for k,v in result.items() if k!="per_block"},indent=2))
    with (R/(directory+".per_block.csv")).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(coredata[0]));w.writeheader();w.writerows(coredata)
(R/"op_per_block_results.json").write_text(json.dumps(results,indent=2))
lines=["# 同一次算子调用的逐 block 执行时间证据","",
"结论：已取得比总耗时模型更直接的证据。msprof op 的 PipeUtilization.csv 记录同一次选定 kernel 调用中全部48个block的aiv_time(us)，其分组与二进制固定分配模型一致。",
"",
"| 调用 | 整个task时间(µs) | 分组 | block数 | block执行时间中位数(µs) | 最小—最大(µs) |",
"|---|---:|---|---:|---:|---:|"]
for r in results:
    for g in r["groups"]:
        lines.append(f"| {r['variant']} / {r['case']} | {r['task_duration_us']:.3f} | 每block长请求数={g['long_requests_on_block']} | {g['count']} | {g['median_us']:.3f} | {g['min_us']:.3f}—{g['max_us']:.3f} |")
lines+=["",
"A为固定24轮，即使block分配到短请求也扫描满24轮。B中短请求执行5轮、长请求执行24轮。",
"在交错布局中，block 0/8/16/24/32/40分别承接两个长请求，共6个block各自执行10个满深度program，其余42个block均为短请求。整体task时间接近6个长block的最大执行时间，而非42个短block的时间。",
"连续布局中，block0–11各有一个长请求和一个短请求，其余36个block全短，最长路径缩短到中间档。数据支持提前退出只有缩短最忙block路径才能缩短整体任务。",
"",
"## 计量边界","",
"这是同一次被选定调用的逐block设备执行时长，不是把重复压测累计周期除以调用次数得到的估算。msprof op 的kernel重放用于采集PMU，不用这些单次值替代前面多轮压测的性能统计。",
"当前频率1850MHz；逐行验证aiv_time(us)=aiv_total_cycles/1850。保留原始OpBasicInfo.csv与PipeUtilization.csv。",
"没有逐block绝对start/end时间戳，因此没有声称获得精确起止时间轨或给出各block完成的绝对先后时刻。可验证的是各block自身执行时长差异，以及整体task耗时与最长block执行时长相近。",
"",
"## 补采过程与未采用的数据","",
"先检查原task-based数据：只有整个task时间及汇总PMU，不能提供逐block时间。",
"随后补采四组sample-based数据，每组100次预热、1000次重复，输出均与CPU参考精确一致。这些采集在ai_vector_core.db中报告50个core ID，未建立它们与48个launch block的一一对应；因此未将这些累计计数当作本报告的逐block结论。原始数据、sample_timeline.csv、per_core_results.json均保留以便审计。",
"最终采用msprof op：PipeUtilization,BasicInfo，launch-count=1，launch-skip-before-match=5，warm-up=5；应用每组先执行5次再执行5次，每个程序最终完整输出仍精确一致。",
"",
"## 环境与留存","",
"80.5.9.129，保留容器codex-scan20854-c8ba92bf-20261112。镜像0015250ad020，Triton-Ascend3.6.0.post2，CANN9.2.1，Ascend910_9392，设备0。grid=(96,5)，480逻辑program，48硬件block，MAX_DEPTH24，BLOCK256，stride1025。A/B源码仍引用父目录kernel_A.py和kernel_B.py，未改写。",
"",
"全部文件：/mnt/share/w00985415/scan20854_c8ba92bf_20261112/ab_loop_exit/per_core_followup/",
"",
"逐block原始文件分别位于op_baseline_A/OPPROF_*/、op_contiguous/OPPROF_*/、op_interleaved/OPPROF_*/下。",
"op_per_block_results.json及op_*.per_block.csv保留全部48个block；op_analyze.py可复算。样本profiling见各案例目录、collect.sh、run.py、analysis.log；单调用profiling见run_op.py与op_*.log。",
"容器继续保留，实验进程已退出。"]
(R/"report.md").write_text("\n".join(lines)+"\n")
