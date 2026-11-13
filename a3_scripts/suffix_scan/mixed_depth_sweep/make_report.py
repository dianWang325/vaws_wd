import pathlib,json,statistics
R=pathlib.Path(__file__).resolve().parent
p=json.loads((R/"performance.json").read_text());m=json.loads((R/"mapping_analysis.json").read_text())
def f(v,n=3):return f"{v:.{n}f}"
lines=["# Scan 长短匹配混合实验","",
"结论：混合场景的收益主要由固定硬件分配下最忙 block 的累计工作量决定，不能由满深度占比单独预测。此前理想均衡估算在本配置下偏乐观；逻辑交错并不必然更均衡，连续排列也不必然更差。",
"",
"固定 96 请求 × 5 历史块 = 480 个逻辑 program，实际发射 48 个硬件 block。短请求每个 program 最长匹配恰好 4，B 执行 5 轮；长请求每个 program 匹配到 MAX_DEPTH=24，B 执行 24 轮。",
"",
"## 实测结果","",
"加速比=A设备耗时/B设备耗时；大于1表示B更快。随机列是五个固定随机种子的最小值—最大值，不是置信区间。",
"",
"| 满深度请求占比 | 连续排列 A/B | 交错排列 A/B | 随机排列 A/B 范围 | 理想均衡成本模型 A/B |",
"|---|---:|---:|---:|---:|"]
for g in p["grouped"]:
    if "control" in g:
        val=g["control"]["speedup_median"];cont=inter=rand=f(val)
    else:
        cont=f(g["contiguous"]["speedup_median"]);inter=f(g["interleaved"]["speedup_median"])
        rand=f(g["random"]["speedup_min"])+"—"+f(g["random"]["speedup_max"])
    lines.append(f"| {g['long_percent']:g}% | {cont}× | {inter}× | {rand}× | {f(g['balanced_model_speedup_median'])}× |")
lines+=["",
"0%和100%为共享端点对照，没有重复测相同布局。随机种子11/23/37/53/71；5个布局仅是有限观察。",
"12.5%（12个长请求）：连续排列B≈2366µs、A≈3632µs，收益约1.535×；交错排列B≈3673µs，反而慢约1.11%。五个随机布局中3个落在≈2366µs档、2个落在≈3672–3673µs档。",
"25%和50%：连续排列仍约1.535×；交错及本轮全部随机种子均约0.989×。75%及以上全部布局均约0.989×。",
"",
"## 二进制与固定分配模型","",
"实际编译二进制中，扁平program id从BLOCKID开始，每次递增48，随后还原二维program id。grid=(96,5)时：",
"",
"~~~text",
"flat = hardware_block_id + 48*k",
"request = flat % 96",
"history_block = (flat // 96) % 5",
"~~~",
"",
"因此硬件block c（0≤c<48）固定处理请求c和c+48的全部五个历史块，共10个逻辑program。",
"反汇编证据：B.disassembly.log的0xf4读取BLOCKID，0x310–0x324加48并循环，0x338–0x360计算二维索引；A.disassembly.log的0xec、0x2e8、0x318–0x330有对应证据。缓存元数据hash已与本轮实际KernelMetadata核对一致；二进制指纹见binary-SHA256SUMS。",
"",
"| 最忙block对应的两个请求中，满深度请求数 | 最忙block的逻辑工作 | 预测B时间 |",
"|---:|---|---:|",
f"| 0 | 10个短program | {f(m['B_all_short_us'])} µs |",
f"| 1 | 5个长+5个短program | {f((m['B_all_short_us']+m['B_all_long_us'])/2)} µs |",
f"| 2 | 10个长program | {f(m['B_all_long_us'])} µs |",
"",
f"按各布局的请求对计算最长累计工作量，全部37组的最大绝对预测误差为{f(m['max_absolute_residual_us'])}µs / {f(m['max_absolute_residual_percent'],4)}%。模型使用两个实测端点校准，并非独立逐core cycle测量；未另外采集per-core时间线。",
"",
"连续排列将前L个请求设为长匹配。当0<L≤48时，请求c+48都是短匹配，所以任意硬件block最多一个长请求，尾部在中间档。L>48后必然有双长请求对。",
"交错排列取floor(k*96/L)为长请求位置；12.5%、25%、50%的周期让c与c+48同时被选中，逻辑上的交错恰好产生全长硬件尾部。",
"",
"## 随机布局的两档分布","",
"从96个请求中均匀选择L个长请求时，L≤48下，没有双长请求对的精确组合概率为C(48,L)*2^L/C(96,L)。出现任一双长请求对，即有10个program全长的硬件block。",
"",
"| 长请求数 / 占比 | 至少一个双长请求对的理论概率 | 本轮五种子观察 |",
"|---|---:|---:|"]
for x in m["uniform_random_layout_probabilities"]:
    lines.append(f"| {x['long_requests']} / {100*x['long_requests']/96:g}% | {100*x['at_least_one_double_long_pair_probability']:.4f}% | {x['observed_five_seeds_double_pair_count']}/5 |")
lines+=["",
"以上概率由固定分配和均匀随机抽样推导，不是把五个种子的频率当作真实概率。实际请求顺序、深度相关性、grid或编译器分配改变后需重新计算。",
"",
"## 环境、校验与采集","",
"2026-11-12（Asia/Shanghai）；80.5.9.129；独立临时容器codex-scan20854-c8ba92bf-20261112继续保留。",
"镜像sha256:0015250ad02044d522977162b0e9b1a9006fae54aee44f792c814c1628b17f82。",
"Triton-Ascend3.6.0.post2、Triton3.6.0、CANN9.2.1、torch2.10.0+cpu、torch_npu2.10.0.post7、Ascend910_9392。设备0执行前无运行进程、AICore0%。",
"实际compile_mode=simd_simt_template、parallel_mode=simd、mix_mode=aiv、AutoBlockify=True；MAX_DEPTH24、BLOCK256、stride1025、num_warps4、Lp1=1026。",
"A为#20854 commit c8ba92bf77e499a1b3ed5da7b9d881757f871fd9的固定循环；B保持相同独立加载mask及地址保护，只增加归约退出条件。未修改上一轮kernel，未加入#19394的动态加载mask。",
"请求映射为恒等映射，所有请求num_sampled=1；每行通过长度d的历史重复片段和尾部d+1个相同token保证五个program深度均恰为d，独立CPU逐候选检查。",
"37组、74项编译和执行校验均通过，35520个完整scratch int64与CPU逐项精确一致；性能结束后再次比较通过。",
"msprof Task Duration(us)设备任务口径；每组每版预热51次，六轮各50次，交替AB/BA并打乱布局顺序，每组每版保留300样本。核对25974个任务：3774预热+22200测量，全部Block Num48。主结果为六轮中位数再取中位数，不含主机提交间隔及Task Wait Time。",
"",
"## 完整布局耗时","",
"| 长请求数 | 布局 | seed | A (µs) | B (µs) | A/B | B耗时降幅 |",
"|---:|---|---:|---:|---:|---:|---:|"]
for x in p["results"]:
    lines.append(f"| {x['long_requests']} | {x['layout']} | {x['seed'] if x['seed'] is not None else '—'} | {f(x['A_us'])} | {f(x['B_us'])} | {f(x['speedup_A_over_B'])}× | {f(x['B_time_reduction_percent'],2)}% |")
lines+=["",
"## 边界与留存","",
"本轮验证了硬件block尾部支配混合场景，推翻了仅凭满深度占比预测收益及逻辑交错必然更优的简化判断。不同grid、更多独立请求、program内部匹配混合和其他编译器调度未覆盖；没有模型或端到端压测，不外推业务吞吐。",
"",
"全部文件：/mnt/share/w00985415/scan20854_c8ba92bf_20261112/ab_loop_exit/mixed_depth_sweep/",
"",
"- mixed.py、manifest.json、correctness.json、*.inputs_reference.npz、*.scratch.npy：输入、完整深度、CPU参考和输出。",
"- analyze.py、performance.json、measured_tasks.csv、analysis.log：全部样本、逐轮与汇总。",
"- mapping_analysis.py、mapping_analysis.json、mapping-analysis.log、A.disassembly.log、B.disassembly.log、binary-SHA256SUMS：映射、模型、概率与二进制证据。",
"- launch_plan.json、profiling.log、msprof/：完整顺序、日志与原始数据。",
"- A.metadata.txt、B.metadata.txt、A.*.txt、B.*.txt、SHA256SUMS：编译信息与指纹。",
"- host-before.log、host-after.log、container-retained.log：设备和保留状态。",
"",
"容器继续保留，实验进程已退出。"]
(R/"report.md").write_text("\n".join(lines)+"\n")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.2))
    groups=p["grouped"];xs=[g["long_percent"] for g in groups]
    def value(g,key,stat="speedup_median"):return g.get(key,g.get("control"))[stat]
    ax.plot(xs,[g["balanced_model_speedup_median"] for g in groups],"--",color="0.5",label="Ideal balanced-cost estimate")
    for key,color,label in [("contiguous","#2563eb","Contiguous"),("interleaved","#dc2626","Interleaved"),("random","#059669","Random (5 seeds, median/range)")]:
        ax.plot(xs,[value(g,key) for g in groups],"o-",color=color,label=label,markersize=5)
        if key=="random":ax.fill_between(xs,[value(g,key,"speedup_min") for g in groups],[value(g,key,"speedup_max") for g in groups],color=color,alpha=.14)
    ax.axhline(1,color="black",linewidth=.8);ax.set_xticks(xs)
    ax.set_xlabel("Full-depth requests (%)");ax.set_ylabel("Speedup A / B (higher is better)")
    ax.set_title("480 logical programs / 48 hardware blocks\nShort depth 4, long depth 24")
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(R/"mixed_speedup.png",dpi=180);plt.close(fig)
    print("REPORT AND PLOT CREATED")
except ImportError:print("REPORT CREATED; matplotlib unavailable, skipped plot")
