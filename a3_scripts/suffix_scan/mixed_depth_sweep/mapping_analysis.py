import pathlib,json,math,hashlib
R=pathlib.Path(__file__).resolve().parent;P=R.parent
perf=json.loads((R/"performance.json").read_text())
metadataA=(R/"A.metadata.txt").read_text();metadataB=(R/"B.metadata.txt").read_text()
paths={"A":P/"triton-cache/23JGCMRGSDFYV2LV76E7OF54DR6OIA4EXGJLOICVFBD4B5SKPX6Q","B":P/"triton-cache/WA3EYVZIG3SFKMDAS7HXO7R2LZAB675I5YLAJA26KHPV77WMFUNQ"}
for label,stem in [("A","_suffix_local_scan_kernel"),("B","_suffix_local_scan_kernel_early_exit")]:
    meta=json.loads((paths[label]/(stem+".json")).read_text())
    text=metadataA if label=="A" else metadataB
    assert meta["hash"] in text,(label,meta["hash"])
b0=next(x["B_us"] for x in perf["results"] if x["long_requests"]==0)
b1=next(x["B_us"] for x in perf["results"] if x["long_requests"]==96)
analysis=[]
for x in perf["results"]:
    selected=set(x["long_batch_indices"])
    paircounts=[int(c in selected)+int(c+48 in selected) for c in range(48)]
    maximum=max(paircounts);pred=b0+(b1-b0)*maximum/2
    analysis.append({"case":x["id"],"long_requests":x["long_requests"],"layout":x["layout"],"seed":x["seed"],"long_counts_per_hardware_block":paircounts,"max_long_requests_on_one_block":maximum,"long_programs_on_busiest_block":maximum*5,"predicted_B_us":pred,"measured_B_us":x["B_us"],"residual_us":x["B_us"]-pred,"residual_percent":100*(x["B_us"]/pred-1)})
random_probability=[]
for count in [12,24,48,72,84]:
    q=math.comb(48,count)*2**count/math.comb(96,count) if count<=48 else 0
    observed=[x for x in analysis if x["long_requests"]==count and x["layout"]=="random"]
    random_probability.append({"long_requests":count,"no_double_long_pair_probability":q,"at_least_one_double_long_pair_probability":1-q,"observed_five_seeds_double_pair_count":sum(x["max_long_requests_on_one_block"]==2 for x in observed),"seeds":5})
out={"mapping":"AutoBlockify binary: flattened logical id starts at BLOCKID and increments by 48; pid0=flat%96, pid1=(flat//96)%5. Each hardware block c processes requests c and c+48, each at all five history blocks.","evidence":{"B_BLOCKID":"B.disassembly.log address 0xf4","B_stride48":"B.disassembly.log addresses 0x310-0x324","B_unflatten":"B.disassembly.log addresses 0x338-0x360","A_BLOCKID":"A.disassembly.log address 0xec","A_stride48":"A.disassembly.log address 0x2e8","A_unflatten":"A.disassembly.log addresses 0x318-0x330","binary_metadata_hashes_matched":True},"model":"B_pred = B_all_short + (B_all_long-B_all_short)*(maximum long requests on any hardware block)/2","B_all_short_us":b0,"B_all_long_us":b1,"max_absolute_residual_us":max(abs(x["residual_us"]) for x in analysis),"max_absolute_residual_percent":max(abs(x["residual_percent"]) for x in analysis),"cases":analysis,"uniform_random_layout_probabilities":random_probability}
(R/"mapping_analysis.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="cases"},indent=2))
