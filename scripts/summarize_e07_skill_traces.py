"""Offline, non-semantic descriptive statistics for existing R2 routing traces."""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

def csvw(p, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]) if rows else []); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trace',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in a.trace.read_text(encoding='utf-8').splitlines() if x.strip()]
    occ=Counter(); bytask=defaultdict(Counter); trans=Counter(); switches=defaultdict(list); depths=defaultdict(list); weights=defaultdict(list); ent={'success':[],'failure':[]}
    byep=defaultdict(list)
    for r in rows:
        sid=int(r['skill_id'][0]); task=r['task']; occ[sid]+=1; bytask[task][sid]+=1; byep[r['episode_id']].append((r['query_step'],sid,task,bool(r['success'])))
        p=r['skill_probs'][0]; ent['success' if r['success'] else 'failure'].append(-sum(x*math.log(max(x,1e-12)) for x in p)); depths[task].append(r['expected_depth']); weights[task].append(r['layer_weights'][0])
    for e,vals in byep.items():
        vals=sorted(vals); ids=[x[1] for x in vals]; switches[vals[0][2]].append(sum(a!=b for a,b in zip(ids,ids[1:])))
        for x,y in zip(ids,ids[1:]): trans[(x,y)]+=1
    occrows=[]
    n=sum(occ.values()); probs=[]
    for i in range(16):
        q=occ[i]/n if n else 0; probs.append(q); occrows.append({'skill_id':i,'count':occ[i],'frequency':q})
    csvw(a.out/'skill_occupancy.csv',occrows)
    csvw(a.out/'skill_transition_matrix.csv',[{'from_skill':i,'to_skill':j,'count':trans[(i,j)]} for i in range(16) for j in range(16)])
    summary=[]
    for t in sorted(depths):
        avgw=[mean(x[i] for x in weights[t]) for i in range(len(weights[t][0]))]
        summary.append({'task':t,'queries':len(depths[t]),'mean_expected_depth':mean(depths[t]),'std_expected_depth':pstdev(depths[t]),'mean_skill_switches':mean(switches[t]),'mean_layer_weights_json':json.dumps(avgw)})
    summary.append({'task':'overall','queries':n,'mean_expected_depth':mean(x['expected_depth'] for x in rows),'std_expected_depth':pstdev(x['expected_depth'] for x in rows),'mean_skill_switches':mean(sum(v) for v in switches.values())/len(switches),'mean_layer_weights_json':json.dumps({'used_skill_categories':sum(x>0 for x in probs),'effective_categories':math.exp(-sum(x*math.log(max(x,1e-12)) for x in probs)),'posterior_entropy_success':mean(ent['success']),'posterior_entropy_failure':mean(ent['failure']),'entropy_success_minus_failure':mean(ent['success'])-mean(ent['failure'])})})
    csvw(a.out/'routing_summary.csv',summary)

if __name__=='__main__': main()
