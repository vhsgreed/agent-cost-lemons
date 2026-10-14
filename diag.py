"""Post-run diagnostics (exploratory, labelled as such in the write-up). Not part of the registered run."""
import sys,numpy as np
from collections import defaultdict,Counter
import analyze as A
bai=A.load_bai()
v2=bai['claude-sonnet-4.5_new_v2_4runs']
print('v2 runs',len(v2),sorted(Counter(k for _,k in v2).items(),key=str),'zero-call',sum(r['calls']==0 for r in v2.values()))
print('v2 runs/task',Counter(Counter(i for i,_ in v2).values()))
cells=[]
for a in bai:
    if a=='claude-sonnet-4.5_new_v2_4runs': continue
    by=defaultdict(dict)
    for (i,k),r in bai[a].items():
        if r['calls']>0: by[i][k]=r
    for i,v in by.items():
        if len(v)==4:
            t=sorted((v[k]['tok'],v[k]['calls'],v[k]['res']) for k in v)
            cells.append((t[-1][0]/t[0][0],a,i,t))
cells.sort(key=lambda c:-c[0])
for c in cells[:6]: print(round(c[0]),c[1][:14],c[2],[(int(x[0]),x[1],x[2]) for x in c[3]])
minc=np.array([c[3][0][1] for c in cells]); print('cells',len(cells),'min-run calls<=3:',int((minc<=3).sum()))
for th in (3,5,10):
    ok=np.array([c[0] for c in cells if c[3][0][1]>th]); print(f'excl min-run calls<={th}: n={len(ok)} max={ok.max():.1f} median={np.median(ok):.2f} >10x={np.mean(ok>10):.3f}')
# A5 by resolve-labelled share
for a in bai:
    lab=sum(r['res'] is not None for r in bai[a].values()); print(a[:24],'labelled',lab,'/',len(bai[a]),'resolve rate',round(np.mean([r['res'] for r in bai[a].values() if r['res'] is not None]),3))
# --- exploratory alternatives for A3/A4 (unit of analysis is not stated in the paper)
meta=A.load_meta(); comp=[a for a in bai if a!='claude-sonnet-4.5_new_v2_4runs']
full={}
for a in comp:
    by=defaultdict(dict)
    for (i,k),r in bai[a].items():
        if r['calls']>0: by[i][k]=r
    full[a]={i:v for i,v in by.items() if len(v)==4}
def a34(d,t,label):
    d,t=np.array(d),np.array(t); e,h=t[d==0],t[d==2]
    print(f'{label}: n={len(t)} tau_b={A.kendall_tau_b(d,t):.3f} easy>hardmean={np.mean(e>h.mean()):.3f} hard<easymean={np.mean(h<e.mean()):.3f}')
D=lambda i:A.DIFF3[meta[i]['difficulty']]
runs=[(D(i),v[k]['tok']) for a in comp for i,v in full[a].items() for k in v]; a34(*zip(*runs),'run level, pooled')
common=set.intersection(*[set(full[a]) for a in comp])
tm=[(D(i),np.mean([np.mean([full[a][i][k]['tok'] for k in full[a][i]]) for a in comp])) for i in sorted(common)]; a34(*zip(*tm),'per-task mean over models')
for a in comp:
    z=[(D(i),np.mean([v[k]['tok'] for k in v])) for i,v in full[a].items()]; a34(*zip(*z),'  '+a[:22])
# --- exploratory A5: accuracy per cost rank bucket (paper Fig 2b layout)
acc=np.zeros(4); n=0
for a in comp:
    for i,v in full[a].items():
        ks=sorted(v,key=lambda k:(v[k]['tok'],k)); acc+=[int(v[k]['res']) for k in ks]; n+=1
print('accuracy Min/Lower/Upper/Max:',np.round(acc/n,3))
