import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

root, target = map(Path,sys.argv[1:])
db, = root.glob("PROF_*/msprof_*.db")
con=sqlite3.connect(db)
marks=con.execute("""select m.startNs,m.endNs,s.value from MSTX_EVENTS m
                     join STRING_IDS s on s.id=m.message where s.value glob 'KDATIME_*'
                     order by m.startNs""").fetchall()
assert len(marks)>0 and len(marks)%11==0
starts=[m[0] for m in marks]
groups=defaultdict(list)
for launch, a,b,i,name in con.execute("""
    select p.startNs,t.startNs,t.endNs,t.globalTaskId,n.value
    from CANN_API p join TASK t on p.connectionId=t.connectionId
    join COMPUTE_TASK_INFO k on k.globalTaskId=t.globalTaskId
    join STRING_IDS n on n.id=k.name join STRING_IDS kind on kind.id=t.taskType
    join STRING_IDS api on api.id=p.name
    where kind.value like 'KERNEL%' and api.value='launch' order by t.startNs
"""):
    pos=bisect.bisect_right(starts,launch)-1
    if pos<0 or launch>marks[pos][1]:
        continue
    key=marks[pos][2]
    assert a<b
    groups[key].append((a,b,i,name))
assert set(groups)=={m[2] for m in marks}
samples=defaultdict(list)
names={}
for key, tasks in groups.items():
    _,batch,mode,rep=key.split("_")
    batch=int(batch)
    assert len(set(t[2] for t in tasks))==len(tasks)
    labels=[t[3] for t in tasks]
    assert mode=="whole" and labels==["main_kernel"],(key,labels)
    k=f"{batch}_{mode}"
    if k in names:assert names[k]==labels,(key,labels,names[k])
    names[k]=labels
    elapsed=0
    end=0
    for a,b,_,_ in tasks:
        elapsed+=max(0,b-max(a,end))
        end=max(end,b)
    samples[k].append(elapsed/1000)
assert len(samples)==len(marks)//11 and all(len(v)==11 for v in samples.values())
target.write_text(json.dumps({k:{"median_us":median(v),"samples_us":v,"kernels":names[k]} for k,v in samples.items()}))
print(target, {k:median(rows) for k,rows in samples.items()})
