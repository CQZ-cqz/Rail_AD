from pathlib import Path
import re
from datetime import datetime
import statistics

p=Path(r"c:\Users\Mark\AppData\Roaming\Code\User\workspaceStorage\4fec56520a5d1d9b9275fcee33f99207\GitHub.copilot-chat\chat-session-resources\c28a6bab-5d71-475c-971b-01a75cfa4607\call_A3MnAlz7sJV1NwVLrlCHLm01__vscode-1775545844647\content.txt")
text=p.read_text(encoding='utf-8')
lines=text.splitlines()
start_re=re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Starting acquisition')
end_re=re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Acquisition complete: (\d+) valid points')

starts=[]
ends=[]
for ln in lines:
    m=start_re.search(ln)
    if m:
        starts.append(datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S,%f'))
    m2=end_re.search(ln)
    if m2:
        ends.append((datetime.strptime(m2.group(1),'%Y-%m-%d %H:%M:%S,%f'), int(m2.group(2))))

pairs=[]
for i, s in enumerate(starts):
    if i < len(ends):
        e, pts = ends[i]
        dur=(e-s).total_seconds()
        pairs.append({'index':i,'start':s,'end':e,'duration_s':dur,'points':pts})

if not pairs:
    print('No acquisition records found')
    raise SystemExit(0)

# compute stats
Durations=[p['duration_s'] for p in pairs]
Points=[p['points'] for p in pairs]
Total_time=(pairs[-1]['end']-pairs[0]['start']).total_seconds()
Total_points=sum(Points)
Avg_dur=statistics.mean(Durations)
Med_dur=statistics.median(Durations)
Avg_pps= Total_points/ sum(Durations) if sum(Durations)>0 else 0

print('Acquisitions:', len(pairs))
print(f'Total acquisition time span: {Total_time:.1f} s')
print(f'Total points acquired: {Total_points:,}')
print(f'Avg per-acq duration: {Avg_dur:.2f} s, median: {Med_dur:.2f} s')
print(f'Avg points/sec during acquisition: {Avg_pps:,.0f} pts/s')
print('\nPer acquisition:')
for p in pairs:
    print(f"#{p['index']:02d}: duration={p['duration_s']:.2f}s points={p['points']:,}")

# basic estimate: disk write time unknown; approximate end-to-end per segment
# end-to-end = acquisition duration + (Total_time - sum(Durations)) / len(pairs) as rough estimate
other_time = Total_time - sum(Durations)
if other_time < 0:
    other_time = 0
avg_other = other_time / len(pairs)
print('\nEstimated additional per-segment time (queue/IO/overhead average): {:.2f} s'.format(avg_other))

# if files exist in pending_dir, estimate write rate by file size / avg_other
from pathlib import Path as P
import os
cfg_pending = P('data/pending')
files = sorted(cfg_pending.glob('segment_*.ply'))
if files:
    sizes = [f.stat().st_size for f in files[-len(pairs):]]
    total_bytes=sum(sizes)
    if avg_other>0:
        print('Estimated bytes written for last segments: {:,} bytes'.format(total_bytes))
        print('Estimated average disk write rate (bytes/s): {:,}'.format(int(total_bytes/ (avg_other*len(sizes)) )))
else:
    print('No pending PLY files found to estimate disk write rate.')
