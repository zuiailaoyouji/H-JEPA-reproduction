"""Stop only the three user-authorized legacy process groups; retain all artifacts."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/home/songxw/H-JEPA')
BASE=Path('/home/songxw/hjepa-data/ckpts')
JOBS=[('ant','ant_hjepa_l2',590971,590969),
      ('pusht','pusht_hjepa_l3',4025065,4025063),
      ('cube','cube_hjepa_l3',180048,180043)]
manifest=[]
for env,model,pid,pgid in JOBS:
    proc=Path(f'/proc/{pid}')
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'main_hjepa.py' in command and f'--config-name {model}' in command,(pid,command)
    assert os.getpgid(pid)==pgid
    assert proc.stat().st_uid==os.getuid()
    run=BASE/env/model/'seed42'
    last=run/'lightning_resume/last.ckpt'
    assert last.is_file() and last.stat().st_size>1000000
    row={'model':model,'run_dir':str(run),'pid':pid,'pgid':pgid,'command':command,
         'checkpoint':str(last),'checkpoint_bytes':last.stat().st_size,
         'checkpoint_mtime':dt.datetime.fromtimestamp(last.stat().st_mtime).isoformat(),
         'action':'SIGTERM to the model-specific process group; all files retained'}
    manifest.append(row)
out=ROOT/'local_env/fixes_20261027/stopped_legacy_jobs.json'
out.write_text(json.dumps(manifest,indent=2))
for row in manifest:
    os.killpg(row['pgid'],signal.SIGTERM)
deadline=time.monotonic()+10
while time.monotonic()<deadline and any(Path(f"/proc/{r['pid']}").exists() for r in manifest):
    time.sleep(.2)
for row in manifest:
    if Path(f"/proc/{row['pid']}").exists():
        # This is still the validated job; no other process group is touched.
        try:os.killpg(row['pgid'],signal.SIGKILL)
        except ProcessLookupError:pass
        row['forced_stop']=True
    row['stopped_at']=dt.datetime.now().astimezone().isoformat(timespec='seconds')
out.write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
