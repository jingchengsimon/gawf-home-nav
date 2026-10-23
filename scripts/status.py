"""Read exactly one run manifest; one SSH query, no state updates."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id')
    parser.add_argument('--no-update', action='store_true', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'nav-[a-z0-9]+(?:-[a-z0-9]+)+', args.experiment_id):
        parser.error('One complete experiment ID is required')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'experiments/monitoring/jobs' / (args.experiment_id + '.json')).read_text())
    alias = manifest['host']
    policy_path = root / '.local/ssh-connection-policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    direct = policy.get(alias) == 'direct'
    if not direct:
        check = subprocess.run(['ssh', '-O', 'check', alias], capture_output=True,
                               text=True, timeout=10)
        direct = check.returncode != 0
    if not direct:
        ssh_options = ['-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
                       '-o', 'ProxyCommand=false']
    else:
        ssh_options = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
                       '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
                       '-o', 'ConnectTimeout=8']
    payload = {k: manifest.get(k) for k in ('experiment_id', 'mode', 'units', 'budget', 'execution', 'completion_evidence')}
    probe = '''import json,re,subprocess
from pathlib import Path
run=json.loads(PAYLOAD)
for unit in run['units']:
 p=Path(unit['result_path']).expanduser(); summary=p/'summary.json'; exitfile=p/'exit.json'
 row={'experiment_id':run['experiment_id'],'unit':unit['name']}
 if summary.is_file():
  s=json.loads(summary.read_text())
  for k in ['status','completed_episodes','expected_episodes','episode_count','success_rate','spl','completed_updates','completed_stages','world_size','global_frames','global_frames_per_rank0_wall_second']:
   if k in s: row[k]=s[k]
 elif run.get('mode')=='formal_train':
  progress=p/'progress.json'
  if progress.is_file():
   state=json.loads(progress.read_text())
   for k in ['status','global_frames','updates_completed','saved_global_frames','target_global_frames','checkpoint_path']:
    if k in state: row[k]=state[k]
  else:
   row.update(status='pending',global_frames=0,target_global_frames=run['budget']['target_global_frames'])
 elif run.get('completion_evidence'):
  row.update(run['completion_evidence'])
 elif run.get('mode')=='train-smoke':
  log=Path(unit['training_log_path']).expanduser()
  matches=re.findall(r'update: (\\d+).*frames: (\\d+)',log.read_text()) if log.is_file() else []
  row.update(completed_updates=int(matches[-1][0])+1 if matches else 0,expected_updates=run['budget']['updates'],global_frames=int(matches[-1][1]) if matches else 0)
 else:
  episodes=p/'episodes.jsonl'; row.update(completed_episodes=len(episodes.read_text().splitlines()) if episodes.is_file() else 0,summary_present=False)
 if exitfile.is_file():
  e=json.loads(exitfile.read_text()); row['failure']=e.get('error') or (('worker exit '+str(e['return_code'])) if e.get('return_code') else None)
 print(json.dumps(row))
job=(run.get('execution') or {}).get('job_id')
if job:
 q=subprocess.run(['squeue','-h','-j',str(job),'-o','%i|%T|%R|%S'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
 if q.stdout.strip(): print(q.stdout.strip())
 else:
  a=subprocess.run(['sacct','-n','-X','-j',str(job),'--format=JobID,State,Elapsed,ExitCode','-P'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10); print(a.stdout.strip())
'''.replace('PAYLOAD', repr(json.dumps(payload)))
    command = shlex.quote(manifest['environment']['python']) + ' -B -c ' + shlex.quote(probe)
    subprocess.run(['ssh', *ssh_options, alias, command], check=True, timeout=40)


if __name__ == '__main__':
    main()
