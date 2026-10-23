"""Package the local project page and validate every recorded video and source."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT/"project-page"
ASSETS=PAGE/"dist/assets"
SUITE="nav-robothor-embclip-project-sixepisodes-twoseeds-sjc-v2-20261023"
ANALYSIS="nav-robothor-embclip-project-matched-gradcam-sjc-v2-20261023"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if sha(source)!=sha(dest):raise RuntimeError(f"Existing asset differs: {dest}")
    else:shutil.copy2(source,dest)


def probe(path,expected_frames,width,height):
    d=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=codec_name,width,height,r_frame_rate,nb_read_frames","-of","json",str(path)]))["streams"][0]
    if (d["codec_name"]!="h264" or (d["width"],d["height"])!=(width,height)
            or d["r_frame_rate"]!="12/1" or int(d["nb_read_frames"])!=expected_frames):
        raise RuntimeError(f"Invalid video {path}: {d}")
    return d


def extract(source,dest,frame,crop=False):
    if dest.exists():return
    vf=f"select=eq(n\\,{frame})"+(" ,crop=500:375:0:70" if crop else "")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-i",str(source),"-vf",vf,"-frames:v","1",str(dest)],check=True)


def main():
    source=ROOT/"results"/SUITE
    sm=json.loads((ROOT/"experiments/monitoring/jobs"/(SUITE+".json")).read_text())
    summary=json.loads((source/"summary.json").read_text())
    if sm["status"]!="complete" or summary["status"]!="complete" or len(summary["pairs"])!=12:
        raise RuntimeError("Suite is not complete")
    if sha(source/"summary.json")!=sm["completion_evidence"]["summary_sha256"]:
        raise RuntimeError("Suite summary hash mismatch")
    inventory=[];episodes=[]
    for spec in sm["episodes"]:
        eid=spec["episode_id"];leaf=source/eid;dest=ASSETS/"videos"/eid;dest.mkdir(parents=True,exist_ok=True)
        trials=[]
        for row in [r for r in summary["pairs"] if r["episode_id"]==eid]:
            seed=row["rollout_seed"];models=[];receipts=[];posters=[]
            for m in row["models"]:
                prefix=m["model"]+"_rollout"+str(seed)
                receipt=json.loads((leaf/(prefix+".json")).read_text())
                path=ROOT/m["video"]
                if sha(path)!=m["video_sha256"] or len(receipt["trajectory"])!=m["steps"]+1 or len(receipt["actions"])!=m["steps"]:
                    raise RuntimeError("Video/trajectory identity mismatch")
                metadata=probe(path,m["steps"]+1,1000,470)
                asset=dest/path.name;copy(path,asset)
                poster=dest/(prefix+"_poster.png");extract(path,poster,min(24,m["steps"]))
                final=dest/(prefix+"_full_path.png");extract(path,final,m["steps"],crop=True)
                models.append({**m,"asset":str(asset.relative_to(PAGE/"dist")),"final_path":str(final.relative_to(PAGE/"dist"))})
                receipts.append(receipt);posters.append(str(poster.relative_to(PAGE/"dist")))
                inventory.append({"source":str(path.relative_to(ROOT)),"asset":str(asset.relative_to(PAGE/"dist")),"sha256":m["video_sha256"],"bytes":path.stat().st_size,"probe":metadata})
            fairness=json.loads((leaf/("paired_rollout"+str(seed)+".fairness.json")).read_text())
            if not fairness["same_episode"] or not fairness["same_actual_pose"]:
                raise RuntimeError("Unfair pair")
            if receipts[0]["episode"]!=receipts[1]["episode"] or receipts[0]["start_actual"]!=receipts[1]["start_actual"]:
                raise RuntimeError("Task or actual pose differs")
            combined=ROOT/row["paired_video"]
            if sha(combined)!=row["paired_video_sha256"]:raise RuntimeError("Combined hash mismatch")
            metadata=probe(combined,max(r["steps"] for r in models)+1,1000,940)
            combined_asset=dest/combined.name;copy(combined,combined_asset)
            inventory.append({"source":str(combined.relative_to(ROOT)),"asset":str(combined_asset.relative_to(PAGE/"dist")),"sha256":row["paired_video_sha256"],"bytes":combined.stat().st_size,"probe":metadata})
            receipt_path=dest/("pair_rollout"+str(seed)+".json")
            receipt_path.write_text(json.dumps({"pair":row,"fairness":fairness,"rollouts":receipts},indent=2,sort_keys=True)+"\n")
            trials.append({"rollout_seed":seed,"models":models,"posters":posters,"poster":posters[0],
                "paired_asset":str(combined_asset.relative_to(PAGE/"dist")),"receipt_asset":str(receipt_path.relative_to(PAGE/"dist"))})
        if len(trials)!=2:raise RuntimeError("Missing repeat")
        e=receipts[0]["episode"]
        label={"FloorPlan_Val1_1_GarbageCan_0":"Same scene & goal · start A",
               "FloorPlan_Val1_1_GarbageCan_2":"Same scene & goal · start B",
               "FloorPlan_Val2_5_GarbageCan_9":"Same goal · different scene"}.get(eid,"Different scene & target")
        episodes.append({"episode_id":eid,"scene":e["scene"],"target":e["object_type"],"start":receipts[0]["start_actual"],"case_label":label,"trials":trials})
    analysis=ROOT/"results"/ANALYSIS
    am=json.loads((ROOT/"experiments/monitoring/jobs"/(ANALYSIS+".json")).read_text())
    a=json.loads((analysis/"summary.json").read_text())
    if am["status"]!="complete" or sha(analysis/"summary.json")!=am["completion_evidence"]["summary_sha256"]:
        raise RuntimeError("Analysis identity mismatch")
    for row in a["records"]:
        dest=ASSETS/"attribution"/row["plot"];copy(analysis/row["plot"],dest);row["asset"]=str(dest.relative_to(PAGE/"dist"))
    # Lead with four different scenes, retaining all 12 frames in the expanded view.
    a["records"].sort(key=lambda r:(r["step"]!=0, r["episode_id"]))
    stat=json.loads((PAGE/"sources/display_statistics.json").read_text())
    data={"episodes":episodes,"statistics":stat,"attribution":a,"action_names":["MoveAhead","RotateLeft","RotateRight","End","LookUp","LookDown"]}
    (ASSETS/"project-data.js").write_text("window.PROJECT = "+json.dumps(data,separators=(",",":"))+";\n")
    for path,name in [(ROOT/"experiments/monitoring/jobs"/(SUITE+".json"),"suite_manifest.json"),(source/"summary.json","suite_summary.json"),(analysis/"summary.json","attribution_summary.json")]:
        copy(path,PAGE/"sources"/name)
    for path in (PAGE/"sources").iterdir():
        if path.is_file():copy(path,ASSETS/"sources"/path.name)
    code=["scripts/embclip/canonical_core/gawf.py","scripts/embclip/canonical_core/rnn.py",
          "scripts/embclip/light_gawf1_config.py","scripts/embclip/light_config.py",
          "scripts/embclip/record_project_suite_sjc.py","scripts/embclip/analyze_project_saliency_sjc.py"]
    for p in code:copy(ROOT/p,ASSETS/"code"/Path(p).name)
    report={"status":"pass","individual_videos":24,"combined_videos":12,"paired_trials":12,
            "episode_ids":[e["episode_id"] for e in episodes],"scenes":sorted({e["scene"] for e in episodes}),
            "targets":sorted({e["target"] for e in episodes}),"attribution_frames":len(a["records"]),
            "source_suite":SUITE,"source_analysis":ANALYSIS,"videos":inventory}
    (PAGE/"sources/verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    copy(PAGE/"sources/verification.json",ASSETS/"sources/verification.json")
    print(json.dumps({k:v for k,v in report.items() if k!="videos"},indent=2))


if __name__=="__main__":main()
