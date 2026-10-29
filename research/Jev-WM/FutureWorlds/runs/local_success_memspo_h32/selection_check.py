import json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,'scripts')
import local_checkpoint_eval as mod
from futureworlds.checkpoints import file_sha256
c=[dict(step=s,lpips32=v) for s,v in zip(mod.STEPS,[.04,.02,.03,.05])]
assert mod.choose_checkpoint(c)["step"]==100
assert mod.choose_checkpoint([dict(step=s,lpips32=.02) for s in mod.STEPS])["step"]==50
for bad in [c[:-1],[dict(step=s,lpips32=float('nan')) for s in mod.STEPS]]:
 try:mod.choose_checkpoint(bad)
 except ValueError:pass
 else:raise AssertionError("Invalid validation must be rejected")
with tempfile.TemporaryDirectory(prefix="selection_check_",dir="runs/local_success_memspo_h32") as tmp:
 out=Path(tmp).resolve();m=out/"data/validation/manifest.json";m.parent.mkdir(parents=True)
 m.write_text(json.dumps(dict(schema_version=1,dataset="robocasa",cases=[dict(id="valid_case",episode_id="valid_case",file="x.npz",sha256="fixture")])))
 for step in mod.STEPS:
  ck=out/f"train/checkpoints/step_{step:06d}";ck.mkdir(parents=True);(ck/"experiment.json").write_text("{}")
 calls=[]
 def fake_launch(a,command,args,distributed=True):
  args=[str(x) for x in args];calls.append((command,args))
  def get(k):return args[args.index(k)+1]
  if command=="export-model":
   bundle=Path(get("--output"));bundle.mkdir(parents=True,exist_ok=True);(bundle/"manifest.json").write_text("{}")
  elif get("--manifest")==str(m):
   dest=Path(get("--output"));dest.mkdir(parents=True,exist_ok=True);step=int(dest.name.split("_")[-1]);score=next(x["lpips32"] for x in c if x["step"]==step)
   val=dict(state="COMPLETE",protocol=dict(cases=["valid_case"],manifest_sha256=file_sha256(m),bundle_manifest_sha256=file_sha256(Path(get("--bundle"))/"manifest.json")),per_case=[{}],aggregate={"32":{"lpips":score}})
   (dest/"summary.json").write_text(json.dumps(val))
  else:
   # Test is launched only after selection is frozen, and uses the selected bundle.
   selected=json.loads((out/"checkpoint_selection.json").read_text())
   assert selected["selected_step"]==100 and not selected["test_used_for_selection"]
   assert Path(get("--bundle")).resolve()==out/"validation_bundles/step_000100"
 a=SimpleNamespace(output=str(out),bundle="fixture_sft",text_model="fixture_t5",dry_run=False,gpus=8)
 with patch.object(mod,"launch",fake_launch):mod.evaluate_checkpoints(a)
 assert len([x for x in calls if x[0]=="evaluate"])==5
 assert (out/"evaluation/validation/rl").resolve()==out/"validation_checkpoints/step_000100"
print("PASS: validation selection, early-step tie break, missing/NaN rejection, selected aliases, test only after frozen selection")

