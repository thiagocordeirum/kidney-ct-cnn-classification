import yaml
from pathlib import Path

runs_dir = Path("runs")
rows = []
for d in sorted(runs_dir.iterdir()):
    if not d.is_dir():
        continue
    ay = d / "args.yaml"
    ty = d / "test_results.yaml"
    if not (ay.exists() and ty.exists()):
        continue
    with open(ay) as f:
        args = yaml.safe_load(f)
    with open(ty) as f:
        test = yaml.safe_load(f)
    rows.append({
        "run":   d.name,
        "model": args.get("model", "?"),
        "cond":  args.get("condition", "?"),
        "acc":   test.get("test_acc", 0),
        "loss":  test.get("test_loss", 0),
        "prec":  test.get("test_precision", 0),
        "rec":   test.get("test_recall", 0),
        "f1":    test.get("test_f1", 0),
    })

print("run,model,cond,acc,loss,prec,rec,f1")
for r in rows:
    print("{},{},{},{},{},{},{},{}".format(
        r["run"], r["model"], r["cond"],
        r["acc"], r["loss"], r["prec"], r["rec"], r["f1"]))
