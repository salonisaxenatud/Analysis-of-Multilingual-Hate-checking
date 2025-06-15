import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
OUT_DIR = r"Final Results/graphs/FINALPLOT"
os.makedirs(OUT_DIR, exist_ok=True)

ROOT  = r"D:\Delft\Uni\Mod 4\NLP for Society\Project\Data\Results\Latest\Final Results\Models"
LANGS = ("polish", "english", "hindi")

def as_prob(v: float) -> float:
    v = float(v)
    return 1 / (1 + np.exp(-v)) if (v < 0 or v > 1) else v

def pick_prob(row: pd.Series, base: str) -> float:
    """
    Return P(class = prediction) for *row*.
    Handles: separate yes/no cols, a single numeric col, or "p_yes,p_no".
    """
    yes, no = f"{base}_yes", f"{base}_no"
    if yes in row and no in row:
        p_yes, p_no = as_prob(row[yes]), as_prob(row[no])
        return p_yes if row["pred_here"] == 1 else p_no

    txt = str(row[base]).strip()
    if "," in txt:
        p_yes, p_no = (as_prob(float(x)) for x in txt.split(","))
        return p_yes if row["pred_here"] == 1 else p_no

    return as_prob(float(txt))
frames = []

for dirpath, _, filenames in os.walk(ROOT):
    if os.path.basename(dirpath).lower() == "graphs":
        continue

    model_name = os.path.relpath(dirpath, ROOT).split(os.sep)[0]

    for fname in (f for f in filenames if f.endswith("_dataset.csv")):
        origin = fname.split("_")[0].lower()
        fpath  = os.path.join(dirpath, fname)

        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip().str.lower()

        if any(f"pred_{l}" in df.columns for l in LANGS):
            for lang in LANGS:
                pcol  = f"pred_{lang}"
                base  = f"conf_{lang}"
                if pcol not in df.columns:
                    continue
                if not ({base, f"{base}_yes", f"{base}_no"} & set(df.columns)):
                    continue

                df["pred_here"] = df[pcol]
                frames.append(pd.DataFrame({
                    "language": lang,
                    "model":    f"{model_name.upper()}_{origin.capitalize()}",
                    "gt":       df["label"],
                    "pred":     df["pred_here"],
                    "correct":  df["label"] == df["pred_here"],
                    "prob":     df.apply(lambda r: pick_prob(r, base), axis=1)
                }))
            continue
        if "pred" not in df.columns:
            continue

        if "language" in df.columns:
            langs_in_file = df["language"].str.lower().unique()
        else:
            tokens = fname.lower().split("_")
            langs_in_file = [t for t in tokens[1:] if t in LANGS]

        for lang in langs_in_file:
            sub = (df[df["language"].str.lower() == lang] if "language" in df.columns else df).copy()
            if sub.empty:
                continue

            sub["pred_here"] = sub["pred"]
            base = "conf"

            frames.append(pd.DataFrame({
                "language": lang,
                "model":    f"{model_name.upper()}_{origin.capitalize()}",
                "gt":       sub["label"],
                "pred":     sub["pred_here"],
                "correct":  sub["label"] == sub["pred_here"],
                "prob":     sub.apply(lambda r: pick_prob(r, base), axis=1)
            }))

data = (pd.concat(frames, ignore_index=True)
          .dropna(subset=["language"])
          .reset_index(drop=True))

if data.empty:
    raise RuntimeError("No rows collected – please check folder paths/columns.")

from matplotlib.patches import Patch

colour_handles = [
    Patch(facecolor="green", edgecolor="green", label="Correct"),
    Patch(facecolor="red",   edgecolor="red",   label="Incorrect")
]

for lang in data["language"].unique():
    plt.figure(figsize=(14.5, 6))
    lang_df = data[data["language"] == lang]

    models = sorted(lang_df["model"].unique())
    print(models)
    xloc   = {m: (i * 3) + 1 for i, m in enumerate(models)}
    xtick_positions = list(xloc.values())
    for mod in models:
        sdf = lang_df[lang_df["model"] == mod]
        if sdf.empty:
            continue
        x = xloc[mod] + np.random.uniform(-0.1, 0.1, len(sdf))
        colours = sdf["correct"].map({True: "green", False: "red"})
        plt.scatter(x, sdf["prob"], c=colours, marker="o", alpha=0.5, s=30)

    xtick_labels = [m.replace("_", "\n") for m in models]
    plt.xticks(xtick_positions, xtick_labels, rotation=0, ha="center")
    plt.title(f"Predicted-class probability – {lang.capitalize()}")
    plt.xlabel("Model (with Dataset origin)")
    plt.ylabel("Probability")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    plt.legend(handles=colour_handles,
               bbox_to_anchor=(1.02, 1), loc="upper left",
               borderaxespad=0.0, title="Prediction")

    outfile = os.path.join(OUT_DIR, f"{lang}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")



