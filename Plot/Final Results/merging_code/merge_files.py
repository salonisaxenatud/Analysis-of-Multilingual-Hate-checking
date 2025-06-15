import os
import pandas as pd
import numpy as np

ROOT         = r"D:\Delft\Uni\Mod 4\NLP for Society\Project\Data\Results\Latest\Final Results\Models"
DATASETS     = ["Polish_dataset.csv", "Hindi_dataset.csv"]
SKIP_MODELS  = {"BLOOMZ_FINETUNED"}
LANGS        = ("polish", "english", "hindi")

YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS  = {"no",  "n", "false", "0"}

def norm_pred(val):
    if pd.isna(val):
        return np.nan
    try:                                    # numeric
        num = float(val)
        return 0 if num == 0 else 1
    except ValueError:                      # text
        s = str(val).strip().lower()
        if s in YES_WORDS:
            return 1
        if s in NO_WORDS:
            return 0
    return np.nan                           # fallback

merged = {ds: None for ds in DATASETS}

for model in os.listdir(ROOT):
    if model.upper() in SKIP_MODELS:
        continue
    mdir = os.path.join(ROOT, model)
    if not os.path.isdir(mdir):
        continue

    for ds in DATASETS:
        src = os.path.join(mdir, ds)
        if not os.path.exists(src):
            continue

        df = pd.read_csv(src)
        df.columns = df.columns.str.strip().str.lower()

        ren = {}
        for lang in LANGS:
            old = f"pred_{lang}"
            if old in df.columns:
                new = f"{model.upper()}_{old}"
                ren[old] = new
                # normalise predictions to 0/1
                df[new] = df[old].apply(norm_pred)

        if not ren:
            continue

        id_col = next(c for c in df.columns if c.lower() == "id")
        keep   = [id_col] + list(ren.values())
        df     = df[keep]

        if merged[ds] is None:  
            merged[ds] = df
        else:
            merged[ds] = pd.merge(
                merged[ds], df, on=id_col,
                how="outer", sort=False, validate="one_to_one"
            )

for ds, table in merged.items():
    if table is not None:
        out = f"Merged_data\merged_{ds}"
        table.to_csv(out, index=False)
        print(f"written → {out}")
