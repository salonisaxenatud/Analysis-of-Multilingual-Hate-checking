import pandas as pd
import os
import numpy as np

src_path = "Final Results/Bloomz_finetuned/Hindi_dataset.csv"  
df = pd.read_csv(src_path)

df.columns = df.columns.str.strip().str.lower()

out_cols = [
    "polish", "english", "hindi", "label",
    "pred_english", "conf_english",
    "pred_polish", "conf_polish",
    "pred_hindi", "conf_hindi"
]

out_df = pd.DataFrame(columns=out_cols)

out_df[["polish", "english", "hindi"]] = df[["polish", "english", "hindi"]]
out_df["label"] = df["label"]

for lang in ["english", "polish", "hindi"]:
    mask = df[lang].notna() & df[lang].astype(str).str.strip().ne("")
    out_df.loc[mask, f"pred_{lang}"] = df.loc[mask, "pred"]
    out_df.loc[mask, f"conf_{lang}"] = df.loc[mask, "conf"]

dst_path = "Final Results/Bloomz_finetuned/Hindi_dataset_reformatted.csv"
out_df.to_csv(dst_path, index=False)

dst_path

