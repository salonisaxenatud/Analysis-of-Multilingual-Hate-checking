import pandas as pd
import os
# ─── file locations ───────────────────────────────────────────────────────
small_path    = "Cultural data/Filtered_output/Hindi/english.csv"     # Bloomz-finetuned
big_path      = "Merged_data/merged_Polish_dataset.csv"      # master file
filtered_path = "Merged_data/Merged_data_Polish/polish.csv"
small_df = pd.read_csv(small_path)
big_df   = pd.read_csv(big_path)

# ── normalise the ID column name to “id” in both frames ───────────────────
def ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        return df
    # look for a nameless / blank header column
    anon = [c for c in df.columns if not str(c).strip()]
    if len(anon) == 1:
        return df.rename(columns={anon[0]: "id"})
    raise KeyError("No ‘id’ column found")
small_df = ensure_id(small_df)
big_df   = ensure_id(big_df)

# ── which language? (all rows in small_df share the same value) ───────────
if "language" not in small_df.columns:
    raise KeyError("'language' column missing in the small file")
language = str(small_df["language"].iloc[0]).strip().lower()

# ── filter rows in big_df whose id is present in small_df ────────────────
ids_to_keep = small_df["id"].unique()
filtered    = big_df[big_df["id"].isin(ids_to_keep)].copy()

# ── trim columns: keep just ‘id’ and those ending with _<language> ────────
lang_suffix = f"_{language}"
keep_cols   = ["id"] + [c for c in filtered.columns if c.endswith(lang_suffix)]
filtered    = filtered[keep_cols]

# ── attach Bloom-finetuned prediction and rename the column ───────────────
pred_col_name = f"BLOOMZ_FINETUNED_pred_{language}"
filtered = (filtered
            .merge(small_df[["id", "pred"]], on="id", how="left")
            .rename(columns={"pred": pred_col_name}))

# ── write the result ──────────────────────────────────────────────────────
os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
filtered.to_csv(filtered_path, index=False)
print(f"Wrote {len(filtered):,} rows to {filtered_path}")
