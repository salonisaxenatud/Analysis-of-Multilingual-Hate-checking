# """
# Re-format a single *_dataset.csv so that downstream code can treat every file
# the same, regardless of the language used for ‘yes’ / ‘no’.
#
# INPUT  (example) :  D:\...\Polish_dataset.csv
# OUTPUT (created) :  D:\...\Polish_dataset_formatted.csv
#                     └── keeps every original column
#                     └── adds: gt, pred, correct, predicted_logit
# """
#
# import os
# import numpy as np
# import pandas as pd
#
# # --------------------------------------------------------------------------
# # 1 configuration –----- change ONLY the next line
# # --------------------------------------------------------------------------
# csv_path = r"Final Results\Bloomz_finetuned\Hindi_dataset.csv"
#
# # ----------------------------------------------------------------------------
# # 2 helper functions
# # ----------------------------------------------------------------------------
# YES = {'yes', 'tak', 'हाँ'}
# NO  = {'no',  'nie', 'नहीं'}
#
# def to_bool(label: str) -> int:
#     """Map every spelling of ‘yes’/‘no’ that occurs in the data to 1 / 0."""
#     label = str(label).strip().lower()
#     if label in YES:
#         return 1
#     if label in NO:
#         return 0
#     raise ValueError(f"Unexpected label value: {label!r}")
#
# def pick_logit(df: pd.DataFrame) -> pd.Series:
#     """
#     For every row return the logit (or probability) that corresponds to
#     *the class the model actually predicted*.
#     """
#     if {'logit_yes', 'logit_no'}.issubset(df.columns):
#         yes_col, no_col = 'logit_yes', 'logit_no'
#     elif {'prob_yes', 'prob_no'}.issubset(df.columns):          # fallback
#         yes_col, no_col = 'prob_yes',  'prob_no'
#     else:
#         raise KeyError("No matching logit/probability columns found.")
#
#     return np.where(df['pred'] == 1, df[yes_col], df[no_col])
#
# # ----------------------------------------------------------------------------
# # 3 read, harmonise, save
# # ----------------------------------------------------------------------------
# df = pd.read_csv(csv_path)
#
# df['gt']   = df['ground_truth'].apply(to_bool)
# df['pred'] = df['prediction'].apply(to_bool)
# df['correct'] = df['gt'] == df['pred']
# df['predicted_logit'] = pick_logit(df)
#
# new_path = csv_path.replace('_dataset.csv', '_dataset_formatted.csv')
# df.to_csv(new_path, index=False)
#
# print(f"Formatted file written to:\n  {new_path}")
# import os
# import pandas as pd
# import numpy as np
#
# # --------------------------------------------------------------------------
# # 1  input ­– point to the *formatted* file you created earlier
# # --------------------------------------------------------------------------
# src = r"Final Results\Bloomz_finetuned\Hindi_dataset_formatted.csv"          # adjust the path
#
# # --------------------------------------------------------------------------
# # 2  load and pivot to the requested wide layout
# # --------------------------------------------------------------------------
# df = pd.read_csv(src)
#
# # one column per language, containing the prompt only for rows of that language
# for lang in ('english', 'polish', 'hindi'):
#     df[lang] = np.where(df['language'] == lang, df['prompt'], pd.NA)
#
# wide = df[['english', 'polish', 'hindi', 'gt', 'pred', 'correct', 'predicted_logit']]
#
# # --------------------------------------------------------------------------
# # 3  write a *new* CSV next to the source file
# # --------------------------------------------------------------------------
# dst = src.replace('_formatted.csv', '_wide.csv')
# wide.to_csv(dst, index=False)
#
# print(f"Wide-format file written to:\n  {dst}")

import pandas as pd
import os
import numpy as np

# Path to the uploaded CSV (adjust if necessary)
src_path = "Final Results/Bloomz_finetuned/Hindi_dataset.csv"  # fallback name

# If the exact uploaded path is different, try the pointer provided
# if not os.path.exists(src_path):
#     src_path = "/mnt/data/file-6cSPVKzWGtj75vRZ6ebXwG"

df = pd.read_csv(src_path)

# Ensure consistent column names
df.columns = df.columns.str.strip().str.lower()

# Prepare new columns
out_cols = [
    "polish", "english", "hindi", "label",
    "pred_english", "conf_english",
    "pred_polish", "conf_polish",
    "pred_hindi", "conf_hindi"
]

# Initialize output DataFrame
out_df = pd.DataFrame(columns=out_cols)

# Copy prompt columns and label
out_df[["polish", "english", "hindi"]] = df[["polish", "english", "hindi"]]
out_df["label"] = df["label"]

# Fill language-specific pred/conf columns
for lang in ["english", "polish", "hindi"]:
    mask = df[lang].notna() & df[lang].astype(str).str.strip().ne("")
    out_df.loc[mask, f"pred_{lang}"] = df.loc[mask, "pred"]
    out_df.loc[mask, f"conf_{lang}"] = df.loc[mask, "conf"]

# Save reformatted CSV
dst_path = "Final Results/Bloomz_finetuned/Hindi_dataset_reformatted.csv"
out_df.to_csv(dst_path, index=False)

dst_path

