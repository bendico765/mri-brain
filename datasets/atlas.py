import torch
from pathlib import Path
import pandas as pd


class ATLAS(torch.utils.data.Dataset):
	@staticmethod
	def get_metadata(data_folder_path: str) -> pd.DataFrame:
		root_dir = Path(f"{data_folder_path}/ATLAS3_Training_Preprocessed")

		entries = []
		for cohort_dir in sorted([ p for p in root_dir.iterdir() if p.is_dir() ]):
			for subject_dir in sorted([p for p in cohort_dir.iterdir()]):
				if not subject_dir.is_dir():
					continue

				t1w_filepath, mask_filepath, metadata_filepath = None, None, None
				subject_scans_path = subject_dir / "ses-1" / "anat"
				for f in subject_scans_path.iterdir():
					if "T1w" in f.name:
						t1w_filepath = str(f).replace(f"{data_folder_path}/", "")
						continue
					if "lesion_mask" in f.name:
						mask_filepath = str(f).replace(f"{data_folder_path}/", "")
						continue
					if "metadata" in f.name:
						metadata_filepath = str(f).replace(f"{data_folder_path}/", "")
						continue

				df = pd.read_csv(f"{data_folder_path}/{metadata_filepath}")
				if len(df) != 0:
					session_id = df.iloc()[0]["SESSION_ID"]
					atlas2_dataset = df.iloc()[0]["ATLAS2_DATASET"]
					days_post_stroke = df.iloc()[0]["DAYS_POST_STROKE"]
					chronicity = df.iloc()[0]["CHRONICITY"]
					site = df.iloc()[0]["SITE"]
				else:
					session_id, atlas2_dataset, days_post_stroke, chronicity, site = None, None, None, None, None

				entries.append(
					(session_id, atlas2_dataset, days_post_stroke, chronicity, site, t1w_filepath, mask_filepath))

		return pd.DataFrame(
			entries,
			columns=[
				"SESSION ID",
				"DATASET",
				"DAYS POST STROKE",
				"CHRONICITY",
				"SITE",
				"T1W FILEPATH",
				"SEGMENTATION FILEPATH"
			])

	def __init__(self):
		super().__init__()

	def __len__(self):
		return 0

	def __getitem__(self, idx):
		pass