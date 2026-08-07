import torch
import nibabel as nib
from pathlib import Path
import pandas as pd

class DBB(torch.utils.data.Dataset):
	@staticmethod
	def get_scans_filepath(data_folder_path: str) -> dict:
		"""
		Returns a dictionary, indexed using each subject id, containing the relative (to the dataset folder path)
		filepaths to the nifti scans
		 
		:param data_folder_path:
		:return:
		"""
		folder_path = Path(f"{data_folder_path}/proj-60a14ca503bcad0ad27cada9")

		filepaths_dict = {}
		sorted_dirs = sorted(p for p in folder_path.iterdir() if p.is_dir() and p.name != "bids")
		for patient_dir in sorted_dirs:
			patient_id = patient_dir.name[-4:]  # keeping only the 4 digits numerical identifier of the subject

			# initializing patient info
			filepaths_dict[patient_id] = {
				"T1-W FILEPATH": None,
				"SEGMENTATION FILEPATH": None,
				"LABELS FILEPATH": None,
				"BRAIN MASK FILEPATH": None,
				"RAW BRAIN MASK FILEPATH": None,
				"AFFINE MATRIX FILEPATH": None
			}

			for p in patient_dir.iterdir():
				if "anat-t1w" in p.name:
					# casting filepath object to str and removing the data_folder path prefix to save up some memory
					filepaths_dict[patient_id]["T1-W FILEPATH"] = str(p / "t1.nii.gz").replace(f"{data_folder_path}/", "")
					continue

				if "parcellation-volume" in p.name:
					filepaths_dict[patient_id]["SEGMENTATION FILEPATH"] = str(p / "parc.nii.gz").replace(f"{data_folder_path}/", "")
					filepaths_dict[patient_id]["LABELS FILEPATH"] = str(p / "label.json").replace(f"{data_folder_path}/", "")
					continue

				if "mask.id" in p.name:
					filepaths_dict[patient_id]["BRAIN MASK FILEPATH"] = str(p / "mask.nii.gz").replace(
						f"{data_folder_path}/", "")
					continue

				if "mask.tag-raw" in p.name:
					filepaths_dict[patient_id]["RAW BRAIN MASK FILEPATH"] = str(p / "mask.nii.gz").replace(
						f"{data_folder_path}/", "")
					continue

				if "transform-nifti" in p.name:
					filepaths_dict[patient_id]["AFFINE MATRIX FILEPATH"] = str(p / "affine.txt").replace(
						f"{data_folder_path}/", "")
					continue

		return filepaths_dict

	@staticmethod
	def get_patients_metadata(data_folder_path: str) -> pd.DataFrame:
		return pd.read_csv(f"{data_folder_path}/metadata.csv", dtype={"SUBJECT": str})

	def __init__(self):
		pass

	def __len__(self):
		return 0

	def __getitem__(self, idx):
		pass