from typing import Optional

import torch
import nibabel as nib
from pathlib import Path


class WMH(torch.utils.data.Dataset):
	"""
	Some notes

	- Different resolutions for different images
	- different number of slices within the same center (e.g. singapore training)
	"""

	@staticmethod
	def get_scans_filepath(dir_path: str):
		"""

		:param dir_path: Path to the center and scanner folder data
		:return:
		"""
		path = Path(dir_path)
		t1_filepaths = []
		flair_filepaths = []
		segmentation_filepaths = []

		for patient_dir in sorted(p for p in path.iterdir() if p.is_dir()):
			t1_pre_nifti_filepath = patient_dir / "pre" / "T1.nii"
			flair_pre_nifti_filepath = patient_dir / "pre" / "FLAIR.nii"
			segmentation_nifti_filepath = patient_dir / "wmh.nii"

			t1_filepaths.append(str(t1_pre_nifti_filepath))
			flair_filepaths.append(str(flair_pre_nifti_filepath))
			segmentation_filepaths.append(str(segmentation_nifti_filepath))

		return t1_filepaths, flair_filepaths, segmentation_filepaths

	def __init__(self, dataset_root_path: str, data_split: Optional[str] = None, center: Optional[str] = None, transform=None):
		"""

		:param dataset_root_path:
		:param data_split: whether to use the training or test set. Possible values are 'training' or 'test', specify None to use both
		:param center: which center data to use, possible values are 'Amsterdam', 'Singapore' and 'Utrecht', or all of them by using None
		:param transform:
		"""
		if data_split is not None and (data_split != "training" and data_split != "test"):
			raise ValueError("The data split value must be None, 'training' or 'test'.")

		if center is not None and (center not in ["Amsterdam", "Singapore", "Utrecht"]):
			raise ValueError("Center must be None, or one of the following values ['Amsterdam', 'Singapore', 'Utrecht]")

		self.t1_filepaths = []
		self.flair_filepaths = []
		self.segmentation_filepaths = []
		self.transform = transform

		if data_split is None or data_split == "training":
			# load up Amsterdam data
			if center is None or center == "Amsterdam":
				t1_filepaths, flair_filepaths, segmentation_filepaths = WMH.get_scans_filepath(f"{dataset_root_path}/wmh_data/training/Amsterdam/GE3T")
				self.t1_filepaths.extend(t1_filepaths)
				self.flair_filepaths.extend(flair_filepaths)
				self.segmentation_filepaths.extend(segmentation_filepaths)

			# load up Singapore data
			if center is None or center == "Singapore":
				t1_filepaths, flair_filepaths, segmentation_filepaths = WMH.get_scans_filepath(
					f"{dataset_root_path}/wmh_data/training/Singapore")
				self.t1_filepaths.extend(t1_filepaths)
				self.flair_filepaths.extend(flair_filepaths)
				self.segmentation_filepaths.extend(segmentation_filepaths)

			# load up Utrecht data
			if center is None or center == "Utrecht":
				t1_filepaths, flair_filepaths, segmentation_filepaths = WMH.get_scans_filepath(
					f"{dataset_root_path}/wmh_data/training/Utrecht")
				self.t1_filepaths.extend(t1_filepaths)
				self.flair_filepaths.extend(flair_filepaths)
				self.segmentation_filepaths.extend(segmentation_filepaths)

		if data_split is None or data_split == "test":
			# load up Amsterdam data
			if center is None or center == "Amsterdam":
				for scanner in ["GE3T", "GE1T5", "Philips_VU .PETMR_01."]:
					t1_filepaths, flair_filepaths, segmentation_filepaths = WMH.get_scans_filepath(
						f"{dataset_root_path}/wmh_data/test/Amsterdam/{scanner}")
					self.t1_filepaths.extend(t1_filepaths)
					self.flair_filepaths.extend(flair_filepaths)
					self.segmentation_filepaths.extend(segmentation_filepaths)
			
			# load up Singapore data
			if center is None or center == "Singapore":
				t1_filepaths, flair_filepaths, segmentation_filepaths = WMH.get_scans_filepath(
					f"{dataset_root_path}/wmh_data/test/Singapore")
				self.t1_filepaths.extend(t1_filepaths)
				self.flair_filepaths.extend(flair_filepaths)
				self.segmentation_filepaths.extend(segmentation_filepaths)

			# load up Utrecht data
			if center is None or center == "Utrecht":
				t1_filepaths, flair_filepaths, segmentation_filepaths = WMH.get_scans_filepath(
					f"{dataset_root_path}/wmh_data/test/Utrecht")
				self.t1_filepaths.extend(t1_filepaths)
				self.flair_filepaths.extend(flair_filepaths)
				self.segmentation_filepaths.extend(segmentation_filepaths)

	def __len__(self):
		return len(self.t1_filepaths)

	def __getitem__(self, idx):
		if self.transform:
			pass

		t1_array = nib.load(self.t1_filepaths[idx]).get_fdata()
		flair_array = nib.load(self.flair_filepaths[idx]).get_fdata()
		segmentation_array = nib.load(self.segmentation_filepaths[idx]).get_fdata()

		return t1_array, flair_array, segmentation_array