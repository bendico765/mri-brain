from typing import Optional

import torch
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import os


class WMH(torch.utils.data.Dataset):

	@staticmethod
	def get_metadata(dataset_root_path: str) -> pd.DataFrame:
		entries = []

		for set in ["training", "test"]:
			for center in ["Amsterdam", "Singapore", "Utrecht"]:
				path = Path(f"{dataset_root_path}/wmh_data/{set}/{center}")

				if center == "Amsterdam":
					for scanner_dir in path.iterdir():
						scanner = scanner_dir.name
						for patient_dir in sorted(p for p in scanner_dir.iterdir() if p.is_dir()):
							patient_name = patient_dir.name

							t1_pre_nifti_filepath = str(patient_dir / "pre" / "T1.nii")
							flair_pre_nifti_filepath = str(patient_dir / "pre" / "FLAIR.nii")
							segmentation_nifti_filepath = str(patient_dir / "wmh.nii")

							t1_pre_nifti_filepath = t1_pre_nifti_filepath.replace(f"{dataset_root_path}/", "")
							flair_pre_nifti_filepath = flair_pre_nifti_filepath.replace(f"{dataset_root_path}/", "")
							segmentation_nifti_filepath = segmentation_nifti_filepath.replace(f"{dataset_root_path}/", "")

							entries.append(
								(set, center, scanner, patient_name, t1_pre_nifti_filepath, flair_pre_nifti_filepath, segmentation_nifti_filepath)
							)
				else:
					scanner = "GE3T"
					for patient_dir in sorted(p for p in path.iterdir() if p.is_dir()):
						patient_name = patient_dir.name

						t1_pre_nifti_filepath = str(patient_dir / "pre" / "T1.nii")
						flair_pre_nifti_filepath = str(patient_dir / "pre" / "FLAIR.nii")
						segmentation_nifti_filepath = str(patient_dir / "wmh.nii")

						t1_pre_nifti_filepath = t1_pre_nifti_filepath.replace(f"{dataset_root_path}/", "")
						flair_pre_nifti_filepath = flair_pre_nifti_filepath.replace(f"{dataset_root_path}/", "")
						segmentation_nifti_filepath = segmentation_nifti_filepath.replace(f"{dataset_root_path}/", "")

						entries.append(
							(set, center, scanner, patient_name, t1_pre_nifti_filepath, flair_pre_nifti_filepath,
							 segmentation_nifti_filepath)
						)

		return pd.DataFrame(entries, columns=[
			"SET",
			"CENTER",
			"SCANNER",
			"PATIENT",
			"T1W FILEPATH",
			"FLAIR FILEPATH",
	        "SEGMENTATION FILEPATH"
		])

	@staticmethod
	def preprocess_patient(
			input_t1w_filepath: str,
			input_flair_filepath: str,
			input_segmentation_filepath: str,
			template_t1w_filepath: str,
			output_dir_filepath: str,
			device="cpu"
	):
		"""
		Creates in the output directory the files "T1w.nii.gz", "FLAIR.nii.gz", "ROI.nii.gz" and an additional folder
		called "additional_files" with the matrices produced.
		This function assumes that t1w and flair are already coregistred (which is true for the WHM dataset)

		:param input_t1w_filepath: Path to the t1w nifti file
		:param input_flair_filepath: Path to the flair nifti file
		:param input_segmentation_filepath: Path to the segmentation mask nifti file
		:param template_t1w_filepath: Path to the t1w template used as reference for registration.
		:param output_dir_filepath: Path to the output directory (if does not exist, the function creates it)
		:param device: Device to be used from skull stripping, can be "cpu" or "gpu"
		"""
		Path(output_dir_filepath).mkdir(parents=True, exist_ok=True)
		Path(f"{output_dir_filepath}/additional_files").mkdir(parents=True, exist_ok=True)

		# skull stripping on t1w
		with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
			os.system(f"hd-bet -i {input_t1w_filepath} -o {tmp.name} -device {device} --disable_tta --save_bet_mask")

			# loading skull-stripped t1w
			t1w_brain_nib = nib.load(tmp.name)
			t1w_brain_affine = t1w_brain_nib.affine
			t1w_brain_array = t1w_brain_nib.get_fdata()

			# loading the brain mask
			path = tmp.name.split(".")[0]  # removing extension
			brain_mask_nib = nib.load(f"{path}_bet.nii.gz")
			brain_mask_array = brain_mask_nib.get_fdata()

			nib.save(brain_mask_nib, f"{output_dir_filepath}/additional_files/t1w_brain_mask.nii.gz")

		# apply brain mask on flair and segmentation masks
		flair_nib = nib.load(input_flair_filepath)
		flair_affine = flair_nib.affine
		flair_array = flair_nib.get_fdata()
		flair_brain_array = np.where(brain_mask_array == 1, flair_array, 0)

		segmentation_nib = nib.load(input_segmentation_filepath)
		segmentation_affine = segmentation_nib.affine
		segmentation_array = segmentation_nib.get_fdata()
		segmentation_brain_array = np.where(brain_mask_array == 1, segmentation_array, 0)

		# register T1->MNI
		with tempfile.TemporaryDirectory() as tmpdir:
			# saving the T1w skull stripped
			nib.save(nib.Nifti1Image(t1w_brain_array, t1w_brain_affine), f"{tmpdir}/t1w_brain.nii.gz")

			print("Perform registration")
			os.system(f"""
	        antsRegistration \
	        --dimensionality 3 \
	        --float 0 \
	        --output ["{output_dir_filepath}/additional_files/T1_to_MNI_","{tmpdir}/T1_MNI.nii.gz","{output_dir_filepath}/additional_files/T1_from_MNI.nii.gz"] \
	        --interpolation Linear \
	        --winsorize-image-intensities [0.005,0.995] \
	        --use-histogram-matching 0 \
	        --initial-moving-transform ["{template_t1w_filepath}","{tmpdir}/t1w_brain.nii.gz",1] \
	        --transform Rigid[0.1] \
	        --metric MI["{template_t1w_filepath}","{tmpdir}/t1w_brain.nii.gz",1,32,Regular,0.25] \
	        --convergence [1000x500x250x100,1e-6,10] \
	        --shrink-factors 8x4x2x1 \
	        --smoothing-sigmas 3x2x1x0vox \
	        --transform Affine[0.1] \
	        --metric MI["{template_t1w_filepath}","{tmpdir}/t1w_brain.nii.gz",1,32,Regular,0.25] \
	        --convergence [1000x500x250x100,1e-6,10] \
	        --shrink-factors 8x4x2x1 \
	        --smoothing-sigmas 3x2x1x0vox \
	        --transform SyN[0.1,3,0] \
	        --metric CC["{template_t1w_filepath}","{tmpdir}/t1w_brain.nii.gz",1,4] \
	        --convergence [100x70x50x20,1e-6,10] \
	        --shrink-factors 8x4x2x1 \
	        --smoothing-sigmas 3x2x1x0vox
	        """)

			# reorient to ras
			output_t1w_nib = nib.load(f"{tmpdir}/T1_MNI.nii.gz")
			output_t1w_nib = nib.as_closest_canonical(output_t1w_nib)
			nib.save(output_t1w_nib, f"{output_dir_filepath}/T1w.nii.gz")

			# register FLAIR-> MNI
			nib.save(nib.Nifti1Image(flair_brain_array, flair_affine), f"{tmpdir}/FLAIR_T1.nii.gz")
			os.system(f"""
	        antsApplyTransforms \
	        --dimensionality 3 \
	        --input "{tmpdir}/FLAIR_T1.nii.gz" \
	        --reference-image "{template_t1w_filepath}" \
	        --output "{tmpdir}/FLAIR_MNI.nii.gz" \
	        --interpolation Linear \
	        --transform "{output_dir_filepath}/additional_files/T1_to_MNI_1Warp.nii.gz" \
	        --transform "{output_dir_filepath}/additional_files/T1_to_MNI_0GenericAffine.mat"
	        """)

			output_flair_nib = nib.load(f"{tmpdir}/FLAIR_MNI.nii.gz")
			output_flair_nib = nib.as_closest_canonical(output_flair_nib)
			nib.save(output_flair_nib, f"{output_dir_filepath}/FLAIR.nii.gz")

			# register ROI->MNI
			nib.save(nib.Nifti1Image(segmentation_brain_array, segmentation_affine), f"{tmpdir}/ROI.nii.gz")
			os.system(f"""
	        antsApplyTransforms \
	        --dimensionality 3 \
	        --input "{tmpdir}/ROI.nii.gz" \
	        --reference-image "{template_t1w_filepath}" \
	        --output "{tmpdir}/ROI_MNI.nii.gz" \
	        --interpolation NearestNeighbor \
	        --transform "{output_dir_filepath}/additional_files/T1_to_MNI_1Warp.nii.gz" \
	        --transform "{output_dir_filepath}/additional_files/T1_to_MNI_0GenericAffine.mat"
	        """)

			output_roi_nib = nib.load(f"{tmpdir}/ROI_MNI.nii.gz")
			output_roi_nib = nib.as_closest_canonical(output_roi_nib)
			nib.save(output_roi_nib, f"{output_dir_filepath}/ROI.nii.gz")

	@staticmethod
	def preprocess_dataset(
			output_dir_filepath: str,
			dataset_root_path: str,
			template_t1w_filepath: str
	):
		df = WMH.get_metadata(dataset_root_path)

		for _, row in tqdm(df.iterrows(), total=len(df)):
			patient_id = row["PATIENT"]

			t1w_filepath = row["T1W FILEPATH"]
			flair_filepath = row["FLAIR FILEPATH"]
			segmentation_filepath = row["SEGMENTATION FILEPATH"]

			t1w_filepath = f"{dataset_root_path}/{t1w_filepath}"
			flair_filepath = f"{dataset_root_path}/{flair_filepath}"
			segmentation_filepath = f"{dataset_root_path}/{segmentation_filepath}"

			# create folder for patient data
			Path(f'{output_dir_filepath}/wmh_data/{patient_id}').mkdir(parents=True, exist_ok=True)

			WMH.preprocess(
				t1w_filepath,
				flair_filepath,
				segmentation_filepath,
				template_t1w_filepath,
				f'{output_dir_filepath}/{patient_id}'
			)

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