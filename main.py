from datasets.wmh import WMH
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wmh_dataset_root_path")
parser.add_argument("--output_dir_path")
parser.add_argument("--t1w_reference_template_filepath")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="gpu")
args = parser.parse_args()

wmh_dataset_root_path = args.wmh_dataset_root_path
output_dir_path = args.output_dir_path
t1w_reference_template_filepath = args.t1w_reference_template_filepath
device = args.device

WMH.preprocess_dataset(
	output_dir_filepath=output_dir_path,
	dataset_root_path=wmh_dataset_root_path,
	template_t1w_filepath=t1w_reference_template_filepath,
	device=device
)