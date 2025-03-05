import torch
from safetensors.torch import load_file
import os

safetensor_dir = "model/state-spaces-mamba-2.8b-hf/" 
safetensor_files = [
    'model-00001-of-00003.safetensors',
    'model-00002-of-00003.safetensors',
    'model-00003-of-00003.safetensors'
]

safetensor_paths = [os.path.join(safetensor_dir, file) for file in safetensor_files]

output_dir = "model/state-spaces-mamba-2.8b-hf/"
os.makedirs(output_dir, exist_ok=True) 

for idx, safetensor_path in enumerate(safetensor_paths, 1):
    part_state_dict = load_file(safetensor_path)
    
    bin_filename = os.path.join(output_dir, f'pytorch_model-{idx:05d}-of-00003.bin')
    torch.save(part_state_dict, bin_filename)
    
    print(f"Saved {bin_filename}")
