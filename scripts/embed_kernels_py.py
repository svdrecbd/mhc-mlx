import os
import glob

KERNEL_DIR = "mhc_mlx/kernels"
OUTPUT_FILE = "mhc_mlx/kernels_embedded.py"

def main():
    print(f"Embedding kernels from {KERNEL_DIR} to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as out:
        out.write('"""Embedded Metal kernels."""\n\n')
        
        metal_files = glob.glob(os.path.join(KERNEL_DIR, "*.metal"))
        for metal_file in metal_files:
            filename = os.path.basename(metal_file)
            var_name = filename.replace(".", "_").upper()
            
            with open(metal_file, "r") as f:
                content = f.read()
                
            out.write(f'{var_name} = r"""\n{content}\n"""\n\n')

if __name__ == "__main__":
    main()

