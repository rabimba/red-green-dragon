import os
import shutil
from glob import glob
from tqdm import tqdm
from datasets import Dataset

def main(as_dir, structured_dir, source_dir, arrow_path):
    os.makedirs(source_dir, exist_ok=True)

    dataset = Dataset.from_file(arrow_path)
    dataset = {d["blob_id"] : os.path.join(structured_dir, d["repo_name"], os.path.basename(d["branch_name"]), d["path"].lstrip("/"))
               for d in dataset}

    assembly_files = glob(os.path.join(as_dir, "*.s"))
    blob_ids = [os.path.splitext(os.path.basename(file))[0] for file in assembly_files]
    for blob_id in tqdm(blob_ids):
        source_file = dataset[blob_id]
        shutil.copy2(source_file, os.path.join(source_dir, f"{blob_id}.cu"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--as-dir", type=str, required=True, help="Directory of assembly files")
    parser.add_argument("--structured-dir", type=str, required=True, help="Structured directory")
    parser.add_argument("--sources-dir", type=str, required=True, help="Output directory for the source files")
    parser.add_argument("--arrow-path", type=str, required=True, help="Path to local arrow file")

    args = parser.parse_args()

    main(args.as_dir, args.structured_dir, args.sources_dir, args.arrow_path)