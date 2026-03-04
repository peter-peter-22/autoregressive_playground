import os
import zipfile


def zip_directory(source_dir, output_zip_path):
    if os.path.exists(output_zip_path):
        raise Exception(f"Output zip path already exists: {output_zip_path}")
    if not os.path.exists(source_dir):
        raise Exception(f"Source directory does not exist: {source_dir}")
    source_dir = os.path.abspath(source_dir)
    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = str(os.path.join(root, file))
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Zipped → {output_zip_path}")


if __name__ == "__main__":
    SOURCE_FOLDER = "output"
    ZIP_PATH = "archive.zip"

    os.remove(ZIP_PATH)
    zip_directory(SOURCE_FOLDER, ZIP_PATH)
