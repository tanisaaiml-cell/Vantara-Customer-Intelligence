"""Run python -m scripts.export_processed to create readable feature CSVs."""

from src.data.exports import export_csv_files

if __name__ == "__main__":
    print(export_csv_files())
