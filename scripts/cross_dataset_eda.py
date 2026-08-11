from pathlib import Path

from benchmark_eda.cross_dataset_eda import run_cross_dataset_eda


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_cross_dataset_eda(root, root / "artifacts/cross_dataset_eda")
    print(f"datasets={len(result['datasets'])}")
    print("saved=artifacts/cross_dataset_eda")
