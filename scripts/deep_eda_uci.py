"""Generate deep UCI Beijing diagnostics used by notebook 01."""

from benchmark_eda.deep_eda import run_deep_eda


if __name__ == "__main__":
    result = run_deep_eda()
    print("saved=artifacts/deep_eda_uci")
    print(result["regional_functional_graph"])
