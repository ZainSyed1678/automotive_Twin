"""
Dataset generation and export script for M2.
Generates 200 cases (140 dev, 60 eval) and exports ground-truth and agent datasets.
"""

from collections import Counter
from casegen import SyntheticCaseGenerator, split_dataset, export_datasets, compute_observation_fingerprint


def main():
    print("=" * 70)
    print("M2: Generating 200 Synthetic Diagnostic Cases (seed=42)...")
    print("=" * 70)

    generator = SyntheticCaseGenerator(seed=42)
    cases = generator.generate_dataset(num_cases=200)

    print(f"Generated {len(cases)} cases across 12 faults.")

    # Split into 140 dev / 60 eval with stratification and anti-leakage checks
    dev_cases, eval_cases = split_dataset(cases, dev_target=140, eval_target=60, seed=42)

    dev_fps = {compute_observation_fingerprint(c.observations) for c in dev_cases}
    eval_fps = {compute_observation_fingerprint(c.observations) for c in eval_cases}
    collisions = dev_fps.intersection(eval_fps)

    print(f"Development Set : {len(dev_cases)} cases ({len(dev_fps)} unique observation fingerprints)")
    print(f"Evaluation Set  : {len(eval_cases)} cases ({len(eval_fps)} unique observation fingerprints)")
    print(f"Fingerprint Intersection: {len(collisions)} collisions (PASSED)")

    if collisions:
        raise RuntimeError(f"FATAL: Data leakage detected! {len(collisions)} duplicate fingerprints!")

    # Export datasets
    exported_paths = export_datasets(dev_cases, eval_cases, base_dir="data")
    print("\nExported Files:")
    for key, path in exported_paths.items():
        print(f"  - {key:20s}: {path}")

    # Summary tables
    print("\n" + "-" * 70)
    print("PER-DIFFICULTY CASE COUNTS:")
    print("-" * 70)
    diff_dev = Counter(c.difficulty.value for c in dev_cases)
    diff_eval = Counter(c.difficulty.value for c in eval_cases)
    all_diffs = sorted(list(set(list(diff_dev.keys()) + list(diff_eval.keys()))))
    print(f"{'Difficulty':<25} | {'Dev (140)':<10} | {'Eval (60)':<10} | {'Total (200)':<10}")
    print("-" * 70)
    for d in all_diffs:
        print(f"{d:<25} | {diff_dev[d]:<10} | {diff_eval[d]:<10} | {diff_dev[d] + diff_eval[d]:<10}")

    print("\n" + "-" * 70)
    print("PER-FAULT CASE COUNTS:")
    print("-" * 70)
    fault_dev = Counter(c.true_fault_id for c in dev_cases)
    fault_eval = Counter(c.true_fault_id for c in eval_cases)
    all_faults = sorted(list(set(list(fault_dev.keys()) + list(fault_eval.keys()))))
    print(f"{'Fault ID':<35} | {'Dev':<6} | {'Eval':<6} | {'Total':<6}")
    print("-" * 70)
    for f in all_faults:
        print(f"{f:<35} | {fault_dev[f]:<6} | {fault_eval[f]:<6} | {fault_dev[f] + fault_eval[f]:<6}")

    print("=" * 70)
    print("M2 Dataset Generation Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
