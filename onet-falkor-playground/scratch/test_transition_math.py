import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "backend"))

from main import fetch_profile, fetch_occupation, cosine_similarity, compute_gap

src_code = "49-9071.00" # Maintenance and Repair Workers, General
tgt_code = "33-2011.00" # Firefighters

src_profile = fetch_profile(src_code)
tgt_profile = fetch_profile(tgt_code)

sim = cosine_similarity(src_profile, tgt_profile)
missing, deficient, transferable = compute_gap(src_profile, tgt_profile)

print(f"Source: {fetch_occupation(src_code).title} (Competencies: {len(src_profile)})")
print(f"Target: {fetch_occupation(tgt_code).title} (Competencies: {len(tgt_profile)})")
print(f"Cosine Similarity: {sim * 100:.2f}%")
print(f"Gaps (Threshold 3.0):")
print(f"  Missing: {len(missing)}")
print(f"  Deficient: {len(deficient)}")
print(f"  Transferable: {len(transferable)}")

# Test with a higher deficiency threshold
# Let's inspect deltas of deficient items
deltas = [d.delta for d in deficient]
print(f"\nDeficient deltas: min={min(deltas):.2f}, max={max(deltas):.2f}, avg={sum(deltas)/len(deltas):.2f}")

# Count items with delta >= 8.0
severe_deficient = [d for d in deficient if d.delta >= 8.0]
mild_deficient = [d for d in deficient if d.delta < 8.0]
print(f"\nIf Deficiency Threshold is raised to 8.0:")
print(f"  Severe Deficient (Gaps >= 8.0): {len(severe_deficient)}")
print(f"  Mild Deficient (transfered to Transferable): {len(mild_deficient)}")
