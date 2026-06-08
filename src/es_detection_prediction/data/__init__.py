"""Data subpackage: CHB-MIT loading, segmentation, filtering, and (later) windowing.

Currently exposes:

- :func:`load_seizure_annotations` — parse ``chbXX-summary.txt`` files into a
  nested ``{subject: {record: {no_of_seizures, seizures}}}`` mapping. Useful
  on its own for sanity checks or dataset statistics without touching the
  EDFs.
- :func:`load_records` — load every ``.edf`` into MNE ``Raw`` objects,
  sort each subject's records chronologically by EDF ``meas_date``
  (filename order is unreliable for some subjects), harmonise to the
  22-channel bipolar montage from ``config.yaml``, attach seizure
  annotations, and place every record on the subject's continuous
  wall-clock timeline so each one knows its absolute start/end relative to
  the subject's first recording (needed for the n-hour ictal buffer).
- :func:`find_eligible_records` — given the loader output, partition records
  per subject into an ictal pool (records that contain seizures) and an
  interictal pool (no-seizure records whose surviving sub-interval, after
  applying the n-hour ictal buffer on each side of every seizure on the subject's
  timeline, is long enough for at least one ``segment_length`` segment). Interictal
  pool entries carry record-relative ``interictal_period_starts_at`` /
  ``interictal_period_ends_at`` bounds for direct use with MNE Raw.
- :func:`filter_and_save_records` / :func:`band_pass_filtering` — band-pass
  filter every record in a segmentation pool **once** and cache to disk
  as FIF so the downstream ``Dataset`` never re-filters per window. Cache
  directory is bandpass-versioned (``bp_{l_freq}-{h_freq}_{h_trans_bandwidth}/``) 
  and writes are idempotent (existing FIFs are re-used), so interrupted runs resume
  for free. After the call returns, each pool record carries a
  ``filtered_path`` key pointing at its cached FIF. The first function is
  the per-record worker; the second is the per-pool orchestrator.
- :func:`summarize_records` / :func:`print_record_summary` — dataset-level
  stats over the loader output, split into ``valid`` / ``invalid`` buckets
  (channel-set match) with combined + valid-only + invalid-only views.
  The first is pure compute; the second is the human-readable formatter.
  Useful as a sanity check after loading and for cross-referencing the
  paper's "total available" / "observed" headline figures.
- :func:`summarize_eligible_records` / :func:`print_eligible_records_summary`
  — counterparts to the loader summaries but over the segmentation pools
  returned by :func:`find_eligible_records`. Includes fittable-segment
  counts and the list of subjects excluded from the interictal pool by
  the ictal buffer.
- :func:`print_excluded_subject_details` — diagnostic dump for those
  excluded subjects (per-record seizure counts, durations and record
  length), so you can see *why* the buffer rejected them.
"""

from .dataloader import (
  load_seizure_annotations,
  load_records,
)
from .segmentation import (
  find_eligible_records
)
from .preprocessing import (
  filter_and_save_records,
  band_pass_filtering,
)
from .summary import (
  summarize_records,
  print_record_summary,
  summarize_eligible_records,
  print_eligible_records_summary,
  print_excluded_subject_details,
)

__all__ = [
  # from dataloader
  "load_seizure_annotations",
  "load_records",
  # from segmentation
  "find_eligible_records",
  # from preprocessing
  "filter_and_save_records",
  "band_pass_filtering",
  # from summary
  "summarize_records",
  "print_record_summary",
  "summarize_eligible_records",
  "print_eligible_records_summary",
  "print_excluded_subject_details",
]
