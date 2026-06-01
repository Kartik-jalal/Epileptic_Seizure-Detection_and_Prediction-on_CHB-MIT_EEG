"""Data subpackage: CHB-MIT loading, segmentation, and (later) windowing.

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
"""

from .dataloader import (
  load_seizure_annotations,
  load_records,
)
from .segmentation import (
  find_eligible_records
)

__all__ = [
  # from dataloader
  "load_seizure_annotations",
  "load_records",
  # from segmentation
  "find_eligible_records",
]
