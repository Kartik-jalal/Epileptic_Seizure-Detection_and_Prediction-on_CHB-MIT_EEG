"""Data subpackage: CHB-MIT loading and (later) windowing/labelling.

Currently exposes the two CHB-MIT ingest helpers from ``dataloader``:

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
  the subject's first recording (needed for the 4-hour interictal buffer).
  The standard entry point for the rest of the pipeline.
"""

from .dataloader import (
    load_seizure_annotations,
    load_records,
)

__all__ = [
    "load_seizure_annotations",
    "load_records",
]
