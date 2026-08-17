"""PyTorch ``Dataset`` wrappers over the cached, filtered CHB-MIT records.

Two variants live here, one per evaluation mode. Both wrap **one** cached
``.fif`` on disk (produced by :func:`band_pass_filtering`) and both slice
windows out of it lazily — they differ in *which* windows they expose and
*what* they return alongside the EEG:

- :class:`RecordWindowDataset` — training and curated segment-level test.
  Windows come from the eligibility bounds
  (:func:`find_eligible_records` → :func:`get_record_windows`), so only
  unambiguously-ictal or buffer-clean-interictal windows appear. Serves
  ``(X, y)`` with a constant class label per record.
  :func:`build_record_datasets` builds one per pool entry.
- :class:`ContinuousRecordDataset` — event-level test only. Windows cover
  the **whole** record with no eligibility filtering, at a finer stride,
  in chronological order. Serves ``(X, position)`` — no label, because a
  window's scoring category is computed later by the scoring loop.
  :func:`build_continuous_datasets` builds one per ``test_continuous``
  entry, and :func:`continuous_collate_fn` batches them.

:func:`build_split_datasets` composes both variants into a
DataLoader-ready fold (one ``ConcatDataset`` per split) straight from a
split-function dict — the usual one-call path from split to training.
Manual composition via :class:`torch.utils.data.ConcatDataset` works
too, when a fold needs custom filtering.

Lazy-open semantics (and why it matters for ``DataLoader`` workers)
-------------------------------------------------------------------
Both classes open the cached FIF on the first ``__getitem__`` call, NOT
in ``__init__``. This matters when training uses
``DataLoader(num_workers > 0)`` on Linux (fork is the default
multiprocessing context):

- Opening in ``__init__`` would attach an MNE ``Raw`` (with an open
  file descriptor) to the Dataset object in the **parent** process.
  ``fork()`` inherits that fd via the OS file table; multiple workers
  then share a single seek pointer and race when they
  ``lseek+read`` concurrently.
- Opening lazily in :meth:`_ensure_raw` means each forked worker
  independently opens its own ``Raw`` on first access. No shared file
  descriptors, no cross-worker races, no ``worker_init_fn`` plumbing.

The lazy pattern also bounds file-descriptor pressure: a
``ConcatDataset`` of 600 per-record Datasets does not hold 600 fds in
the parent process; each worker only opens what it actually touches in
its working set.
"""

import mne
import torch
from torch.utils.data import Dataset, ConcatDataset


from .windowing import get_record_windows, enumerate_windows


class RecordWindowDataset(Dataset):
    """One filtered record's worth of EEG windows, served as PyTorch tensors.

    Use this together with ``torch.utils.data.ConcatDataset`` to build a
    full training fold: instantiate one ``RecordWindowDataset`` per ictal
    / interictal pool entry (typically via :func:`build_record_datasets`),
    concatenate them, and wrap in a ``DataLoader``.

    Public attributes (useful for post-construction filtering of fold
    splits):

    - :attr:`subject` (``str``): the subject this record belongs to
      (e.g. ``"chb05"``). Filter on this for cross-subject splits.
    - :attr:`record_name` (``str``): the EDF filename this record came
      from (e.g. ``"chb05_06.edf"``).
    - :attr:`label` (``int``): ``0`` for interictal, ``1`` for ictal.
    - :attr:`windows` (``list[tuple[float, float]]``): the
      ``(start, end)`` window bounds in record-relative seconds, as
      returned by :func:`get_record_windows`.

    Tensor types returned by :meth:`__getitem__`:

    - ``X``: ``float32`` tensor of shape ``(1, n_channels, n_times)``
      where ``n_channels`` is the harmonised CHB-MIT bipolar count (22)
      and ``n_times = round(segment_length * sfreq)`` (e.g. 1280 at
      ``segment_length=5`` and ``sfreq=256``). The leading ``1`` is the
      spatial/depth axis EEGNet's first ``Conv2d`` (or any ``Conv2d``)
      layer expects. ``float32`` is the deep-learning standard —
      ``float64`` would double memory and compute on GPU with no
      model-quality gain.
    - ``y``: ``long`` (``int64``) scalar tensor. Required by
      :class:`torch.nn.CrossEntropyLoss` and the standard dtype for
      classification heads.
    """

    def __init__(
        self,
        record: dict,
        label: int,
        segment_length: float,
        stride: float | None = None,
    ):
        """Build the per-window index for one record.

        Does NOT open the cached FIF — only stores the path, identity
        metadata, and the list of windows (in record-relative seconds).
        The file is opened lazily by :meth:`_ensure_raw` on first
        :meth:`__getitem__` call.

        Args:
            record: Pool entry from
                :func:`...segmentation.find_eligible_records`. Must
                carry a ``filtered_path`` key (attached by
                :func:`...preprocessing.band_pass_filtering`) plus
                ``subject`` and ``record`` (set by the loader). The
                other required keys depend on the ``label`` — see
                :func:`get_record_windows`.
            label: ``0`` for interictal, ``1`` for ictal. Kept as an
                int so it can be passed straight to PyTorch loss
                functions without conversion.
            segment_length: Window length, in seconds. From
                ``config.yaml`` (``data.chb_mit.segament_window``).
            stride: Step between consecutive window starts, in seconds.
                ``None`` → non-overlapping (defaults to ``segment_length``
                inside :func:`enumerate_windows`). Set
                ``stride < segment_length`` for overlapping windows
                (knob for inflating training data and easing class
                imbalance).
        """
        self.filtered_path = record["filtered_path"]
        self.label = label
        self.subject = record["subject"]
        self.record_name = record["record"]
        self.windows = get_record_windows(
            record=record,
            label=label,
            segment_length=segment_length,
            stride=stride,
        )
        # Populated on first __getitem__ call (per worker); see _ensure_raw.
        self._raw = None
        self._sfreq = None

    def __len__(self):
        """Number of windows this record contributes."""
        return len(self.windows)

    def _ensure_raw(self):
        """Lazy-open the cached FIF and cache ``sfreq`` on first access.

        Called from :meth:`__getitem__`. Runs in the calling worker
        process, so each ``DataLoader`` worker ends up with its own
        ``Raw`` and its own file descriptor — see module docstring for
        the fork-safety rationale.
        """
        if self._raw is None:
            self._raw = mne.io.read_raw_fif(
                self.filtered_path,
                preload=False,
                verbose="ERROR",
            )
            # Cache sfreq so __getitem__'s hot path is a single dict-free
            # attribute read rather than a per-call info-dict lookup.
            self._sfreq = self._raw.info["sfreq"]
        return self._raw

    def __getitem__(self, index):
        """Return one window as a ``(X, y)`` tensor pair.

        Args:
            index: 0-based position in :attr:`windows`.

        Returns:
            ``(X, y)`` where ``X`` is a ``float32`` tensor of shape
            ``(1, n_channels, n_times)`` and ``y`` is a scalar ``long``
            tensor carrying this record's class label.
        """
        raw = self._ensure_raw()
        start, end = self.windows[index]
        # Seconds → integer sample indices. Deterministic round per call,
        # so two requests for the same window always hit the same sample
        # range (matters for reproducibility across DataLoader epochs).
        s0 = int(round(start * self._sfreq))
        s1 = int(round(end * self._sfreq))
        X = raw.get_data(start=s0, stop=s1)   # shape (n_channels, s1 - s0)

        # Add Conv2d spatial/depth axis and cast to float32 in one step.
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.label, dtype=torch.long)
        return X, y


def build_record_datasets(
    pool: dict,
    label: int,
    segment_length: float,
    stride: float | None = None,
) -> list[RecordWindowDataset]:
    """Construct one :class:`RecordWindowDataset` per pool entry.

    Args:
        pool: ``ictal_records`` or ``interictal_records`` from
            :func:`find_eligible_records`. Keyed by subject; values
            are per-subject lists of record dicts.
        label: ``0`` for interictal, ``1`` for ictal. Applied to every
            Dataset produced by this call.
        segment_length: Forwarded to :class:`RecordWindowDataset`.
        stride: Forwarded to :class:`RecordWindowDataset`.

    Returns:
        Flat list of :class:`RecordWindowDataset` instances, one per
        record across every subject in ``pool``. Wrap in
        :class:`torch.utils.data.ConcatDataset` to form a single
        Dataset suitable for ``DataLoader``.

    """
    return [
        RecordWindowDataset(record, label, segment_length, stride)
        for records in pool.values()
        for record in records
    ]


class ContinuousRecordDataset(Dataset):
    """One filtered record streamed window-by-window for event-level testing.

    Used **only** for event-level evaluation after training, never for
    training itself. Where :class:`RecordWindowDataset` exposes the
    curated, eligibility-filtered windows the model was trained on, this
    class walks the **entire** record so the evaluation sees what a
    wearable would see in deployment — including the pre-ictal and
    post-ictal stretches the n-hour buffer excluded from training.

    Three deliberate differences from :class:`RecordWindowDataset`:

    1. **Whole-record windows.** Calls :func:`enumerate_windows` over
       ``[0, record_total_duration]`` directly rather than
       :func:`get_record_windows`, so nothing is dropped — boundary
       windows, buffer-zone windows and all.
    2. **Finer stride** (default 1.0 s vs the training default of
       ``segment_length``). The stride sets the resolution of the
       detection-latency metric, so a 5 s stride would quantise every
       alarm time to the nearest 5 s.
    3. **No label.** :meth:`__getitem__` returns ``(X, position)``. The
       scoring category (``"ictal"`` / ``"interictal"`` / ``"other"``)
       is *not* computed here — it depends on the interictal pool, which
       this class has no access to. The scoring loop derives it from
       ``position`` instead.

    Public attributes (mirroring :class:`RecordWindowDataset`):

    - :attr:`subject` (``str``): e.g. ``"chb05"``.
    - :attr:`record_name` (``str``): e.g. ``"chb05_06.edf"``. The key the
      scoring loop uses to look up this record's seizure list and
      interictal bounds.
    - :attr:`record_starts_at` (``float``): seconds since the start of
      the subject's first recording. Needed for the absolute clock —
      see the ordering note below.
    - :attr:`windows` (``list[tuple[float, float]]``): every
      ``(start, end)`` pair in the record, in record-relative seconds.

    Returned by :meth:`__getitem__`:

    - ``X``: ``float32`` tensor of shape ``(1, n_channels, n_times)`` —
      identical contract to :class:`RecordWindowDataset`.
    - ``position``: ``dict`` of ``subject``, ``record_name``,
      ``window_start``, ``window_end``, ``record_starts_at``. All plain
      ``str`` / ``float``, so batching stays cheap.

    Ordering requirements (easy to miss, and they break metrics silently):

    - The ``DataLoader`` **must** use ``shuffle=False``. The
      post-processing state machine is stateful (smoothing buffer,
      hysteresis, refractory timer) and only produces meaningful alarms
      if it sees windows in real-time order.
    - The scoring loop **must** feed the state machine
      ``record_starts_at + window_start``, not ``window_start`` alone.
      ``window_start`` is record-relative and resets to ``0.0`` at every
      record boundary, so using it directly makes the clock jump
      backwards ~1 hour between records and silently disengages the
      refractory period.
    """

    def __init__(
        self,
        record: dict,
        segment_length: float,
        stride: float = 1.0,
    ):
        """Build the whole-record window index for one record.

        Does NOT open the cached FIF — only stores the path, identity
        metadata, and the window list. The file is opened lazily by
        :meth:`_ensure_raw` on first :meth:`__getitem__` call.

        Args:
            record: Entry from a split's ``test_continuous`` pool. Must
                carry ``filtered_path``, ``subject``, ``record``,
                ``record_starts_at`` and ``record_total_duration``.
                ``test_continuous`` entries come from different sources
                per split mode — loader records for cross-subject, pool
                records for intra-subject — but all five keys are
                present either way.
            segment_length: Window length, in seconds. Must match what
                the model was trained on or the input shape won't fit.
            stride: Step between consecutive window starts, in seconds.
                Defaults to 1.0 rather than ``segment_length`` because
                this sets detection-latency resolution — non-overlapping
                windows would round every alarm time to the nearest
                ``segment_length``.
        """
        self.filtered_path = record["filtered_path"]
        self.subject = record["subject"]
        self.record_name = record["record"]
        self.record_starts_at = record["record_starts_at"]

        # Whole record, start to finish — no eligibility filtering. This is
        # what makes the evaluation deployment-realistic: buffer-zone
        # windows the model never trained on are included, because the
        # device would be fed them too.
        self.windows = enumerate_windows(
            start=0.0,
            end=record["record_total_duration"].total_seconds(),
            segment_length=segment_length,
            stride=stride,
        )

        # Populated on first __getitem__ call (per worker); see _ensure_raw.
        self._raw = None
        self._sfreq = None

    def __len__(self):
        """Number of windows this record contributes."""
        return len(self.windows)

    def _ensure_raw(self):
        """Lazy-open the cached FIF and cache ``sfreq`` on first access.

        Called from :meth:`__getitem__`. Runs in the calling worker
        process, so each ``DataLoader`` worker ends up with its own
        ``Raw`` and its own file descriptor — see module docstring for
        the fork-safety rationale.
        """
        if self._raw is None:
            self._raw = mne.io.read_raw_fif(
                self.filtered_path,
                preload=False,
                verbose="ERROR",
            )
            # Cache sfreq so __getitem__'s hot path is a single dict-free
            # attribute read rather than a per-call info-dict lookup.
            self._sfreq = self._raw.info["sfreq"]
        return self._raw

    def __getitem__(self, index):
        """Return one window as an ``(X, position)`` pair.

        Args:
            index: 0-based position in :attr:`windows`.

        Returns:
            ``(X, position)`` where ``X`` is a ``float32`` tensor of
            shape ``(1, n_channels, n_times)`` and ``position`` is a
            dict locating the window on the subject's timeline. No
            label is returned — see the class docstring for why the
            scoring category is derived downstream instead.
        """
        raw = self._ensure_raw()
        start, end = self.windows[index]
        # Seconds → integer sample indices. Deterministic round per call,
        # so two requests for the same window always hit the same sample
        # range (matters for reproducibility across evaluation passes).
        s0 = int(round(start * self._sfreq))
        s1 = int(round(end * self._sfreq))
        X = raw.get_data(start=s0, stop=s1)   # shape (n_channels, s1 - s0)

        # Add Conv2d spatial/depth axis and cast to float32 in one step.
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        # Everything the scoring loop needs: `record_name` to look up this
        # record's seizure list / interictal bounds, and `record_starts_at`
        # to place the window on the subject's absolute timeline for the
        # state machine's clock. Kept to plain str/float so batching is cheap.
        position = {
            "subject": self.subject,
            "record_name": self.record_name,
            "window_start": start,  # record-relative seconds
            "window_end": end,
            "record_starts_at": self.record_starts_at,  # for absolute-timeline math
        }

        return X, position


def continuous_collate_fn(batch):
    """Collate :class:`ContinuousRecordDataset` items into a batch.

    Stacks the ``X`` tensors as usual and keeps ``position`` as a plain
    **list of dicts**, one per window.

    Torch's ``default_collate`` would also work here — ``position`` holds
    only ``str`` and ``float`` values, all of which it handles — but it
    recurses into dicts field-wise and hands back a *dict of lists*
    instead::

        # default_collate
        {"window_start": tensor([0., 1.]), "record_name": ["a.edf", "a.edf"]}
        # this function
        [{"window_start": 0.0, "record_name": "a.edf"}, {...}]

    The list-of-dicts form lets the scoring loop iterate
    ``zip(probs, positions)`` and read plain floats, rather than
    index-juggling parallel arrays and unwrapping 0-d tensors with
    ``.item()``. Purely an ergonomics choice — the default collate is a
    drop-in replacement if you prefer the struct-of-arrays shape.

    Args:
        batch: List of ``(X, position)`` pairs from
            :meth:`ContinuousRecordDataset.__getitem__`.

    Returns:
        ``(X, positions)`` where ``X`` is a ``float32`` tensor of shape
        ``(batch_size, 1, n_channels, n_times)`` and ``positions`` is a
        list of the per-window dicts, aligned with ``X``'s first axis.
    """
    X = torch.stack([item[0] for item in batch])
    positions = [item[1] for item in batch]

    return X, positions


def build_continuous_datasets(
    test_continuous: dict,
    segment_length: float,
    stride: float = 1.0,
) -> list[ContinuousRecordDataset]:
    """Construct one :class:`ContinuousRecordDataset` per ``test_continuous`` record.

    Args:
        test_continuous: The ``"test_continuous"`` pool from
            :func:`split_cross_subject` or :func:`split_intra_subject`.
            Keyed by subject; values are per-subject lists of record
            dicts already sorted by ``record_meas_date``.
        segment_length: Forwarded to :class:`ContinuousRecordDataset`.
            Must match the value the model was trained with.
        stride: Forwarded to :class:`ContinuousRecordDataset`. Defaults
            to 1.0 s for fine detection-latency resolution.

    Returns:
        Flat list of :class:`ContinuousRecordDataset` instances in
        **chronological order**. Wrap in
        :class:`torch.utils.data.ConcatDataset` and pass to a
        ``DataLoader`` with ``shuffle=False`` and
        ``collate_fn=continuous_collate_fn``.
    """
    return [
        ContinuousRecordDataset(record, segment_length, stride)
        for records in test_continuous.values()
        for record in records
    ]


def build_split_datasets(
    splits: dict,
    segment_length: float,
    stride: float | None = None,
    continuous_stride: float = 1.0,
    pool_labels: dict | None = None,
) -> dict:
    """Compose fold-level ``ConcatDataset``s from a split's pools.

    The one-call bridge between the split layer and the training loop:
    takes the dict returned by :func:`split_cross_subject` /
    :func:`split_intra_subject` and wraps every pool of every split in
    the right Dataset class, so a notebook goes from split to
    DataLoader-ready fold in a single call.

    Args:
        splits: The dict returned by :func:`split_cross_subject` or
            :func:`split_intra_subject`. The ``"training"`` / ``"val"`` /
            ``"test"`` entries are ``{"ictal": pool, "interictal": pool}``
            dicts; ``"test_continuous"`` is a plain pool.
        segment_length: Window length, in seconds. Forwarded to both
            builders — one value for the whole fold, since the model's
            input shape is fixed by it.
        stride: Step between consecutive window starts for the curated
            splits, in seconds. Forwarded to
            :func:`build_record_datasets`; ``None`` → non-overlapping.
            See :class:`RecordWindowDataset` for details.
        continuous_stride: Step for the event-level stream, in seconds.
            Forwarded to :func:`build_continuous_datasets`; the 1.0 s
            default gives fine detection-latency resolution. See
            :class:`ContinuousRecordDataset` for details.
        pool_labels: Maps pool name → integer class label. Defaults to
            ``{"interictal": 0, "ictal": 1}`` (the Detection Phase contract).
            A parameter rather than a hardcoded map so PredictionPhase 2 can
            pass ``{"interictal": 0, "preictal": 1}`` without touching this
            function. An unknown pool name raises ``KeyError`` — by
            design, so a typo'd or unexpected pool fails loudly instead
            of being silently mislabelled.

    Returns:
        Dict of fold-level ``ConcatDataset``s —
        ``{"training": ..., "val": ..., "test": ..., "test_continuous": ...}``
        — each key present only if it was present in ``splits``.
    """
    if pool_labels is None:
        # Default built inside the function, not in the signature: a
        # mutable default dict would be shared across every call (the
        # classic Python default-argument trap).
        pool_labels = {"interictal": 0, "ictal": 1}

    fold = {}
    for split_name in ("training", "val", "test"):
        if split_name not in splits:
            continue

        record_datasets = []
        for pool_name, pool in splits[split_name].items():
            record_datasets.extend(
                build_record_datasets(
                    pool=pool,
                    label=pool_labels[pool_name],  # KeyError on unknown pools, by design
                    segment_length=segment_length,
                    stride=stride,
                )
            )

        fold[split_name] = ConcatDataset(record_datasets)

    if "test_continuous" in splits:
        fold["test_continuous"] = ConcatDataset(
            build_continuous_datasets(
                test_continuous=splits["test_continuous"],
                segment_length=segment_length,
                stride=continuous_stride,
            )
        )

    return fold