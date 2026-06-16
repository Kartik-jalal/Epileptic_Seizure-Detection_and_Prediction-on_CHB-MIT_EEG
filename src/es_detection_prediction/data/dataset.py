"""PyTorch ``Dataset`` wrapper that exposes one filtered record's windows.

One :class:`RecordWindowDataset` instance corresponds to one cached,
filtered ``.fif`` on disk (produced by :func:`band_pass_filtering`). It
enumerates the record's windows via :func:`get_record_windows`, lazily
opens the FIF on first access, and serves ``(X, y)`` tensors to the
training loop.

Lazy-open semantics (and why it matters for ``DataLoader`` workers)
-------------------------------------------------------------------
The cached FIF is opened on the first ``__getitem__`` call, NOT in
``__init__``. This matters when training uses
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

import torch
from torch.utils.data import Dataset
import mne

from .windowing import get_record_windows


class RecordWindowDataset(Dataset):
    """One filtered record's worth of EEG windows, served as PyTorch tensors.

    Use this together with ``torch.utils.data.ConcatDataset`` to build a
    full training fold: instantiate one ``RecordWindowDataset`` per ictal
    / interictal pool entry, concatenate them, and wrap in a
    ``DataLoader``. 

    Tensor types returned by :meth:`__getitem__`:

    - ``X``: ``float32`` tensor of shape ``(1, n_channels, n_times)``
      where ``n_channels`` is the harmonised CHB-MIT bipolar count (22)
      and ``n_times = round(segment_length * sfreq)`` (e.g. 1280 at
      ``segment_length=5`` and ``sfreq=256``). The leading ``1`` is the
      spatial/depth axis EEGNet's first ``Conv2d`` or any ``Conv2d`` layer expects.
      ``float32`` is the deep-learning standard — ``float64`` would
      double memory and compute on GPU with no model-quality gain.
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

        Does NOT open the cached FIF — only stores the path and the
        list of windows (in record-relative seconds). The file is
        opened lazily by :meth:`_ensure_raw` on first
        :meth:`__getitem__` call.

        Args:
            record: Pool entry from
                :func:`...segmentation.find_eligible_records`. Must
                carry a ``filtered_path`` key (attached by
                :func:`...preprocessing.band_pass_filtering`). The other
                required keys depend on the ``label`` — see
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
                (knob for inflating training data and
                easing class imbalance).
        """
        self.filtered_path = record["filtered_path"]
        self.label = label
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
