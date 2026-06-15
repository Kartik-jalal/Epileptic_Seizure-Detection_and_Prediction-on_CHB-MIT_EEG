"""Enumerate (window_start, window_end) tuples from per-record bounds.

Windowing deliberately lives downstream of :func:`find_eligible_records`: 
the eligibility module stops at *bounds* 
(seizure intervals + trimmed interictal sub-intervals), and the Dataset
calls into this module from its ``__init__`` to enumerate the actual
windows for the chosen ``segment_length`` / ``stride``.

All coordinates here are **record-relative seconds**, matching MNE
``Raw``'s per-record time axis. The interictal pool's
``interictal_period_starts_at`` / ``_ends_at`` and the ictal pool's
``seizure_info["seizures"]`` already use this convention, so windows
returned by :func:`get_record_windows` can be passed straight to
``raw.get_data(tmin=..., tmax=...)`` or ``raw.crop(...)``.
"""


def enumerate_windows(
    start: float,
    end: float,
    segment_length: float,
    stride: float | None = None,
) -> list[tuple[float, float]]:
    """Enumerate ``(window_start, window_end)`` tuples fitting inside ``[start, end]``.

    Windows are emitted left-to-right at offsets ``start + i*stride`` for
    ``i = 0, 1, ..., n-1`` where ``n`` is the largest integer such that
    the last window still fits inside ``[start, end]``. The integer-times-
    stride form avoids the float drift you'd get from
    ``np.arange(start, end, stride)`` accumulating rounding error across
    the loop.

    Args:
        start: Inclusive left bound of the eligible interval, in
            record-relative seconds.
        end: Right bound (exclusive in spirit — the last window's
            ``window_end`` may equal ``end``). Same units as ``start``.
        segment_length: Window length, in seconds. From ``config.yaml``
            (``data.chb_mit.segament_window``).
        stride: Step (s) between consecutive window starts. Defaults to
            ``segment_length`` (non-overlapping windows). Set
            ``stride < segment_length`` for overlapping windows
            (knob for inflating training data and easing
            class imbalance).

    Returns:
        List of ``(window_start, window_end)`` tuples. Empty if the
        interval is shorter than ``segment_length``.
    """
    if stride is None:
        # Non-overlapping default. The moment overlap is enabled,
        # the caller passes an explicit smaller stride.
        stride = segment_length

    span = end - start
    if span < segment_length:
        # Defence in depth — upstream callers should already guarantee
        # ``span >= segment_length``: ``find_eligible_records`` enforces
        # this for both pools (interictal via the buffer-trim check,
        # ictal via the per-seizure length filter). The guard stays so
        # this function is safe to call from exploratory or future code
        # paths that might bypass that upstream filter.
        return []

    # ``n`` = number of strides that still leave room for a full window
    # at the end. Integer-floored to avoid emitting a partial trailing
    # window that would run past ``end``.
    n = int((span - segment_length) // stride) + 1
    return [
        (start + i * stride, start + i * stride + segment_length)
        for i in range(n)
    ]


def get_record_windows(
    record: dict,
    label: int,
    segment_length: float,
    stride: float | None = None,
) -> list[tuple[float, float]]:
    """Enumerate every window for one record, given its class label.

    Dispatches into :func:`enumerate_windows` over the appropriate
    bounds for the chosen class:

    - ``label == 0`` (interictal): the record's single trimmed sub-interval
      ``[interictal_period_starts_at, interictal_period_ends_at]``,
      attached by :func:`find_eligible_records`.
    - ``label == 1`` (ictal): every seizure in
      ``record["seizure_info"]["seizures"]`` contributes its own windows;
      the per-seizure windows are concatenated into one flat list. Note
      that for ictal-pool entries this list has already been filtered to
      seizures of duration ≥ ``segment_length`` by
      :func:`find_eligible_records`.

    Args:
        record: Pool entry from
            :func:`...segmentation.find_eligible_records`. The required
            keys depend on the ``label``:

            - ``label=0``: ``interictal_period_starts_at`` /
              ``interictal_period_ends_at``.
            - ``label=1``: ``seizure_info["seizures"]`` as a list of
              ``[start, end]`` pairs.
        label: ``0`` for interictal, ``1`` for ictal. Kept as an int
            so it can be passed straight to PyTorch loss functions
            (cross-entropy, BCE) without conversion.
        segment_length: Window length in seconds. Forwarded to
            :func:`enumerate_windows`.
        stride: Step between windows. ``None`` → non-overlapping
            (defaults to ``segment_length``). Forwarded as-is.

    Returns:
        Flat list of ``(window_start, window_end)`` tuples in
        record-relative seconds. Possibly empty if the bounds don't fit
        any window (shouldn't happen with eligible-pool records, but
        :func:`enumerate_windows` handles the case defensively).
    """
    assert label in (0, 1), (
        f"label must be 0 (interictal) or 1 (ictal), got {label!r}"
    )

    if label == 0:
        return enumerate_windows(
            record["interictal_period_starts_at"],
            record["interictal_period_ends_at"],
            segment_length,
            stride,
        )

    # label == 1 (ictal): one record can contain multiple seizures
    # (CHB-MIT records have between 1 and ~7 each), so concatenate the
    # per-seizure window lists into one flat list.
    windows: list[tuple[float, float]] = []
    for seizure_start, seizure_end in record["seizure_info"]["seizures"]:
        windows.extend(
            enumerate_windows(seizure_start, seizure_end, segment_length, stride)
        )
    return windows
