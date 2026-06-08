"""Band-pass filter eligible records to disk so downstream training never re-filters.

This module is the third stage of the data pipeline. After
:func:`find_eligible_records` partitions loader records into ictal and
interictal pools, this module band-pass filters every record in either
pool **once** and writes the result to a versioned FIF cache. The
``Dataset`` then loads windows from those cached files — no MNE filter
ever runs inside the training loop.

Cache layout
------------
Filtered records are written under::

    {output_dir}/bp_{l_freq}-{h_freq}_{h_trans_bandwidth}/{subject}/{record_stem}_filtered.fif

Encoding the bandpass cutoffs in the directory name gives free cache
versioning: changing ``l_freq`` / ``h_freq`` / ``h_trans_bandwidth`` in
``config.yaml`` lands on a fresh directory and the old cache is left untouched.

A re-run that targets an existing FIF skips the filter and re-uses the
file on disk, so an interrupted job can be resumed simply by re-invoking
:func:`band_pass_filtering` with the same parameters.

One non-obvious gotcha baked into this module
---------------------------------------------
**Worker mutations do NOT propagate back to the parent.**
``process_map`` spawns subprocesses. Each worker calls
``raw.load_data().filter(...)`` on its own pickled copy of the
record dict — the parent's record dict is untouched. The worker
therefore *returns* the save path and the parent stitches it back
onto each record under the ``filtered_path`` key after
``process_map`` returns.

The pool dicts passed in here are themselves shallow copies of the
loader records (see :func:`find_eligible_records`'s Returns), so
the ``filtered_path`` key written here lands on the pool entry only
— the loader records list is left untouched.
"""

from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path


def filter_and_save_records(
    record: dict,
    output_dir: Path,
    l_freq: float,
    h_freq: float,
    h_trans_bandwidth: float,
) -> Path:
    """Band-pass filter a single record and save the result as a FIF cache.

    This is the per-record worker invoked by :func:`band_pass_filtering`
    via ``process_map``.

    The function is idempotent: if the destination FIF already exists,
    the existing path is returned without re-running the filter. That
    makes interrupted runs trivially resumable — just call
    :func:`band_pass_filtering` again with the same parameters.

    Args:
        record: A record dict from :func:`find_eligible_records`. Must
            carry ``subject``, ``record`` (e.g. ``"chb01_03.edf"``),
            and ``raw`` (an MNE ``Raw`` — ``load_data`` is called
            inside, so ``preload`` state does not matter).
        output_dir: The bandpass-versioned root directory (e.g.
            ``.../bp_0.5-50_4.0``). The function creates
            ``{output_dir}/{subject}/`` if needed and writes the FIF
            under it.
        l_freq: Low cutoff (Hz). Passed straight to ``raw.filter``.
        h_freq: High cutoff (Hz). Passed straight to ``raw.filter``.
        h_trans_bandwidth: Width (Hz) of the upper transition band.

    Returns:
        Absolute ``Path`` to the saved FIF.
    """
    output_dir = output_dir / record["subject"]
    output_dir.mkdir(exist_ok=True, parents=True)

    # `record["record"]` is the EDF filename (e.g. "chb01_03.edf"); strip
    # the suffix so the FIF lands next to it with a clean name.
    save_path = output_dir / f"{record['record'].split('.')[0]}_filtered.fif"

    # Idempotent cache: an interrupted run re-uses files already on disk
    # instead of re-filtering them. The cache directory carries the
    # bandpass cutoffs in its name, so two runs landing here mean the
    # filter configuration matches.
    if save_path.exists():
        return save_path

    # n_jobs=1 pins each worker to a single thread for filtering — outer
    # parallelism is provided by process_map.
    # verbose="ERROR" silences MNE's per-record INFO log spam; real
    # failures still raise.
    record["raw"].load_data().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        h_trans_bandwidth=h_trans_bandwidth,
        n_jobs=1,
        verbose="ERROR",
    )

    # CHB-MIT anonymises EDF dates to 2074-2075. FIF stores meas_date as
    # int32 seconds since epoch (overflows ~2038), so writing fails. The
    # timeline info we actually use downstream lives on the record dict
    # (record_starts_at / record_ends_at / gap), NOT in the FIF — so
    # it's safe to drop meas_date from the saved file. 
    record["raw"].set_meas_date(None)

    record["raw"].save(fname=save_path, overwrite=True, verbose="ERROR")

    return save_path


def band_pass_filtering(
    pool: dict,
    output_dir: Path,
    desc: str,
    l_freq: float = 0.5,
    h_freq: float = 50,
    h_trans_bandwidth: float = 4,
    max_workers: int = 4,
) -> None:
    """Filter every record in a pool dict in parallel and cache to disk.

    Iterates the pool one subject at a time so the per-record progress
    bar is grouped per subject (easier to read for a 24-subject corpus
    than a single 600+ record bar). 

    After ``process_map`` returns for a subject, each record in
    ``pool[subject]`` is mutated in-place to carry a ``filtered_path``
    key pointing at the saved FIF. Pool entries are shallow copies of
    the loader records (see :func:`find_eligible_records`), so this
    mutation stays local to the pool entry — the loader records list
    is left untouched, regardless of whether the pool is ``ictal_records``
    or ``interictal_records``. The ``raw`` MNE object is shared by
    reference between the pool entry and the loader record (shallow
    copy semantics), but that's harmless because workers operate on
    pickled subprocess copies and never mutate the parent's ``raw``.

    Args:
        pool: Output of :func:`find_eligible_records` — either
            ``ictal_records`` or ``interictal_records``. A dict keyed
            by subject (``"chb01"``, ``"chb02"``, ...) whose values are
            lists of record dicts.
        output_dir: Root cache directory (typically
            ``data.chb_mit.filtered_dir`` from ``config.yaml``). The
            actual writes go to ``{output_dir}/bp_{l_freq}-{h_freq}_{h_trans_bandwidth}/``
            so different bandpass settings produce different caches.
        desc: Prefix shown on the per-subject tqdm bar (e.g.
            ``"ictal"`` or ``"interictal"``) so the two passes through
            this function are visually distinguishable.
        l_freq: Low cutoff (Hz). 0.5 Hz matches workflow §5.1 —
            removes baseline drift / electrode DC offset while
            preserving seizure morphology.
        h_freq: High cutoff (Hz). 50 Hz matches workflow §5.1 — passes
            the seizure-relevant band (delta through low gamma) while
            attenuating EMG and mains coupling above it.
        h_trans_bandwidth: Controls how sharply the upper edge of the band-pass
            rolls off (Hz). MNE centres the transition band on h_freq (the −6 dB
            point), so with h_freq=50 and h_trans_bandwidth=4 the passband edge
            sits at 48 Hz and the stopband edge at 52 Hz — narrow enough to keep
            low-gamma content intact, wide enough that the FIR kernel stays short
            (kernel length scales as sfreq / h_trans_bandwidth, so halving this
            doubles both the compute cost and the boundary ringing). 4 Hz also
            leaves the US 60 Hz mains well inside the stopband for CHB-MIT, so no
            separate notch is needed.
        max_workers: Pool size for ``process_map``. With ``n_jobs=1``
            inside each worker, total in-use cores ≈ ``max_workers``.
            4 is a safe default on most workstations; bump if you have
            many physical cores and ample RAM (each worker holds one
            full record in memory).
    """
    output_dir = output_dir / f"bp_{l_freq}-{h_freq}_{h_trans_bandwidth}"

    # `partial` binds every kwarg except `record` so the resulting
    # callable matches process_map's expected one-arg-per-task signature.
    func = partial(
        filter_and_save_records,
        output_dir=output_dir,
        l_freq=l_freq,
        h_freq=h_freq,
        h_trans_bandwidth=h_trans_bandwidth
    )

    for subject, records in pool.items():
        # chunksize=1: each filter job runs for seconds, so per-task
        # dispatch overhead is negligible relative to compute. Higher
        # chunksizes only help when individual tasks are sub-millisecond
        # (and would here just delay the per-record progress updates).
        filtered_paths = process_map(
            func,
            records,
            max_workers=max_workers,
            chunksize=1,
            desc=f"{desc} - {subject}",
            leave=False,
        )

        # process_map preserves input order (it's Pool.imap underneath),
        # so zip(records, filtered_paths) lines up record i with its
        # path. Mutating the dict here is the only way to propagate
        # state back from the workers.
        for record, filtered_path in zip(records, filtered_paths):
            record["filtered_path"] = filtered_path
