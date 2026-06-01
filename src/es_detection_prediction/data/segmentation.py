"""Split CHB-MIT records into ictal and valid-interictal pools.

This module is the bridge between the dataloader (which produces records
placed on a per-subject continuous wall-clock timeline) and the downstream
window-extraction / labelling step. It does NOT assign per-window labels —
but it *does* identify, for each loader record:

- whether the record contains any seizures (so its ictal windows can be
  extracted downstream), and
- if not, which sub-interval of the record (if any) is at least
  ``ictal_buffer`` seconds away from every seizure in the subject's
  continuous timeline.

The n-hour ictal buffer is enforced in absolute timeline coordinates so the buffer
naturally spans across record boundaries and across the silent gaps between 
consecutively-recorded EDFs (captured by the loader's ``gep`` variable on each record).
A no-seizure record is dropped from the interictal pool if, after trimming, its surviving 
sub-interval is shorter than one ``segment_length`` segment.

Subjects whose seizures are too dense to permit *any* buffer-distant
interictal sub-interval (chb12 and chb24 in CHB-MIT, both with >50% of their
records seizure-containing) will simply be absent from the returned
``interictal_records`` dict. They can still contribute ictal windows.

The pool is computed per subject. The timer in the loader's record output
is already per-subject, so we never compare seizure timelines across
subjects — that has no physical meaning.
"""

import bisect
from collections import defaultdict


def find_eligible_records(
    records: list,
    ictal_buffer: float,
    segment_length: float,
) -> tuple[dict, dict]:
    """Partition loader records into ictal and valid-interictal pools, per subject.

    For each subject this function:

    1. Collects every annotated seizure on the subject's continuous timeline
       as an ``(abs_start, abs_end)`` pair, using ``record_starts_at`` from
       the loader to convert from per-record to per-subject coordinates.
       The result is a single chronologically-sorted seizure list per
       subject (cheap to do once, then reused for fast lookups).
    2. Walks the subject's records in chronological order. Records that are
       marked ``is_valid=False`` by the loader (channel-set mismatch) are
       skipped entirely — they can't contribute training segments.
    3. Records with one or more seizures go into ``ictal_records[subject]``
       unchanged.
    4. No-seizure records have their candidate interictal interval
       ``[record_starts_at, record_ends_at]`` trimmed by:

       - the post-ictal buffer of the **latest** seizure that ends at or
         before ``record_starts_at`` (binary-searched in the subject's
         seizure list), if any;
       - the pre-ictal buffer of the **earliest** seizure that starts at or
         after ``record_ends_at`` (binary-searched), if any.

       For CHB-MIT records (max 4 hours) and a 4-hour ictal buffer, at most one
       seizure on each side can be the binding constraint, so the trimmed
       interval is always a single contiguous range. The latest preceding
       seizure dominates because its post-ictal buffer extends furthest
       into the candidate record; symmetrically for the earliest following
       seizure.
    5. If the surviving sub-interval is shorter than ``segment_length``, the
       record is skipped (no usable interictal segmant fits). Otherwise the
       record is shallow-copied and the new fields
       ``interictal_period_starts_at`` / ``interictal_period_ends_at`` are
       added in **record-relative seconds** (matching the convention used
       for seizure start/end in ``seizure_info``).

    Args:
        records: List of record dicts from
            :func:`es_detection_prediction.data.load_records`. Each must
            carry ``subject``, ``is_valid``, ``seizure_info``,
            ``record_starts_at`` and ``record_ends_at``.
        ictal_buffer: Seconds of separation required around every
            seizure.
        segment_length: Minimum length, in seconds, that a surviving
            interictal sub-interval must reach to be useful. Records whose
            trimmed interval is shorter are dropped from the interictal
            pool. Comes from ``config.yaml`` (``data.chb_mit.segment_length``).

    Returns:
        Tuple ``(ictal_records, interictal_records)`` of two dicts keyed by
        subject (e.g. ``"chb01"``). Each value is a list of record dicts following
        the order present in ``records`` for each subject.

        - ``ictal_records[subject]`` entries are the original loader dicts
          (same object identity — not copied).
        - ``interictal_records[subject]`` entries are shallow copies of the
          loader dicts extended with two extra keys::

              "interictal_period_starts_at": float  # record-relative seconds
              "interictal_period_ends_at":   float  # record-relative seconds

          Both bounds satisfy
          ``0.0 <= ... <= record_total_duration.total_seconds()`` and
          ``end - start >= segment_length``.

    Notes:
        Subjects whose seizures are too dense for any buffer-distant
        interictal sub-interval to exist will be absent from the returned
        ``interictal_records`` dict but still present in ``ictal_records``.
        On the current CHB-MIT snapshot with ``interictal_buffer=14400`` s
        this affects ``chb12`` and ``chb24`` (both >50% seizure records).
        That's "the buffer wastes data but produces clean labels" tradeoff 
        in action — not a bug.
    """
    # Group loader records by subject. The loader already sorts records
    # chronologically (by EDF meas_date) within each subject, so the per-
    # subject lists below inherit that ordering for free.
    by_subject: dict = defaultdict(list)
    for record in records:
        by_subject[record["subject"]].append(record)

    # Build per-subject seizure lists in absolute timeline coordinates. The
    # loader sorts seizures within a record by start time and records by
    # meas_date, so concatenating in iteration order gives a globally sorted
    # list — no extra sort needed. We keep two parallel lists (starts, ends)
    # because `bisect` operates on a single sorted sequence per call.
    seizure_starts_by_subject: dict = {}
    seizure_ends_by_subject: dict = {}
    for subject, subj_records in by_subject.items():
        starts = []
        ends = []
        for record in subj_records:
            for seizure_start, seizure_end in record["seizure_info"]["seizures"]:
                starts.append(record["record_starts_at"] + seizure_start)
                ends.append(record["record_starts_at"] + seizure_end)
        seizure_starts_by_subject[subject] = starts
        seizure_ends_by_subject[subject] = ends

    ictal_records: dict = defaultdict(list)
    interictal_records: dict = defaultdict(list)

    for subject, subj_records in by_subject.items():
        seizure_starts = seizure_starts_by_subject[subject]
        seizure_ends = seizure_ends_by_subject[subject]

        for record in subj_records:
            if not record["is_valid"]:
                # Channel-set mismatch — can't contribute training segments.
                # Loader keeps these so the timeline (and the seizures they
                # carry) stays intact, but we never label them.
                continue

            if record["seizure_info"]["no_of_seizures"] > 0:
                ictal_records[subject].append(record)
                continue

            # No-seizure record: trim its [start, end] in absolute timeline
            # coordinates by the ictal buffers of the binding seizures on each
            # side.
            record_start = record["record_starts_at"]
            record_end = record["record_ends_at"]

            interictal_abs_start = record_start
            interictal_abs_end = record_end

            # Latest seizure that ENDS at or before record_start. Its
            # post-ictal buffer pushes the leading edge forward in time.
            # `bisect_right`/`bisect` returns the index where x should be
            #  inserted. If x already exists, it gives the position after
            #  (to the right of) existing entries. 
            # So, here `bisect_right - 1` returns the index of the last element <=
            # record_start, or -1 if none exists.
            idx_prev = bisect.bisect_right(seizure_ends, record_start) - 1
            if idx_prev >= 0:
                interictal_abs_start = max(
                    interictal_abs_start,
                    seizure_ends[idx_prev] + ictal_buffer,
                )

            # Earliest seizure that STARTS at or after record_end. Its
            # pre-ictal buffer pulls the trailing edge backward in time.
            # `bisect_left` returns the index where x should be inserted.
            #  If x is already in the list, it gives the position before 
            # (to the left of) existing entries.
            # So, here `bisect_left` returns the first index of an element >=
            # record_end, or len(...) if none exists.
            idx_next = bisect.bisect_left(seizure_starts, record_end)
            if idx_next < len(seizure_starts):
                interictal_abs_end = min(
                    interictal_abs_end,
                    seizure_starts[idx_next] - ictal_buffer,
                )

            # After trimming, is there room for at least one segment?
            if interictal_abs_end - interictal_abs_start < segment_length:
                continue

            # Store the trimmed bounds in record-relative seconds so they
            # match `seizure_info`'s convention and can be used directly
            # against MNE Raw's per-record time axis.
            interictal_records[subject].append({
                **record,
                "interictal_period_starts_at": interictal_abs_start - record_start,
                "interictal_period_ends_at": interictal_abs_end - record_start,
            })

    return dict(ictal_records), dict(interictal_records)
