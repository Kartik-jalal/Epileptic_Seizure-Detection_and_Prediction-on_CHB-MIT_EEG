"""Read CHB-MIT EEG records and place them on a per-subject continuous timeline.

This module is the entry point of the data pipeline. For each subject it:

1. Parses ``chbXX-summary.txt`` to recover the ground-truth seizure
   intervals (sorted onset/offset in seconds, relative to each record).
2. Loads every ``.edf`` via MNE and **orders the subject's records by EDF
   ``meas_date``** — filename order is not always chronological in CHB-MIT
   (``chb02_16+.edf`` lexically precedes ``chb02_16.edf`` though it was
   recorded ~1 hour later, and ``chb03_24/25`` are numbered out of
   chronological order). Seizure annotations are still sourced from the
   summary file; absolute timing comes from the EDF header.
3. Harmonises each record to the fixed 22-channel bipolar montage defined
   in ``config.yaml``.
4. Walks the chronologically-sorted records and accumulates a per-subject
   timer so every record carries an absolute ``record_starts_at`` /
   ``record_ends_at`` (seconds since the start of that subject's first
   recording). Downstream code uses this timeline to enforce the 4-hour
   interictal buffer across neighbouring records.

Records missing any of the required channels are still returned (marked
``is_valid=False``) because their seizure positions and timeline positions
are needed for the 4-hour buffer of their neighbours, even though they
themselves can't contribute training windows.
"""

import mne
from pathlib import Path
from datetime import timedelta



def load_seizure_annotations(chb_mit_dir : Path) -> dict:
    """Parse the per-subject ``*-summary.txt`` files for seizure annotations.

    Each CHB-MIT subject directory contains a ``chbXX-summary.txt`` listing
    every recording for that subject and, for each recording, the number of
    seizures and the sorted onset/offset times (in seconds from the start of that
    recording). This function walks the summary file line by line, pairs
    each ``Seizure ... Start Time`` with its following ``Seizure ... End
    Time``, and validates that the parsed count matches the declared
    ``Number of Seizures in File`` value.

    Args:
        chb_mit_dir: Root of the CHB-MIT dataset (the directory that
            contains ``chb01/``, ``chb02/``, ...).

    Returns:
        Nested dict of the shape::

            {
                "chb01": {
                    "chb01_03.edf": {
                        "no_of_seizures": 1,
                        "seizures": [[2996, 3036]],
                    },
                    ...
                },
                ...
            }

        Only records that appear in the summary file are present. Onsets
        and offsets are integer seconds relative to the start of that
        record.

    Raises:
        ValueError: If a record's declared seizure count does not match the
            parsed start/end pairs, or if start/end lines are interleaved
            in an unexpected order.
    """
    subject_records = {}

    for subject in sorted(chb_mit_dir.iterdir()):
        if not subject.name.startswith("chb"):
            continue

        seizure_info = {}

        with open(subject / f"{subject.name}-summary.txt", "r") as summary_file:
            record = None
            no_of_seizures = None
            seizures_present = []
            current_seizure = []

            def finalize_record():
                """Commit the record currently being parsed and sanity-check it."""
                if record is None:
                    return
                if no_of_seizures is None:
                    raise ValueError(
                        f"Record {record} (subject {subject.name}) is missing "
                        f"'Number of Seizures in File:'"
                    )
                if current_seizure:
                    raise ValueError(
                        f"Record {record} (subject {subject.name}) has a Seizure "
                        f"Start Time without a matching End Time"
                    )
                if len(seizures_present) != no_of_seizures:
                    raise ValueError(
                        f"Record {record} (subject {subject.name}) declares "
                        f"{no_of_seizures} seizures but {len(seizures_present)} "
                        f"were parsed from the summary file"
                    )
                seizure_info[record] = {
                    "no_of_seizures": no_of_seizures,
                    "seizures": sorted(seizures_present), # sort all the seizures present based on thier onset/offset 
                }

            for line in summary_file:

                if line.startswith("File Name:"):
                    # A new record starts here, so commit whatever was being
                    # built for the previous record before resetting state.
                    finalize_record()
                    record = line.split(":")[1].strip()
                    no_of_seizures = None
                    seizures_present = []
                    current_seizure = []
                elif line.startswith("Number of Seizures in File:"):
                    no_of_seizures = int(line.split(":")[1].strip())
                elif "Seizure" in line and "Start Time" in line:
                    # Summary files use both the single-seizure form
                    # ("Seizure Start Time:") and the multi-seizure form
                    # ("Seizure 1 Start Time:"). The substring check matches
                    # both without needing a separate branch.
                    if current_seizure:
                        raise ValueError(
                            f"Record {record} (subject {subject.name}) has two "
                            f"consecutive Seizure Start Times with no End Time between them"
                        )
                    seizure_start = int(line.split(":")[1].strip().split(" ")[0])
                    current_seizure.append(seizure_start)
                elif "Seizure" in line and "End Time" in line:
                    if len(current_seizure) != 1:
                        raise ValueError(
                            f"Record {record} (subject {subject.name}) has a Seizure "
                            f"End Time without a preceding Start Time"
                        )
                    seizure_end = int(line.split(":")[1].strip().split(" ")[0])
                    current_seizure.append(seizure_end)
                    seizures_present.append(current_seizure)
                    current_seizure = []

            # The summary file ends without a trailing "File Name:" line,
            # so the last record is not committed by the loop above — flush
            # it here.
            finalize_record()

        subject_records[subject.name] = seizure_info

    return subject_records


def load_records(chb_mit_dir : Path, allowed_channels : list) -> list:
    """Load every CHB-MIT EDF, harmonise channels, and place each on the subject's timeline.

    Channel harmonisation
    ---------------------
    The CHB-MIT channel set varies across recordings — some files contain
    ECG, VNS markers, or a duplicate ``T8-P8`` column (an archival
    artifact: ``T8-P8-0`` and ``T8-P8-1``). To keep the model input shape
    stable at ``(22, 768)``, this function:

    1. Drops the ``T8-P8-1`` duplicate where present and renames
       ``T8-P8-0`` to ``T8-P8`` so the channel name matches the config.
    2. Marks records still missing any ``allowed_channels`` as
       ``is_valid=False`` and stores them with ``raw=None``. They are kept
       in the returned list (not silently dropped) because the downstream
       interictal buffer is computed across the full chronological
       recording timeline of a subject, including records whose EEG will
       not be used for training.
    3. For valid records, drops every channel outside ``allowed_channels``
       so the surviving channel set is exactly the configured montage.

    Per-subject timeline construction
    ---------------------------------
    Within each subject, records are sorted by EDF ``meas_date`` (filename
    order is not reliable — see the module docstring). A timer initialised
    to ``0.0`` at the subject's first record then accumulates each record's
    duration plus the silent gap to the next record, so that every record
    carries its position on the subject's continuous wall-clock timeline.
    The timer is reset for each new subject because cross-subject
    continuity has no physical meaning.

    Args:
        chb_mit_dir: Root of the CHB-MIT dataset.
        allowed_channels: The fixed 22-channel bipolar montage from
            ``config.yaml`` (``data.chb_mit.allowed_channels``).

    Returns:
        List of dicts, one per ``.edf`` file, each with keys::

            {
                "subject": str,                       # e.g. "chb01"
                "record": str,                        # e.g. "chb01_03.edf"
                "is_valid": bool,                     # True iff all allowed_channels present
                "raw": mne.io.Raw | None,             # None when is_valid is False
                "seizure_info": dict,                 # see load_seizure_annotations
                "record_meas_date": datetime,         # absolute recording start time (from EDF header)
                "record_total_duration": timedelta,   # full record length
                "record_starts_at": float,            # seconds since this subject's first record start
                "record_ends_at": float,              # record_starts_at + record_total_duration
                "buffer": float,                      # silent gap (s) since the previous record; 0.0 for the first
            }

        Entries are ordered by subject, then chronologically (by EDF
        ``meas_date``) within each subject.

    Raises:
        AssertionError: If any record's ``meas_date`` is missing from the
            EDF header, if two consecutive records share a ``meas_date``
            (suggests a duplicate EDF), or if the computed buffer is
            negative (this record overlaps in real time with the previous
            one — the meas_date check alone cannot catch this).
    """
    subject_seizure_annotations = load_seizure_annotations(chb_mit_dir)

    raws = []

    for subject in sorted(chb_mit_dir.iterdir()):
        if not subject.name.startswith("chb"):
            continue

        # Pre-pass: load every EDF for this subject so we can sort by
        # meas_date. Filename order is NOT reliable in CHB-MIT —
        # `chb02_16+.edf` lexically sorts before `chb02_16.edf` despite
        # being recorded ~1 hour later, and `chb03_24/25` are numbered out
        # of chronological order. Sorting by meas_date fixes all such cases
        # uniformly without hardcoding subject-specific swaps.
        records = []
        for record in subject.glob("*.edf"):
            raw = mne.io.read_raw_edf(record)
            # CHB-MIT stated sample freq is 256 Hz for every record
            # Catch off-spec records here rather than late
            assert raw.info['sfreq'] == 256, (
                f"{subject.name}/{record.name}: sample rate is "
                f"{raw.info['sfreq']} Hz, expected 256 Hz"
            )
            assert raw.info["meas_date"] is not None, (
                f"{subject.name}/{record.name} has no meas_date in EDF header"
            )
            records.append((record, raw))
        records.sort(key=lambda pr: pr[1].info["meas_date"])

        # Per-subject timeline state. The timer is reset for each new
        # subject because cross-subject continuity has no physical meaning.
        timer = 0.
        previous_record_flag = False
        prev_record_meas_date = None
        prev_record_total_duration = None

        for record, raw in records:
            # Some CHB-MIT records archive T8-P8 twice (T8-P8-0 and T8-P8-1
            # are bitwise identical). Drop the duplicate and normalise the
            # remaining name to match `allowed_channels`.
            if 'T8-P8-0' in raw.ch_names:
                raw.drop_channels(['T8-P8-1'])
                raw.rename_channels({'T8-P8-0' : 'T8-P8'})

            # Records with no seizures still need a placeholder entry so
            # downstream code can iterate over `seizure_info` uniformly.
            if record.name in subject_seizure_annotations[subject.name]:
                seizure_info = subject_seizure_annotations[subject.name][record.name]
            else:
                seizure_info = {"no_of_seizures": 0, "seizures": []}

            record_meas_date = raw.info["meas_date"]
            record_total_duration = timedelta(seconds=raw.n_times / raw.info['sfreq'])

            # Summary.txt is human-typed and can list seizure offsets past
            # the EDF's actual end (data-entry error or post-hoc EDF
            # truncation). Catch it here rather than later when window
            # slicing tries to read past raw.n_times.
            for seizure_start, seizure_end in seizure_info["seizures"]:
                assert seizure_end <= record_total_duration.total_seconds(), (
                    f"{subject.name}/{record.name}: seizure "
                    f"[{seizure_start}, {seizure_end}] exceeds record "
                    f"duration ({record_total_duration.total_seconds():.1f}s)"
                )

            # Silent gap (s) between the end of the previous record and the
            # start of this one, measured from EDF meas_dates. 0.0 for the
            # first record of each subject.
            buffer = 0.
            if previous_record_flag:
                # Strict-increasing meas_date catches duplicate timestamps
                # (which would otherwise produce buffer = 0 and silently
                # overlap two records on the timeline).
                assert record_meas_date > prev_record_meas_date, (
                    f"{subject.name}/{record.name} shares a meas_date with "
                    f"its previous record — possible duplicate EDF"
                )

                buffer = record_meas_date - prev_record_meas_date
                buffer = buffer.total_seconds() - prev_record_total_duration.total_seconds()

                # ...but a strictly-later meas_date is NOT sufficient: it
                # still allows this record's meas_date to fall inside the
                # previous record's recording window (i.e. the two EDFs
                # overlap in real time). That would produce a negative
                # buffer and silently corrupt the subject's timeline, so
                # assert non-overlap explicitly.
                assert buffer >= 0, (
                    f"{subject.name}/{record.name}: negative buffer "
                    f"({buffer:.1f}s) — meas_date falls inside the previous "
                    f"record's recording window (overlapping EDFs)"
                )

            # Place this record on the subject's continuous timeline, then
            # advance the timer and snapshot the meas_date/duration for
            # the next iteration's buffer computation.
            record_starts_at = timer + buffer
            record_ends_at = record_starts_at + record_total_duration.total_seconds()

            timer = record_ends_at
            previous_record_flag = True
            prev_record_meas_date = record_meas_date
            prev_record_total_duration = record_total_duration

            if not set(allowed_channels).issubset(set(raw.ch_names)):
                # Cannot contribute training windows, but the metadata
                # (especially seizure times) still matters: the 4-hour
                # interictal buffer is computed across all
                # records of a subject, so excluding this entry entirely
                # would corrupt the buffer for its neighbours.
                raws.append({
                    "subject": subject.name,
                    "record": record.name,
                    "is_valid": False,
                    "raw": None,
                    "seizure_info": seizure_info,
                    "record_meas_date": record_meas_date,
                    "record_total_duration": record_total_duration,
                    "record_starts_at": record_starts_at,
                    "record_ends_at": record_ends_at,
                    "buffer": buffer
                })
            else:
                # Strip everything outside the configured montage so every
                # valid record has identical channel ordering and count.
                raw.drop_channels(
                    [ch for ch in raw.ch_names if ch not in allowed_channels]
                )

                raws.append({
                    "subject": subject.name,
                    "record": record.name,
                    "is_valid": True,
                    "raw": raw,
                    "seizure_info": seizure_info,
                    "record_meas_date": record_meas_date,
                    "record_total_duration": record_total_duration,
                    "record_starts_at": record_starts_at,
                    "record_ends_at": record_ends_at,
                    "buffer": buffer
                })

    return raws
