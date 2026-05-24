"""Read CHB-MIT EEG records and their seizure annotations.

This module is the entry point of the data pipeline. It parses each subject's
``chbXX-summary.txt`` to recover the ground-truth seizure intervals, loads the
matching ``.edf`` recordings via MNE, and harmonises the channel set to the fixed
22-channel bipolar montage defined in ``config.yaml``.

Records missing any of the required channels are still returned (marked
``is_valid=False``) because their seizure positions are needed downstream
when enforcing the 4-hour interictal buffer across neighbouring records from the same subject.
"""

import mne
from pathlib import Path
from datetime import timedelta



def load_seizure_annotations(chb_mit_dir : Path) -> dict:
    """Parse the per-subject ``*-summary.txt`` files for seizure annotations.

    Each CHB-MIT subject directory contains a ``chbXX-summary.txt`` listing
    every recording for that subject and, for each recording, the number of
    seizures and the onset/offset times (in seconds from the start of that
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
                    "seizures": seizures_present,
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
    """Load every CHB-MIT EDF record and tag it with its seizure annotations.

    The CHB-MIT channel set varies across recordings — some files contain
    ECG, VNS markers, or a duplicate ``T8-P8`` column (an archival
    artifact: ``T8-P8-0`` and ``T8-P8-1``). To keep the model input shape
    stable at ``(22, 768)``, this function:

    1. Drops the ``T8-P8-1`` duplicate where present and renames
       ``T8-P8-0`` to ``T8-P8`` so the channel name matches the config.
    2. Marks records still missing any ``allowed_channels`` as
       ``is_valid=False`` and stores them with ``raw=None``. They are kept
       in the returned list (not silently dropped) because the downstream
       interictal buffer is computed across the full
       chronological recording timeline of a subject, including records
       whose EEG will not be used for training.
    3. For valid records, drops every channel outside ``allowed_channels``
       so the surviving channel set is exactly the configured montage.

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
                "record_meas_date": datetime,         # absolute recording start time
                "record_total_duration": timedelta,   # full record length
            }

        Entries are ordered by subject then by sorted EDF filename, which
        mirrors the chronological order in which CHB-MIT records were
        collected.
    """
    subject_records = load_seizure_annotations(chb_mit_dir)

    raws = []

    for subject in sorted(chb_mit_dir.iterdir()):
        if not subject.name.startswith("chb"):
            continue

        for record in sorted(subject.glob("*.edf")):
            raw = mne.io.read_raw_edf(record)

            # Some CHB-MIT records archive T8-P8 twice (T8-P8-0 and T8-P8-1
            # are bitwise identical). Drop the duplicate and normalise the
            # remaining name to match `allowed_channels`.
            if 'T8-P8-0' in raw.ch_names:
                raw.drop_channels(['T8-P8-1'])
                raw.rename_channels({'T8-P8-0' : 'T8-P8'})

            # Records with no seizures still need a placeholder entry so
            # downstream code can iterate over `seizure_info` uniformly.
            if record.name in subject_records[subject.name]:
                seizure_info = subject_records[subject.name][record.name]
            else:
                seizure_info = {"no_of_seizures": 0, "seizures": []}

            record_meas_date = raw.info["meas_date"]
            record_total_duration = timedelta(seconds=raw.n_times / raw.info['sfreq'])

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
                    "record_total_duration": record_total_duration
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
                    "record_total_duration": record_total_duration
                })

    return raws
