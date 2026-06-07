"""Dataset-level summary statistics for CHB-MIT loader and segmentation output.

A pure read-only helper that answers the "what's actually in this dataset?"
questions you ask right after loading and again after segmentation.

Loader-output helpers (over the raw record list from ``load_records``):

- :func:`summarize_records` — pure compute. Returns a structured dict
  keyed by ``"valid"`` / ``"invalid"``. No printing, no side effects —
  safe from tests and scripts.
- :func:`print_record_summary` — convenience formatter. Calls
  :func:`summarize_records` and prints three sections to stdout:
  the combined ``valid + invalid`` view ("total available" totals), 
  ``valid only`` ( channel-filtered totals), and ``invalid only``
  (what the channel filter costs).

Segmentation-output helpers (over the ``(ictal_records, interictal_records)``
pair returned by ``find_eligible_records``):

- :func:`summarize_eligible_records` — pure compute. Returns a single
  structured dict with ictal-pool and interictal-pool counts, durations,
  and fittable ``segment_length``-second segment counts, plus the set of
  subjects excluded from the interictal pool by the buffer.
- :func:`print_eligible_records_summary` — convenience formatter. Prints
  the segmentation-pool counts as one block, including just the *names*
  of excluded subjects.
- :func:`print_excluded_subject_details` — diagnostic. For each subject
  excluded from the interictal pool, dumps the full per-record listing
  (seizure count, seizure durations, record duration) so you can see
  *why* the buffer couldn't fit. Kept separate from the summary so the
  default printout doesn't get buried.

None of these run automatically — the loader and segmentation modules stay
focused on their own jobs. Call these from a notebook cell when you want
the view.
"""


def _empty_bucket() -> dict:
    """Fresh stats accumulator.

    Built via a function rather than a module-level constant to avoid the
    classic Python aliasing trap where ``info = {"valid": META, "invalid": META}``
    makes both keys point to the same dict — every update would silently
    mutate both buckets.
    """
    return {
        "ictal_subjects": set(),
        "ictal_records": 0,
        "ictal_events": 0,
        "ictal_duration": 0.0,
        "non_ictal_duration_in_ictal_records": 0.0,
        "interictal_subjects": set(),
        "interictal_records": 0,
        "interictal_duration": 0.0,
    }


def _combine_buckets(*buckets: dict) -> dict:
    """Sum any number of buckets into one (used to derive the combined view).

    Sets are unioned (``|=``) so subject uniqueness is preserved across
    the merge; numeric fields are summed (``+=``).
    """
    out = _empty_bucket()
    for b in buckets:
        out["ictal_subjects"] |= b["ictal_subjects"]
        out["ictal_records"] += b["ictal_records"]
        out["ictal_events"] += b["ictal_events"]
        out["ictal_duration"] += b["ictal_duration"]
        out["non_ictal_duration_in_ictal_records"] += b["non_ictal_duration_in_ictal_records"]
        out["interictal_subjects"] |= b["interictal_subjects"]
        out["interictal_records"] += b["interictal_records"]
        out["interictal_duration"] += b["interictal_duration"]
    return out


def summarize_records(records: list) -> dict:
    """Compute dataset-level statistics over the loader's record list.

    For every record we decide which top-level bucket it falls into based
    on ``record["is_valid"]`` (whether the loader could harmonise it to
    the 22-channel bipolar montage), then update either the ictal or
    interictal sub-bucket depending on whether the record contains any
    seizures.

    Args:
        records: List of record dicts from
            :func:`es_detection_prediction.data.load_records`. Each must
            carry ``subject``, ``is_valid``, ``seizure_info`` and
            ``record_total_duration`` (a ``datetime.timedelta``).

    Returns:
        Dict of the form::

            {
                "valid":   {<see below>},
                "invalid": {<see below>},
            }

        Each bucket has the keys::

            ictal_subjects:                       set[str]  # subjects with >=1 seizure-containing record in this bucket
            ictal_records:                        int       # records containing >=1 seizure
            ictal_events:                         int       # sum of no_of_seizures across ictal records
            ictal_duration:                       float     # sum of seizure (end - start) in seconds
            non_ictal_duration_in_ictal_records:  float     # record duration minus seizure duration, summed over ictal records
            interictal_subjects:                  set[str]  # subjects with >=1 no-seizure record in this bucket
            interictal_records:                   int       # records with no seizures
            interictal_duration:                  float     # full duration of no-seizure records, in seconds

        The "total non-ictal seconds" figure that the literature reports
        is ``interictal_duration + non_ictal_duration_in_ictal_records``
        — i.e. all time that is not inside an annotated seizure interval,
        regardless of which record it sits in.
    """
    info = {"valid": _empty_bucket(), "invalid": _empty_bucket()}

    for record in records:
        # is_valid splits the top-level bucket; the loader-determined flag
        # tells us whether this record's channel set matches the configured
        # 22-channel montage. Invalid records are kept by the loader purely
        # to preserve the per-subject timeline — they can't contribute
        # training segments, but their seizure positions still matter for
        # the 4-hour buffer of their neighbours.
        bucket = info["valid" if record["is_valid"] else "invalid"]
        record_duration_sec = record["record_total_duration"].total_seconds()

        if record["seizure_info"]["no_of_seizures"] > 0:
            bucket["ictal_subjects"].add(record["subject"])
            bucket["ictal_records"] += 1
            bucket["ictal_events"] += record["seizure_info"]["no_of_seizures"]
            seizure_dur = sum(
                end - start for start, end in record["seizure_info"]["seizures"]
            )
            bucket["ictal_duration"] += seizure_dur
            # The non-seizure portion of an ictal record still counts as
            # non-ictal time in the dataset-wide totals (the paper bundles
            # it that way), so track it separately from pure no-seizure
            # records to keep both views available.
            bucket["non_ictal_duration_in_ictal_records"] += record_duration_sec - seizure_dur
        else:
            bucket["interictal_subjects"].add(record["subject"])
            bucket["interictal_records"] += 1
            bucket["interictal_duration"] += record_duration_sec

    return info


def _format_bucket(title: str, bucket: dict) -> str:
    """Render one bucket (valid / invalid / combined) as a printable block.

    Right-aligns numbers in a 12-character column with thousands separators
    so the three views line up visually when stacked.
    """
    total_records = bucket["ictal_records"] + bucket["interictal_records"]
    all_subjects = bucket["ictal_subjects"] | bucket["interictal_subjects"]
    total_non_ictal = (
        bucket["interictal_duration"]
        + bucket["non_ictal_duration_in_ictal_records"]
    )

    lines = [
        title,
        "=" * len(title),
        f"  Total records:                              {total_records:>12,}",
        f"  Total subjects (ictal ∪ interictal):        {len(all_subjects):>12,}",
        "",
        "  Ictal",
        f"    Subjects with seizure-containing records: {len(bucket['ictal_subjects']):>12,}",
        f"    Seizure-containing records:               {bucket['ictal_records']:>12,}",
        f"    Seizure events:                           {bucket['ictal_events']:>12,}",
        f"    Seizure-event duration (sec):             {bucket['ictal_duration']:>12,.1f}",
        "",
        "  Interictal",
        f"    Subjects with no-seizure records:         {len(bucket['interictal_subjects']):>12,}",
        f"    No-seizure records:                       {bucket['interictal_records']:>12,}",
        f"    No-seizure record duration (sec):         {bucket['interictal_duration']:>12,.1f}",
        f"    + non-seizure time in ictal records:      {bucket['non_ictal_duration_in_ictal_records']:>12,.1f}",
        f"    = Total non-ictal duration (sec):         {total_non_ictal:>12,.1f}",
    ]
    return "\n".join(lines)


def print_record_summary(records: list) -> None:
    """Pretty-print dataset stats for the loader output, three views.

    Calls :func:`summarize_records` once over ``records`` and renders three
    sections to stdout:

    1. **Combined (valid + invalid)** — the totals you'd compare against
       the "total available" column of Ali, Angelova, Karmakar (2024)
       Table 1 (the paper's full-dataset headline numbers).
    2. **Valid only** (``is_valid=True``) — what survives the 22-channel
       bipolar-montage filter. Comparable to the paper's "observed" column
       (modulo the slightly different channel-membership rule discussed
       in workflow.md / the T8-P8 rescue note).
    3. **Invalid only** (``is_valid=False``) — records the loader keeps
       for timeline integrity but which cannot contribute training
       segments. Useful to see how much the channel filter is costing you.

    Args:
        records: List of record dicts from
            :func:`es_detection_prediction.data.load_records`.
    """
    info = summarize_records(records)
    combined = _combine_buckets(info["valid"], info["invalid"])

    blocks = [
        _format_bucket("Combined (valid + invalid)", combined),
        _format_bucket("Valid only (is_valid=True)", info["valid"]),
        _format_bucket("Invalid only (is_valid=False)", info["invalid"]),
    ]
    print("\n\n".join(blocks))


# ---------------------------------------------------------------------------
# Segmentation-output helpers
# ---------------------------------------------------------------------------


def summarize_eligible_records(
    ictal_records: dict,
    interictal_records: dict,
    segment_length: float,
) -> dict:
    """Compute stats over the segmentation pools (output of ``find_eligible_records``).

    Segment counts are computed *per* seizure / per interictal interval,
    not by summing durations then dividing — the latter would slightly
    over-count, since a 7-second seizure yields ``7 // 5 = 1`` segment,
    not ``7 / 5 = 1.4``.

    Args:
        ictal_records: From :func:`...segmentation.find_eligible_records`.
            ``{subject: [record, ...]}`` — every record contains >=1 seizure.
        interictal_records: From :func:`...segmentation.find_eligible_records`.
            ``{subject: [record_with_period, ...]}`` — each entry carries
            ``interictal_period_starts_at`` / ``interictal_period_ends_at``.
        segment_length: Window length, in seconds. Comes from
            ``config.yaml`` (``data.chb_mit.segament_window``). Used to
            count how many non-overlapping segments fit into the available
            ictal / interictal duration.

    Returns:
        Dict with the keys::

            ictal_subjects:               set[str]  # subjects with >=1 ictal record
            ictal_records:                int       # total ictal records across subjects
            ictal_events:                 int       # sum of no_of_seizures across ictal records
            ictal_duration:               float     # sum of seizure (end - start) in seconds
            ictal_segments:               int       # sum of (seizure_dur // segment_length) per seizure
            interictal_subjects:          set[str]  # subjects with >=1 valid interictal record
            interictal_records:           int       # total interictal records across subjects
            interictal_duration:          float     # sum of trimmed interictal-period lengths in seconds
            interictal_segments:          int       # sum of (period_dur // segment_length) per record
            excluded_interictal_subjects: set[str]  # in ictal pool but not in interictal pool
    """
    ictal_subjects: set = set()
    n_ictal_records = 0
    n_ictal_events = 0
    ictal_dur = 0.0
    ictal_segs = 0

    for subject, recs in ictal_records.items():
        ictal_subjects.add(subject)
        n_ictal_records += len(recs)
        for record in recs:
            n_ictal_events += record["seizure_info"]["no_of_seizures"]
            for seizure_start, seizure_end in record["seizure_info"]["seizures"]:
                dur = seizure_end - seizure_start
                ictal_dur += dur
                # Per-seizure integer segment count — workflow.md §4
                # requires the 3-s (or 5-s) window to lie *fully inside*
                # the seizure interval, so a 7-s seizure yields 1 segment,
                # not 1.4.
                ictal_segs += int(dur // segment_length)

    interictal_subjects: set = set()
    n_interictal_records = 0
    interictal_dur = 0.0
    interictal_segs = 0

    for subject, recs in interictal_records.items():
        interictal_subjects.add(subject)
        n_interictal_records += len(recs)
        for record in recs:
            dur = (
                record["interictal_period_ends_at"]
                - record["interictal_period_starts_at"]
            )
            interictal_dur += dur
            interictal_segs += int(dur // segment_length)

    # "Excluded" = has seizure-containing records (so the subject's
    # timeline contains seizures the buffer must respect) but no record
    # whose trimmed interictal period survives. Mirrors the notebook's
    # `ictal.keys() - interictal.keys()` check.
    excluded = ictal_subjects - interictal_subjects

    return {
        "ictal_subjects": ictal_subjects,
        "ictal_records": n_ictal_records,
        "ictal_events": n_ictal_events,
        "ictal_duration": ictal_dur,
        "ictal_segments": ictal_segs,
        "interictal_subjects": interictal_subjects,
        "interictal_records": n_interictal_records,
        "interictal_duration": interictal_dur,
        "interictal_segments": interictal_segs,
        "excluded_interictal_subjects": excluded,
    }


def print_eligible_records_summary(
    ictal_records: dict,
    interictal_records: dict,
    segment_length: float,
) -> None:
    """Pretty-print segmentation-pool stats in one block.

    Includes the *names* of subjects excluded from the interictal pool
    (because their seizures are too dense for the buffer to fit) but NOT
    the per-record breakdown — for that, call
    :func:`print_excluded_subject_details` separately.

    Args:
        ictal_records: From :func:`...segmentation.find_eligible_records`.
        interictal_records: From :func:`...segmentation.find_eligible_records`.
        segment_length: Window length in seconds (see
            :func:`summarize_eligible_records`).
    """
    info = summarize_eligible_records(
        ictal_records, interictal_records, segment_length
    )

    total_records = info["ictal_records"] + info["interictal_records"]
    excluded = sorted(info["excluded_interictal_subjects"])
    excluded_str = ", ".join(excluded) if excluded else "(none)"

    # Render segment_length as a clean integer in the labels when it is one
    # (5.0 → "5") so the text isn't peppered with stray ".0"s.
    seg_label = (
        f"{int(segment_length)}"
        if float(segment_length).is_integer()
        else f"{segment_length}"
    )

    title = f"Eligible records (channel + ictal-buffer filter, {seg_label}-s segments)"
    lines = [
        title,
        "=" * len(title),
        f"  Total records:                              {total_records:>12,}",
        "",
        "  Ictal",
        f"    Subjects:                                 {len(info['ictal_subjects']):>12,}",
        f"    Records:                                  {info['ictal_records']:>12,}",
        f"    Seizure events:                           {info['ictal_events']:>12,}",
        f"    Seizure-event duration (sec):             {info['ictal_duration']:>12,.1f}",
        f"    Fittable {seg_label}-s segments:                    {info['ictal_segments']:>12,}",
        "",
        "  Interictal (after ictal-buffer trim)",
        f"    Subjects:                                 {len(info['interictal_subjects']):>12,}",
        f"    Records:                                  {info['interictal_records']:>12,}",
        f"    Interictal-period duration (sec):         {info['interictal_duration']:>12,.1f}",
        f"    Fittable {seg_label}-s segments:                    {info['interictal_segments']:>12,}",
        "",
        f"  Subjects excluded from interictal pool ({len(excluded)}): {excluded_str}",
    ]
    print("\n".join(lines))


def print_excluded_subject_details(
    records: list,
    ictal_records: dict,
    interictal_records: dict,
) -> None:
    """Diagnostic dump for subjects excluded from the interictal pool.

    For each subject in ``ictal_records.keys() - interictal_records.keys()``,
    walks the full loader ``records`` list (including invalid ones, since
    the buffer math considers them too) and prints every record's seizure
    count, per-seizure durations, and total record duration. Useful to see
    *why* the 4-hour buffer couldn't fit anywhere on the subject's
    timeline — typically because seizure-containing records are spaced
    closer than the buffer width.

    Args:
        records: The full record list from
            :func:`...dataloader.load_records`. Needed because excluded
            subjects may not appear at all in ``interictal_records``, and
            their full record list (valid + invalid) tells the story.
        ictal_records: From :func:`...segmentation.find_eligible_records`.
        interictal_records: From :func:`...segmentation.find_eligible_records`.
    """
    excluded = sorted(set(ictal_records.keys()) - set(interictal_records.keys()))

    if not excluded:
        print("No subjects excluded from the interictal pool.")
        return

    for sub in excluded:
        print(f"\nSubject {sub} — no valid interictal period:")
        for record in records:
            if record["subject"] != sub:
                continue
            durations = [
                end - start for start, end in record["seizure_info"]["seizures"]
            ]
            print(
                f"  {record['record']:<22}"
                f" seizures={record['seizure_info']['no_of_seizures']:>2}"
                f" durations={durations}"
                f" record_duration={record['record_total_duration']}"
            )
