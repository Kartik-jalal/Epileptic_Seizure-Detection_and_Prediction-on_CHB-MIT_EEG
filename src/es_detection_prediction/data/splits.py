"""Train/val/test split helpers for the curated CHB-MIT pools.

This module is the bridge between the segmentation pools (output of
:func:`find_eligible_records`) and the per-fold Dataset builders
(:func:`build_record_datasets`). It produces **3-way splits** (train,
val, test) shaped as **pool dicts**, ready to feed straight into
``build_record_datasets`` without restructuring.

Two split strategies are provided:

- :func:`split_cross_subject` — holds out **one whole subject** for
  test and another for val. Used for cross-subject training, where the
  model is asked to generalise to a never-seen patient. ``chb12`` and
  ``chb24`` contains no interictal pool and auto-join training as
  ictal-only contributors but cannot be held out for val/test (sensitivity
  needs ictal, FA/h needs interictal — they must have both).
- :func:`split_intra_subject` — picks one patient and holds out one of
  their ictal records + one of their interictal records each for val
  and test. Used for per-patient training where one
  model is trained per patient. Requires the patient to have ≥
  ``minimum_records_per_type`` records of each type (default 3 = 1 train
  + 1 val + 1 test).

Independence of ictal / interictal selection (intra-subject)
------------------------------------------------------------
:func:`split_intra_subject` picks val/test ictal records **independently**
of val/test interictal records — there is no chronological pairing
between them. ``find_eligible_records`` already enforces the n-hour
ictal buffer on every interictal pool entry, so any interictal record
is valid regardless of temporal proximity to a seizure. Forcing
chronological pairing risks val-test brain-state coupling: if val and
test interictal records sat near val and test ictal records in time,
they could end up in the same recording window and val performance
would trivially predict test performance, defeating the early-stopping
signal.
"""

import numpy as np


def split_cross_subject(
    ictal_records: dict,
    interictal_records: dict,
    val_subject: str | None = None,
    test_subject: str | None = None,
    seed: int | None = None,
) -> dict:
    """Single 3-way subject-level holdout: train / val / test by subject.

    Picks one subject for val and one for test (random from eligible
    subjects unless explicitly specified), and assigns everyone else
    to training. Subjects present in only one pool (e.g. ``chb12``,
    ``chb24``) automatically join training as thier specific pool contributors,
    since they can't serve as val/test (val/test need both classes for the
    early-stopping metric and for FA/h).

    Args:
        ictal_records: From :func:`find_eligible_records`. Keyed by
            subject; values are per-subject lists of ictal record dicts.
        interictal_records: Same shape as ``ictal_records``.
        val_subject: If given, use this subject for val. Otherwise
            randomly picked from shared subjects (subject to ``seed``).
        test_subject: If given, use this subject for test. Otherwise
            randomly picked.
        seed: For reproducibility when val/test subjects aren't
            explicitly given. Uses :class:`numpy.random.Generator`.

    Returns:
        Dict with the keys::

            "val_subject":  str       # held out for val
            "test_subject": str       # held out for test
            "training": {"ictal": ictal_pool, "interictal": interictal_pool}
            "val":      {"ictal": ictal_pool, "interictal": interictal_pool}
            "test":     {"ictal": ictal_pool, "interictal": interictal_pool}

    Raises:
        ValueError: If fewer than 3 subjects are present in both pools,
            or if the supplied ``val_subject`` / ``test_subject`` aren't
            in the shared-subjects set, or if they coincide.
    """
    rng = np.random.default_rng(seed)

    # Subjects in BOTH pools — eligible for val/test. Subjects in only
    # one pool (chb12, chb24 on the standard config — no interictal
    # data) become training-only ictal contributors. sorted() makes the
    # subsequent randomisation deterministic from the seed.
    shared_subjects = sorted(ictal_records.keys() & interictal_records.keys())
    training_only_subjects = sorted(ictal_records.keys() ^ interictal_records.keys())

    if len(shared_subjects) < 3:
        raise ValueError(
            f"Need ≥3 subjects present in both pools for train+val+test, "
            f"got {len(shared_subjects)}: {shared_subjects}"
        )

    # Pick val/test subjects when not explicitly given.
    if val_subject is None or test_subject is None:
        permuted = rng.permutation(shared_subjects).tolist()
        if val_subject is None and test_subject is None:
            val_subject, test_subject = permuted[0], permuted[1]
        elif val_subject is None:
            val_subject = next(s for s in permuted if s != test_subject)
        else:  # test_subject is None
            test_subject = next(s for s in permuted if s != val_subject)

    # Validate explicit picks against the shared-subjects set
    if val_subject == test_subject:
        raise ValueError(
            f"val_subject and test_subject must differ, both are {val_subject!r}."
        )
    for role, subj in (("val_subject", val_subject), ("test_subject", test_subject)):
        if subj not in shared_subjects:
            raise ValueError(
                f"{role}={subj!r} is not in the shared-subjects set. "
                f"Eligible: {shared_subjects}."
            )

    train_subjects = [
        s for s in shared_subjects
        if s != val_subject and s != test_subject
    ] + training_only_subjects

    # Pool subset helper: filter a pool dict to the listed subjects.
    # The `if s in pool` guard lets us pass training_only subjects
    # (who appear in ictal_records but not interictal_records) without
    # raising — the resulting interictal pool simply skips them.
    def pool_subset(pool: dict, subjects: list) -> dict:
        return {s: pool[s] for s in subjects if s in pool}

    return {
        "val_subject": val_subject,
        "test_subject": test_subject,
        "training": {
            "ictal":      pool_subset(ictal_records, train_subjects),
            "interictal": pool_subset(interictal_records, train_subjects),
        },
        "val": {
            "ictal":      pool_subset(ictal_records, [val_subject]),
            "interictal": pool_subset(interictal_records, [val_subject]),
        },
        "test": {
            "ictal":      pool_subset(ictal_records, [test_subject]),
            "interictal": pool_subset(interictal_records, [test_subject]),
        },
    }


def split_intra_subject(
    ictal_records: dict,
    interictal_records: dict,
    subject: str | None = None,
    minimum_records_per_type: int = 3,
    seed: int | None = None,
) -> dict:
    """Single 3-way per-patient holdout: train / val / test within ONE patient.

    Picks one patient (random from eligible unless specified), then
    independently picks one ictal record for val and one for test, and
    likewise one interictal record for val and one for test. Training
    gets the remaining records of each type. Both val and test see one
    ictal + one interictal record so the early-stopping signal can
    meaningfully include ictal information.

    Ictal and interictal selection are **independent** (not chronologically
    paired). ``find_eligible_records`` already guarantees buffer safety on 
    every interictal pool entry, so any interictal record is valid regardless
    of when it sits on the subject's timeline.

    Args:
        ictal_records: From :func:`find_eligible_records`. Keyed by
            subject; values are per-subject lists of ictal record dicts.
        interictal_records: Same shape as ``ictal_records``.
        subject: If given, scope to this patient (must be eligible —
            see ``minimum_records_per_type``). Otherwise randomly picks
            an eligible patient.
        minimum_records_per_type: A patient is eligible iff they have ≥
            this many records of **both** ictal and interictal. Default
            3 = 1 train + 1 val + 1 test (tight; 1-record training set
            per class). Bump to 4 or 5 for a healthier training pool.
            On the standard CHB-MIT config this gives roughly:
            ``3 → 19 eligible``, ``4 → 10``, ``5 → 9``.
        seed: For reproducibility when ``subject`` isn't given.

    Returns:
        Dict with the keys::

            "subject":                str   # patient being trained
            "val_ictal_record":       str   # held-out ictal record filename
            "test_ictal_record":      str
            "val_interictal_record":  str   # held-out interictal record filename
            "test_interictal_record": str
            "training": {"ictal": ictal_pool, "interictal": interictal_pool}
            "val":      {"ictal": ictal_pool, "interictal": interictal_pool}
            "test":     {"ictal": ictal_pool, "interictal": interictal_pool}


    Raises:
        ValueError: If ``minimum_records_per_type`` is < 3, if no subject
            meets the eligibility threshold, or if an explicit ``subject``
            isn't eligible.
    """
    if minimum_records_per_type < 3:
        raise ValueError(
            f"minimum_records_per_type must be ≥ 3 (1 train + 1 val + 1 test), "
            f"got {minimum_records_per_type}."
        )

    rng = np.random.default_rng(seed)

    # Eligible patients: present in both pools AND with ≥
    # minimum_records_per_type of each type. sorted() makes the
    # eligible list deterministic so the same seed always picks the
    # same patient.
    eligible_subjects = sorted(
        s for s in ictal_records.keys()
        if s in interictal_records
        and len(ictal_records[s]) >= minimum_records_per_type
        and len(interictal_records[s]) >= minimum_records_per_type
    )

    if not eligible_subjects:
        raise ValueError(
            f"No subject has ≥{minimum_records_per_type} records of both "
            f"ictal and interictal. Either lower minimum_records_per_type "
            f"or inspect the input pools."
        )

    if subject is None:
        # Pick a random eligible subject. permutation + [0] gives a
        # deterministic-from-seed result with explicit intent (no
        # rng.choice with default replacement).
        subject = rng.permutation(eligible_subjects).tolist()[0]
    elif subject not in eligible_subjects:
        raise ValueError(
            f"subject={subject!r} is not eligible (needs ≥"
            f"{minimum_records_per_type} records of each type). "
            f"Eligible subjects: {eligible_subjects}."
        )

    patient_ictal = ictal_records[subject]
    patient_inter = interictal_records[subject]

    # Independent random selection (without replacement, so val ≠ test)
    # of one val + one test record from each type. Permute the indices
    # and take the first two
    ictal_perm = rng.permutation(len(patient_ictal))
    inter_perm = rng.permutation(len(patient_inter))

    val_ictal_idx, test_ictal_idx = int(ictal_perm[0]), int(ictal_perm[1])
    val_inter_idx, test_inter_idx = int(inter_perm[0]), int(inter_perm[1])

    train_ictal = [patient_ictal[i] for i in ictal_perm[2:]]
    train_inter = [patient_inter[i] for i in inter_perm[2:]]
    val_ictal_rec = patient_ictal[val_ictal_idx]
    test_ictal_rec = patient_ictal[test_ictal_idx]
    val_inter_rec = patient_inter[val_inter_idx]
    test_inter_rec = patient_inter[test_inter_idx]


    return {
        "subject": subject,
        "val_ictal_record": val_ictal_rec["record"],
        "test_ictal_record": test_ictal_rec["record"],
        "val_interictal_record": val_inter_rec["record"],
        "test_interictal_record": test_inter_rec["record"],
        "training": {
            "ictal":      {subject: train_ictal},
            "interictal": {subject: train_inter},
        },
        "val": {
            "ictal":      {subject: [val_ictal_rec]},
            "interictal": {subject: [val_inter_rec]},
        },
        "test": {
            "ictal":      {subject: [test_ictal_rec]},
            "interictal": {subject: [test_inter_rec]},
        },
    }
