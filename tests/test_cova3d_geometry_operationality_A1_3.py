from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_A1_3_protocol():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_protocol_amendment_A1_3_operational_neutralization.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert cfg[
        "status"
    ] == "PROSPECTIVELY_FROZEN_BEFORE_DENSE_OUTCOME_OPENING"

    assert cfg[
        "amendment_id"
    ] == "A1.3_OPERATIONAL_GEOMETRY_NEUTRALIZATION"

    assert cfg[
        "trigger_evidence"
    ][
        "unique_alias_physical_components"
    ] == 8

    assert cfg[
        "trigger_evidence"
    ][
        "quota2_DIS_equals_FRG"
    ] == 0

    assert cfg[
        "neutralization_rule"
    ][
        "coverage_selection_changed"
    ] is False

    assert cfg[
        "neutralization_rule"
    ][
        "per_component_quota_changed"
    ] is False

    assert cfg[
        "neutralization_rule"
    ][
        "patient_FG_budget_changed"
    ] is False

    assert cfg[
        "neutralization_rule"
    ][
        "background_changed"
    ] is False


def test_A1_3_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_repair_A1_3.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "status"
    ] == "PASS"

    assert audit[
        "post_repair_operationality"
    ][
        "operational_component_aliases"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "neutral_identity_failures"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "operational_topology_failures"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "informative_whole_case_aliases"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "strict_pass"
    ] is True

    assert audit[
        "authorization"
    ][
        "dense_sanity_evaluation"
    ] is True

    assert audit[
        "authorization"
    ][
        "factorial_training"
    ] is False


def test_A1_3_R2_compatibility():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_A1_3_R2_development_compatibility_v1_0.csv"
    )

    assert len(frame) == 24

    assert frame[
        "exact_sparse_arrays_equal"
    ].all()

    assert frame[
        "annotation_semantic_hash_equal"
    ].all()


def test_A1_3_artifact_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_candidate_artifact_manifest_v1_3.csv"
    )

    assert len(frame) == 120

    assert frame[
        "sparse_only"
    ].all()

    assert frame[
        "coordinate_level_QA"
    ].all()

    assert frame[
        "trainer_candidate"
    ].all()

    assert (
        ~frame[
            "training_authorized"
        ]
    ).all()


def test_A1_3_operational_classes():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_operational_geometry_class_v1_0.csv"
    )

    counts = (
        frame[
            "A1_3_operational_geometry_class"
        ]
        .value_counts()
        .to_dict()
    )

    assert counts[
        "GLOBAL_OPERATIONALLY_NEUTRAL_ALIAS"
    ] == 8

    assert (
        sum(
            counts.values()
        )
        == len(frame)
    )


def test_A1_3_current_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "last_completed_block"
    ] == "09E-GEOM-OP-REPAIR"

    assert state[
        "A1_3_status"
    ] == "FROZEN_PASS"

    assert state[
        "annotation_methodology_structurally_ready"
    ] is True

    assert state[
        "A1_3_R2_compatibility"
    ] == "PASS"

    assert state[
        "sanity_fit_R2_retraining_required_after_A1_3"
    ] is False

    assert state[
        "dense_outcomes_opened_in_cova3d"
    ] is False

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "next_block"
    ] == "09E-SANITY-EVAL"
