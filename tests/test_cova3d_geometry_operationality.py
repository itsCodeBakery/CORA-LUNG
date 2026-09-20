from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_decision_rule_is_strict():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_geometry_operationality_decision_rule_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    req = cfg[
        "strict_pass_requirements"
    ]

    assert req[
        "resolvable_COH_equals_DIS"
    ] == 0

    assert req[
        "resolvable_COH_equals_FRG"
    ] == 0

    assert req[
        "resolvable_DIS_equals_FRG"
    ] == 0

    assert req[
        "whole_case_geometry_aliases"
    ] == 0


def test_geometry_audit_firewall():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    firewall = audit[
        "firewall"
    ]

    assert firewall[
        "CT_arrays_accessed"
    ] == 0

    assert firewall[
        "dense_lesion_masks_accessed"
    ] == 0

    assert firewall[
        "dense_lung_masks_accessed"
    ] == 0

    assert firewall[
        "predictions_accessed"
    ] == 0

    assert firewall[
        "checkpoints_accessed"
    ] == 0

    assert firewall[
        "optimizer_steps"
    ] == 0

    assert firewall[
        "final_outer_cv_outcome_access"
    ] == 0

    assert audit[
        "authorization_after_diagnostic"
    ][
        "factorial_training"
    ] is False


def test_geometry_component_table_consistent_with_classification():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_geometry_operationality_component_audit_v1_0.csv"
    )

    resolvable = frame[
        frame[
            "resolution_class"
        ]
        == "GLOBAL_GEOMETRY_RESOLVABLE"
    ]

    aliases = int(
        resolvable[
            "any_pairwise_alias"
        ].sum()
    )

    assert aliases == audit[
        "resolvable_aliases"
    ][
        "any_pairwise_alias"
    ]

    classification = audit[
        "classification"
    ]

    if classification == "PASS_STRICT_GEOMETRY_OPERATIONALITY":

        assert aliases == 0

        assert audit[
            "case_level"
        ][
            "whole_case_geometry_aliases"
        ] == 0

        assert audit[
            "structural_checks"
        ][
            "limited_identity_failures"
        ] == 0

        assert audit[
            "structural_checks"
        ][
            "resolvable_topology_failures"
        ] == 0

        assert audit[
            "authorization_after_diagnostic"
        ][
            "dense_sanity_evaluation"
        ] is True

        assert audit[
            "next_block"
        ] == "09E-SANITY-EVAL"

    else:

        assert (
            aliases > 0
            or audit[
                "case_level"
            ][
                "whole_case_geometry_aliases"
            ] > 0
            or audit[
                "structural_checks"
            ][
                "limited_identity_failures"
            ] > 0
            or audit[
                "structural_checks"
            ][
                "resolvable_topology_failures"
            ] > 0
            or audit[
                "structural_checks"
            ][
                "geometry_claim_flag_failures"
            ] > 0
        )

        assert audit[
            "authorization_after_diagnostic"
        ][
            "dense_sanity_evaluation"
        ] is False

        assert audit[
            "next_block"
        ] == "09E-GEOM-OP-REPAIR"


def test_state_matches_geometry_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

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
    ] == "09E-GEOM-OP-DIAG"

    assert state[
        "geometry_operationality_status"
    ] == audit[
        "classification"
    ]

    assert state[
        "dense_outcomes_opened_in_cova3d"
    ] is False

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "final_outer_cv_access_in_cova3d"
    ] == 0
