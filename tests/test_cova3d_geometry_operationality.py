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


def test_geometry_historical_audit_firewall():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "classification"
    ] == "REQUIRES_PROSPECTIVE_GEOMETRY_AMENDMENT"

    assert audit[
        "resolvable_aliases"
    ][
        "DIS_equals_FRG"
    ] == 8

    assert audit[
        "resolvable_aliases"
    ][
        "quota2_DIS_equals_FRG"
    ] == 0

    assert audit[
        "case_level"
    ][
        "whole_case_geometry_aliases"
    ] == 2

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


def test_geometry_historical_component_table():

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

    assert int(
        resolvable[
            "dis_equals_frg"
        ].sum()
    ) == 8

    assert int(
        resolvable[
            "any_pairwise_alias"
        ].sum()
    ) == audit[
        "resolvable_aliases"
    ][
        "any_pairwise_alias"
    ]
