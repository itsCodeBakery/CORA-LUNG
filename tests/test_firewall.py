"""
Dense lesion masks must never appear in training batches.
"""

import pytest


@pytest.mark.skip(reason="Activated when implementation reaches this stage.")
def test_scientific_contract_placeholder():
    assert True
