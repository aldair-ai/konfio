"""The frozen selected_model section is the only route to the final pipeline."""

import copy

import pytest

from intent.data import load_config
from intent.models import final_config, selected_rules, selected_weight_grid

CFG = load_config()


def test_frozen_section_matches_artifact_settings() -> None:
    view, sm = final_config(CFG)
    assert sm["name"] == "E9b" and view["finetune"]["lr"] == sm["finetuned_head"]["lr"]
    assert selected_rules(sm).at_least_one and selected_rules(sm).no_exclusive
    assert len(selected_weight_grid(sm)) == 21


@pytest.mark.parametrize("mutate", [
    lambda c: c["selected_model"].update(name="E7"),
    lambda c: c["selected_model"].update(experiment="E7_hybrid_lgbm"),
    lambda c: c["selected_model"]["finetuned_head"].update(lr=3e-5),
    lambda c: c["selected_model"]["tfidf_lr"]["tfidf"].update(min_df=1),
    lambda c: c["selected_model"]["thresholds"].update(method="fixed_0.5"),
])
def test_any_other_configuration_is_refused(mutate) -> None:
    cfg = copy.deepcopy(CFG)
    mutate(cfg)
    with pytest.raises(ValueError):
        final_config(cfg)
