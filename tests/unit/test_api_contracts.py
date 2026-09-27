import pytest
from pydantic import ValidationError

from smallforge.api.contracts import ExperimentRequest


def test_experiment_request_rejects_duplicate_modes() -> None:
    with pytest.raises(ValidationError):
        ExperimentRequest(
            model_id="ollama:tiny",
            benchmark_id="core-v1",
            modes=["baseline", "baseline"],
        )


def test_experiment_request_has_comparison_defaults() -> None:
    request = ExperimentRequest(model_id="ollama:tiny", benchmark_id="core-v1")
    assert request.modes == ["baseline", "full"]
