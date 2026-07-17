from src.scripts.run_phase2_benchmark import _format_completion_summary


def test_completion_summary_labels_validation_only_result() -> None:
    result = {
        "seeds": [{
            "selected_checkpoint_timesteps": 10,
            "test_selected": None,
            "curve": [{
                "actual_timesteps": 10,
                "deterministic": {"success_rate": 0.4},
                "stochastic": {"success_rate": 0.5},
            }],
        }],
        "aggregate": {
            "deterministic": {"mean": 0.0},
            "stochastic": {"mean": 0.0},
        },
    }

    summary = _format_completion_summary(result)

    assert summary == (
        "Final test sealed; selected validation success across seeds: "
        "deterministic=40.0%, stochastic=50.0%"
    )


def test_completion_summary_uses_test_aggregate_when_test_ran() -> None:
    result = {
        "seeds": [{"test_selected": {"deterministic": {}, "stochastic": {}}}],
        "aggregate": {
            "deterministic": {"mean": 0.25},
            "stochastic": {"mean": 0.3},
        },
    }

    assert _format_completion_summary(result) == (
        "Held-out success across seeds: deterministic=25.0%, stochastic=30.0%"
    )
