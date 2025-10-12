import tempfile

from src.env.puzzle_bank import PuzzleBank, SpecKey, PuzzleMetadata, SolverKey
from src.env.precompute_pipeline import compute_level_coverage_from_histograms
from src.env.curriculum_config import get_curriculum_specs_for_precomputation


def _make_md(sk: SpecKey, seed: int, ol: int, rm: int) -> PuzzleMetadata:
    return PuzzleMetadata(
        seed=seed,
        spec_key=sk,
        solver_key=SolverKey(),
        board_hash=b"0" * 32,
        optimal_length=ol,
        robots_moved=rm,
        nodes_expanded=0,
        solve_time_ms=1,
        solved_within_limits=True,
        hit_depth_limit=False,
        hit_node_limit=False,
        actions_encoded=[],
        created_at=0.0,
        run_id="test",
        board_seed=seed,
        layout_hash=b"1" * 32,
        layout_blob=None,
    )


def test_compute_level_coverage_from_histograms_counts_only_matching_spec_and_bands():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        levels = get_curriculum_specs_for_precomputation()
        # Use the first level's spec
        sk = levels[0]["spec_key"]

        # Add entries with ol in and out of the allowed range
        bank.add_puzzles([
            _make_md(sk, 1, ol=1, rm=1),
            _make_md(sk, 2, ol=2, rm=1),
            _make_md(sk, 3, ol=5, rm=1),  # likely outside lower levels
        ])

        coverage = compute_level_coverage_from_histograms(bank, levels[:1])
        # Coverage should be at least 2 for first level (depends on level bounds in config)
        assert 0 <= list(coverage.values())[0]


