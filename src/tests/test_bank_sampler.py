import tempfile

from src.env.puzzle_bank import PuzzleBank, PuzzleMetadata, SpecKey, SolverKey, BankSampler


def _make_metadata(spec_key: SpecKey, seed: int, optimal_length: int, robots_moved: int) -> PuzzleMetadata:
    # Minimal, consistent metadata for testing; hashes can be arbitrary non-empty bytes
    return PuzzleMetadata(
        seed=seed,
        spec_key=spec_key,
        solver_key=SolverKey(),
        board_hash=seed.to_bytes(32, "big"),
        optimal_length=optimal_length,
        robots_moved=robots_moved,
        nodes_expanded=0,
        solve_time_ms=1,
        solved_within_limits=True,
        hit_depth_limit=False,
        hit_node_limit=False,
        actions_encoded=[],
        created_at=0.0,
        run_id="test",
        board_seed=seed,
        layout_hash=(seed + 1).to_bytes(32, "big"),
        layout_blob=None,
    )


def test_sample_puzzles_stratified_allocates_by_weights():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(height=16, width=16, num_robots=4, edge_t_per_quadrant=2, central_l_per_quadrant=2)

        # Add a small pool across optimal_length bands 1,2,3
        puzzles = []
        for i in range(20):
            puzzles.append(_make_metadata(spec, seed=1000 + i, optimal_length=1, robots_moved=1))
        for i in range(10):
            puzzles.append(_make_metadata(spec, seed=2000 + i, optimal_length=2, robots_moved=1))
        for i in range(5):
            puzzles.append(_make_metadata(spec, seed=3000 + i, optimal_length=3, robots_moved=1))
        bank.add_puzzles(puzzles)

        sampler = BankSampler(bank)
        bands = [
            {"min_optimal_length": 1, "max_optimal_length": 1, "min_robots_moved": 1, "max_robots_moved": 1, "weight": 1.0},
            {"min_optimal_length": 2, "max_optimal_length": 2, "min_robots_moved": 1, "max_robots_moved": 1, "weight": 2.0},
            {"min_optimal_length": 3, "max_optimal_length": 3, "min_robots_moved": 1, "max_robots_moved": 1, "weight": 3.0},
        ]

        # Request 12 puzzles; expected allocation roughly 2:4:6 by weights
        samples = sampler.sample_puzzles_stratified(spec_key=spec, total_count=12, bands=bands, random_seed=42)
        assert len(samples) == 12
        plan = sampler.get_last_stratified_plan()
        assert plan is not None
        # Verify requested distribution aligns with weights
        requested = [b["requested"] for b in plan["bands"]]
        assert requested == [2, 4, 6]
        # Verify we sampled at most requested per band
        for b in plan["bands"]:
            assert 0 <= b["sampled"] <= b["requested"]


def test_stratified_respects_bounds_and_manifest_histogram_updates():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(height=16, width=16, num_robots=4, edge_t_per_quadrant=2, central_l_per_quadrant=2)

        # Add only optimal_length=2 entries so band for ol=3 should sample zero
        puzzles = [_make_metadata(spec, seed=5000 + i, optimal_length=2, robots_moved=1) for i in range(3)]
        bank.add_puzzles(puzzles)

        # Check manifest histogram key exists and counts are reflected
        part_key = spec.to_partition_key()
        hist = bank.manifest["partitions"][part_key]["histogram"]["by_optimal_length_and_robots_moved"]
        assert hist.get("ol=2|rm=1", 0) >= 3

        sampler = BankSampler(bank)
        bands = [
            {"min_optimal_length": 2, "max_optimal_length": 2, "min_robots_moved": 1, "max_robots_moved": 1},
            {"min_optimal_length": 3, "max_optimal_length": 3, "min_robots_moved": 1, "max_robots_moved": 1},
        ]
        samples = sampler.sample_puzzles_stratified(spec_key=spec, total_count=4, bands=bands, random_seed=7)
        assert len(samples) <= 4
        plan = sampler.get_last_stratified_plan()
        # The second band (ol=3) should have sampled zero
        assert plan["bands"][1]["sampled"] == 0


def test_stratified_falls_back_to_other_bands_when_requested_band_empty():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(height=16, width=16, num_robots=4, edge_t_per_quadrant=2, central_l_per_quadrant=2)

        # Populate only the first band's bounds
        puzzles = [
            _make_metadata(spec, seed=6000 + i, optimal_length=1, robots_moved=1)
            for i in range(5)
        ]
        bank.add_puzzles(puzzles)

        sampler = BankSampler(bank)
        bands = [
            {"min_optimal_length": 1, "max_optimal_length": 1, "min_robots_moved": 1, "max_robots_moved": 1, "weight": 0.0},
            {"min_optimal_length": 2, "max_optimal_length": 2, "min_robots_moved": 1, "max_robots_moved": 1, "weight": 1.0},
        ]

        samples = sampler.sample_puzzles_stratified(spec_key=spec, total_count=1, bands=bands, random_seed=13)
        assert len(samples) == 1

        plan = sampler.get_last_stratified_plan()
        assert plan is not None
        # Second band is empty but was requested
        assert plan["bands"][1]["requested"] == 1
        assert plan["bands"][1]["sampled"] == 0
        # First band should report the redistributed draw
        assert plan["bands"][0].get("extra_sampled") == 1


def test_sampler_seed_replays_and_hash_splits_are_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(16, 16, 4, 2, 2)
        bank.add_puzzles([
            _make_metadata(spec, 10_000 + i, optimal_length=1, robots_moved=1)
            for i in range(100)
        ])

        sampler = BankSampler(bank, seed=123)
        first = [sampler.sample_puzzle(spec).seed for _ in range(12)]
        sampler.reseed(123)
        replay = [sampler.sample_puzzle(spec).seed for _ in range(12)]
        assert replay == first

        train = BankSampler(bank, split="train", eval_fraction=0.25, split_seed=9)
        evaluation = BankSampler(bank, split="eval", eval_fraction=0.25, split_seed=9)
        train_seeds = {p.seed for p in train.sample_puzzles(spec, 100)}
        eval_seeds = {p.seed for p in evaluation.sample_puzzles(spec, 100)}

        assert train_seeds
        assert eval_seeds
        assert train_seeds.isdisjoint(eval_seeds)
        assert train_seeds | eval_seeds == {10_000 + i for i in range(100)}
        assert [p.seed for p in train.list_puzzles(spec)] == sorted(
            train_seeds,
            key=lambda seed: ((seed + 1).to_bytes(32, "big"), seed.to_bytes(32, "big"), seed),
        )


def test_repeated_single_draw_stratification_advances_and_reseeds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(16, 16, 4, 2, 2)
        bank.add_puzzles([
            _make_metadata(spec, length * 1_000 + i, optimal_length=length, robots_moved=1)
            for length in range(1, 5)
            for i in range(8)
        ])
        bands = [
            {
                "min_optimal_length": length,
                "max_optimal_length": length,
                "min_robots_moved": 1,
                "max_robots_moved": 1,
                "weight": 1.0,
            }
            for length in range(1, 5)
        ]
        sampler = BankSampler(bank, seed=123)

        first = [
            sampler.sample_puzzles_stratified(spec, total_count=1, bands=bands)[0].optimal_length
            for _ in range(24)
        ]
        sampler.reseed(123)
        replay = [
            sampler.sample_puzzles_stratified(spec, total_count=1, bands=bands)[0].optimal_length
            for _ in range(24)
        ]

        assert len(set(first)) > 1
        assert replay == first


def test_sampler_resume_state_replays_cached_and_stratified_streams() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(16, 16, 4, 2, 2)
        bank.add_puzzles([
            _make_metadata(spec, length * 10_000 + i, optimal_length=length, robots_moved=1)
            for length in range(1, 4)
            for i in range(20)
        ])
        bands = [
            {
                "min_optimal_length": length,
                "max_optimal_length": length,
                "min_robots_moved": 1,
                "max_robots_moved": 1,
                "weight": float(length),
            }
            for length in range(1, 4)
        ]
        original = BankSampler(bank, seed=77)
        for _ in range(7):
            original.sample_puzzle(spec, min_optimal_length=1, max_optimal_length=2)
            original.sample_puzzles_stratified(spec, total_count=1, bands=bands)
        state = original.get_resume_state()
        expected_plain = [
            original.sample_puzzle(spec, min_optimal_length=1, max_optimal_length=2).seed
            for _ in range(25)
        ]
        expected_stratified = [
            original.sample_puzzles_stratified(spec, total_count=1, bands=bands)[0].seed
            for _ in range(25)
        ]

        resumed = BankSampler(bank, seed=77)
        resumed.set_resume_state(state)
        actual_plain = [
            resumed.sample_puzzle(spec, min_optimal_length=1, max_optimal_length=2).seed
            for _ in range(25)
        ]
        actual_stratified = [
            resumed.sample_puzzles_stratified(spec, total_count=1, bands=bands)[0].seed
            for _ in range(25)
        ]
        assert actual_plain == expected_plain
        assert actual_stratified == expected_stratified


def test_append_invalidates_partition_read_cache():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(16, 16, 4, 2, 2)
        bank.add_puzzles([_make_metadata(spec, 1, 1, 1)])
        assert len(bank.get_partition_dataframe(spec)) == 1

        bank.add_puzzles([_make_metadata(spec, 2, 1, 1)])

        assert len(bank.get_partition_dataframe(spec)) == 2


def test_layout_split_refuses_rows_without_layout_hash():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PuzzleBank(tmp)
        spec = SpecKey(16, 16, 4, 2, 2)
        metadata = _make_metadata(spec, 1, 1, 1)
        metadata.layout_hash = None
        bank.add_puzzles([metadata])

        sampler = BankSampler(bank, split="train", split_key="layout_hash")
        try:
            sampler.sample_puzzle(spec)
        except ValueError as exc:
            assert "requires a valid layout_hash" in str(exc)
        else:
            raise AssertionError("missing layout_hash should not silently fall back")
