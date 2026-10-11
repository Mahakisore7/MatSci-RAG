from scripts.benchmark_questions import BENCHMARK_V1, BENCHMARK_V2, BENCHMARKS


def test_v2_extends_v1():
    assert BENCHMARK_V2[: len(BENCHMARK_V1)] == BENCHMARK_V1
    assert len(BENCHMARK_V1) == 7
    assert len(BENCHMARK_V2) == 28


def test_no_duplicate_questions():
    assert len(set(BENCHMARK_V2)) == len(BENCHMARK_V2)


def test_registry_and_question_shape():
    assert set(BENCHMARKS) == {"v1", "v2"}
    assert all(q.endswith("?") and len(q) < 1000 for q in BENCHMARK_V2)
