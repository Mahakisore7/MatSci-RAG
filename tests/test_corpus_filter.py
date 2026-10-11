from src.rag.corpus_filter import filter_corpus, is_fuel_cell_paper


def test_keeps_fuel_cell_papers():
    assert is_fuel_cell_paper({"title": "Ni-YSZ anodes for SOFCs", "abstract": ""})
    assert is_fuel_cell_paper({"title": "", "abstract": "a solid-oxide electrolysis study"})
    assert is_fuel_cell_paper({"title": "Proton exchange membrane fuel cell", "abstract": ""})


def test_drops_off_topic_papers():
    assert not is_fuel_cell_paper({"title": "Si anodes for lithium-ion batteries", "abstract": ""})
    assert not is_fuel_cell_paper({"title": "Charged cell membrane in electrolyte", "abstract": ""})
    assert not is_fuel_cell_paper({"title": "Fuel-constrained path planning", "abstract": ""})


def test_filter_corpus_keeps_ids():
    corpus = {
        "a": {"title": "SOFC cathodes", "abstract": ""},
        "b": {"title": "Li-S battery", "abstract": ""},
    }
    assert list(filter_corpus(corpus)) == ["a"]
