"""
Benchmark question sets for scripts/evaluate_ragas.py (ADR-018).

V1 is the original 7-question set used for every result up to ADR-017.
V2 keeps those 7 and adds 21 more, each written against a topic that at
least one paper in data/sofc_corpus.jsonl covers (read from titles and
abstracts, not from the retrieval output, so the set is not tuned to what
the retriever already finds). It spans SOFC materials, microstructure,
modelling, SOEC, proton-conducting oxides and PEM fuel cells.
"""

BENCHMARK_V1 = [
    "What materials are used as electrolytes in solid oxide fuel cells?",
    "What is the effect of temperature on SOFC performance?",
    "Which cathode materials show high oxygen reduction activity in SOFCs?",
    "What causes anode degradation in Ni-YSZ SOFCs?",
    "How does grain size affect the ionic conductivity of ceria?",
    "What is the role of GDC as an interlayer in SOFC cathodes?",
    "What sintering temperatures are typically used for LSCF cathodes?",
]

_V2_EXTRA = [
    # SOFC materials
    "Why are high-entropy oxides studied for solid oxide fuel cells?",
    "How does yttrium doping affect oxygen conductivity in barium-based perovskite cathodes?",
    "What are the advantages of double perovskite cathodes for intermediate-temperature SOFCs?",
    "How is yttria-stabilized zirconia electrolyte fabricated as a thin dense layer?",
    "What happens at the interface between gadolinia-doped ceria and yttria-stabilized zirconia?",
    "What makes a cobalt-free cathode attractive for proton-conducting fuel cells?",
    # microstructure and degradation
    "How does anode microstructure change over time in a solid oxide fuel cell?",
    "How can 3D imaging be used to characterise SOFC electrode microstructure?",
    "How does ceramic film cracking occur in solid oxide fuel cells?",
    "How does sulfur poisoning affect a PEM fuel cell electrode?",
    "What is the effect of ammonia on proton exchange membrane fuel cell performance?",
    # modelling and diagnostics
    "How is electrochemical impedance spectroscopy used to analyse fuel cell electrodes?",
    "What transport model is used for hydrogen and water in porous SOFC anodes?",
    "How does machine learning predict SOFC electrode polarization resistance?",
    "What does the distribution of relaxation times reveal in PEM fuel cell impedance?",
    # electrolysis and operation
    "What limits the durability of solid oxide electrolysis cells under dynamic load cycling?",
    "How is oxygen partial pressure distributed across a two-layer electrolyte in an SOEC?",
    "How do grain boundaries influence conductivity in solid oxide electrolytes?",
    # PEM fuel cells
    "How does water management affect PEM fuel cell performance?",
    "How does the gas diffusion layer thickness influence PEM fuel cell performance?",
    "What causes carbon corrosion in a polymer electrolyte fuel cell cathode?",
]

BENCHMARK_V2 = BENCHMARK_V1 + _V2_EXTRA

BENCHMARKS = {"v1": BENCHMARK_V1, "v2": BENCHMARK_V2}
