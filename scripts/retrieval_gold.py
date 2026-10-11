"""
Gold paper targets for the 21 new questions in benchmark v2 (ADR-019).

For each question, title substrings (case-insensitive) of corpus papers that a
domain reader would accept as an answering source. They were written from the
corpus titles and the question wording, by the author of the system, so they
are a judgement, not ground truth: a question may have other valid sources
that are not listed, which makes the hit rate a slightly pessimistic measure.

The original 7 questions (v1) are deliberately not included; their answering
papers were never identified.
"""

GOLD_TARGETS: dict[str, list[str]] = {
    "Why are high-entropy oxides studied for solid oxide fuel cells?": [
        "high-entropy oxides in solid oxide fuel cell",
        "High-entropy perovskites",
    ],
    "How does yttrium doping affect oxygen conductivity in barium-based perovskite cathodes?": [
        "Effects of Yttrium Doping on Oxygen Conductivity",
    ],
    "What are the advantages of double perovskite cathodes for intermediate-temperature SOFCs?": [
        "Sr2CoNbO6",
        "LnBaCo2O6",
    ],
    "How is yttria-stabilized zirconia electrolyte fabricated as a thin dense layer?": [
        "Fabrication of condense and thin YSZ",
        "Three-dimensional printed Yttria-stabilized Zirconia",
    ],
    "What happens at the interface between gadolinia-doped ceria and yttria-stabilized zirconia?": [
        "Interdiffusion between gadolinia doped ceria",
    ],
    "What makes a cobalt-free cathode attractive for proton-conducting fuel cells?": [
        "cobalt-free cathode for proton-conducting",
    ],
    "How does anode microstructure change over time in a solid oxide fuel cell?": [
        "Microstructure Evolution of Solid Oxide Fuel Cell Anodes",
        "Tracking solid oxide cell electrode microstructural evolution",
    ],
    "How can 3D imaging be used to characterise SOFC electrode microstructure?": [
        "Comparing the 3D morphology",
        "Tracking solid oxide cell electrode microstructural evolution",
        "Analyses of microstructural and elastic properties of porous SOFC cathodes",
    ],
    "How does ceramic film cracking occur in solid oxide fuel cells?": [
        "Crack Formation in Ceramic Films",
    ],
    "How does sulfur poisoning affect a PEM fuel cell electrode?": [
        "Analysis of Sulfur Poisoning",
    ],
    "What is the effect of ammonia on proton exchange membrane fuel cell performance?": [
        "Poisoning effect of ammonia",
        "Poisoning mechanism of ammonia",
    ],
    "How is electrochemical impedance spectroscopy used to analyse fuel cell electrodes?": [
        "Evaluating uncertainties in electrochemical impedance spectra",
        "Impedance of a hydrogen--fed SOFC anode",
        "A kernel for PEM fuel cell distribution of relaxation times",
    ],
    "What transport model is used for hydrogen and water in porous SOFC anodes?": [
        "Dusty-gas model conservation law",
        "Impedance of a hydrogen--fed SOFC anode",
    ],
    "How does machine learning predict SOFC electrode polarization resistance?": [
        "Topology-Informed Machine Learning",
    ],
    "What does the distribution of relaxation times reveal in PEM fuel cell impedance?": [
        "A kernel for PEM fuel cell distribution of relaxation times",
    ],
    "What limits the durability of solid oxide electrolysis cells under dynamic load cycling?": [
        "Durability of solid oxide electrolysis stack",
    ],
    "How is oxygen partial pressure distributed across a two-layer electrolyte in an SOEC?": [
        "The oxygen partial pressure in solid oxide electrolysis cells",
    ],
    "How do grain boundaries influence conductivity in solid oxide electrolytes?": [
        "Grain Boundary Space Charge Engineering",
        "Inversion of oxygen potential transitions at grain boundaries",
    ],
    "How does water management affect PEM fuel cell performance?": [
        "Localized Water Management",
        "A model for water transport in the membrane",
        "Influence of wettability on liquid water transport",
    ],
    "How does the gas diffusion layer thickness influence PEM fuel cell performance?": [
        "Effects of gas diffusion layer thickness",
    ],
    "What causes carbon corrosion in a polymer electrolyte fuel cell cathode?": [
        "A Carbon Corrosion Model",
    ],
}
