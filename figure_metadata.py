"""Curated metadata to support manuscript-facing Figures 7–9.

The pipeline can run without these mappings; when a gene is not present here, scripts
fall back to the raw gene identifier. This file mainly improves label readability and
provides default biological annotation for the current manuscript panel.
"""

from __future__ import annotations

CURATED_GENE_METADATA = {
    "ENSG00000175785": {
        "display_label": "PRIMA1",
        "gene_symbol": "PRIMA1",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Neural-islet axis",
        "pathway_context": "acetylcholinesterase anchoring / neural-islet axis",
        "t2d_context": "neural signaling",
    },
    "PRIMA1": {
        "display_label": "PRIMA1",
        "gene_symbol": "PRIMA1",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Neural-islet axis",
        "pathway_context": "acetylcholinesterase anchoring / neural-islet axis",
        "t2d_context": "neural signaling",
    },
    "ENSG00000050165": {
        "display_label": "DKK3",
        "gene_symbol": "DKK3",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "WNT inhibition",
        "pathway_context": "WNT pathway inhibition / dedifferentiation",
        "t2d_context": "beta-cell dedifferentiation",
    },
    "DKK3": {
        "display_label": "DKK3",
        "gene_symbol": "DKK3",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "WNT inhibition",
        "pathway_context": "WNT pathway inhibition / dedifferentiation",
        "t2d_context": "beta-cell dedifferentiation",
    },
    "ENSG00000081181": {
        "display_label": "ARG2",
        "gene_symbol": "ARG2",
        "biotype": "protein_coding",
        "direction": "down",
        "pathway_group": "Arginine metabolism",
        "pathway_context": "arginine-NO metabolism / oxidative stress",
        "t2d_context": "mitochondrial stress",
    },
    "ARG2": {
        "display_label": "ARG2",
        "gene_symbol": "ARG2",
        "biotype": "protein_coding",
        "direction": "down",
        "pathway_group": "Arginine metabolism",
        "pathway_context": "arginine-NO metabolism / oxidative stress",
        "t2d_context": "mitochondrial stress",
    },
    "ENSG00000151834": {
        "display_label": "GABRA2",
        "gene_symbol": "GABRA2",
        "biotype": "protein_coding",
        "direction": "down",
        "pathway_group": "GABA signaling",
        "pathway_context": "GABA signaling / beta-cell identity",
        "t2d_context": "beta-cell identity",
    },
    "GABRA2": {
        "display_label": "GABRA2",
        "gene_symbol": "GABRA2",
        "biotype": "protein_coding",
        "direction": "down",
        "pathway_group": "GABA signaling",
        "pathway_context": "GABA signaling / beta-cell identity",
        "t2d_context": "beta-cell identity",
    },
    "ENSG00000163581": {
        "display_label": "SLC2A2",
        "gene_symbol": "SLC2A2",
        "biotype": "protein_coding",
        "direction": "down",
        "pathway_group": "Glucose transport",
        "pathway_context": "glucose transport / beta-cell identity",
        "t2d_context": "beta-cell identity",
    },
    "SLC2A2": {
        "display_label": "SLC2A2",
        "gene_symbol": "SLC2A2",
        "biotype": "protein_coding",
        "direction": "down",
        "pathway_group": "Glucose transport",
        "pathway_context": "glucose transport / beta-cell identity",
        "t2d_context": "beta-cell identity",
    },
    "ENSG00000163377": {
        "display_label": "TAFA4",
        "gene_symbol": "TAFA4",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Neuroimmune signaling",
        "pathway_context": "neuroimmune signaling / inflammation",
        "t2d_context": "inflammation",
    },
    "TAFA4": {
        "display_label": "TAFA4",
        "gene_symbol": "TAFA4",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Neuroimmune signaling",
        "pathway_context": "neuroimmune signaling / inflammation",
        "t2d_context": "inflammation",
    },
    "ENSG00000010282": {
        "display_label": "HHATL",
        "gene_symbol": "HHATL",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Hedgehog signaling",
        "pathway_context": "hedgehog / developmental regulation",
        "t2d_context": "developmental signaling",
    },
    "HHATL": {
        "display_label": "HHATL",
        "gene_symbol": "HHATL",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Hedgehog signaling",
        "pathway_context": "hedgehog / developmental regulation",
        "t2d_context": "developmental signaling",
    },
    "ENSG00000138964": {
        "display_label": "PARVG",
        "gene_symbol": "PARVG",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Integrin/cytoskeleton",
        "pathway_context": "integrin signaling / cytoskeletal remodeling",
        "t2d_context": "cell remodeling",
    },
    "PARVG": {
        "display_label": "PARVG",
        "gene_symbol": "PARVG",
        "biotype": "protein_coding",
        "direction": "up",
        "pathway_group": "Integrin/cytoskeleton",
        "pathway_context": "integrin signaling / cytoskeletal remodeling",
        "t2d_context": "cell remodeling",
    },
    "ENSG00000199488": {
        "display_label": "RNU1-70P",
        "gene_symbol": "RNU1-70P",
        "biotype": "pseudogene",
        "direction": "down",
        "pathway_group": "RNA processing",
        "pathway_context": "RNA processing / splicing candidate",
        "t2d_context": "RNA processing",
    },
    "RNU1-70P": {
        "display_label": "RNU1-70P",
        "gene_symbol": "RNU1-70P",
        "biotype": "pseudogene",
        "direction": "down",
        "pathway_group": "RNA processing",
        "pathway_context": "RNA processing / splicing candidate",
        "t2d_context": "RNA processing",
    },
    "ENSG00000284653": {
        "display_label": "ENSG00000284653",
        "gene_symbol": "ENSG00000284653",
        "biotype": "lncRNA",
        "direction": "up",
        "pathway_group": "Non-coding regulation",
        "pathway_context": "novel long non-coding RNA",
        "t2d_context": "non-coding regulation",
    },
}

IMMUNESTRESS_GENES = [
    "MICB", "HLA-DRA", "HLA-DPA1", "IL1R2", "IL1RL1", "IDO1", "SERPING1",
    "FPR3", "LTB4R", "GBP2", "TNFRSF10A", "CFH", "ADORA3", "APOL1",
]

BETACELLIDENTITYSECRETION_GENES = [
    "RASGRP1", "PPP1R1A", "ENTPD3", "ADCYAP1", "FFAR4", "TMED6", "PLCXD3",
    "PDE8B", "CASR", "PFKFB2", "ACLY", "TGFBR3", "ASB9", "PPM1K",
]

DEFAULT_EDGE_THRESHOLD = 0.35


def lookup_gene_metadata(gene_id: str) -> dict:
    meta = CURATED_GENE_METADATA.get(gene_id, {}).copy()
    if not meta:
        meta = {
            "display_label": gene_id,
            "gene_symbol": gene_id,
            "biotype": "unknown",
            "direction": "unknown",
            "pathway_group": "Unannotated",
            "pathway_context": "manual annotation needed",
            "t2d_context": "manual annotation needed",
        }
    return meta
