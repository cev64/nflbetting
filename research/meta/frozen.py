"""Designs frozen after the DEV search (2010-2017). Best config per design type by dev log-loss.
PRODUCTION choices (made before the holdout was run) are marked."""
SU_FROZEN = [
    ("market", {"hl": 4}),
    ("logit", {"alpha": 0.03}),                                   # flat per-model stack
    ("logit", {"alpha": 0.01, "by": "family"}),                   # flat family stack
    ("logit", {"alpha": 0.03, "by": "family", "ctx": 1}),         # contextual (interactions) family stack
    ("logit", {"alpha": 0.03, "ctx": 1}),                         # contextual per-model stack
    ("lgbm", {"n": 100, "hl": 4}),                                # LightGBM on market offset
    ("bma", {"tau": 1.0, "hl": 4}),                               # time-varying BMA
    ("bma", {"tau": 0.02, "hl": 4, "cells": ["pickem", "early", "playoff"]}),  # contextual BMA
    ("router", {"k": 3, "conf": 0.5}),                            # disagreement router
    ("guard", {"base": "bma", "base_cfg": {"tau": 1.0, "hl": 4}, "delta": 0.02}),
    ("guard", {"base": "logit", "base_cfg": {"alpha": 0.01, "by": "family"}, "delta": 1.0}),  # PRODUCTION
]
ATS_FROZEN = [
    ("logit", {"signals": None, "alpha": 0.001}),                 # structure + injury gap only
    ("logit", {"signals": "model", "alpha": 0.03}),
    ("logit", {"signals": "family", "alpha": 0.01}),              # PRODUCTION
    ("logit", {"signals": "family", "alpha": 0.01, "ctx": 1}),
    ("logit", {"signals": "mean", "alpha": 0.001}),
    ("lgbm", {"signals": "family", "n": 100}),
    ("vote", {}),
]
SU_PRODUCTION = SU_FROZEN[-1]
ATS_PRODUCTION = ATS_FROZEN[2]
