"""
models.py — Modelos preditivos para rastreamento de parto prematuro espontâneo < 35 semanas
Banco de dados: UNIFESP — Setor de Predição de Parto Prematuro (2010–2018), n=524

Referências:
[1] Andrade Júnior VL, França MS, Hatanaka AR et al.
    J Matern Fetal Neonatal Med. 2023;36(2):2241100.
    DOI: 10.1080/14767058.2023.2241100
[2] França MS, Andrade Júnior VL, Hatanaka AR, Santos R et al.
    Variáveis selecionadas por Regressão Logística — Banco de Dados UNIFESP.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 1: Regressão Logística — Andrade Júnior et al. (2023) — Tabela 2 (PDF)
# AUC = 0.749 | N = 524 gestações únicas | 18–24 semanas
# ─────────────────────────────────────────────────────────────────────────────

COEFF_ANDRADE = {
    "constant":       -3.116,  # Intercepto (β₀)
    "funneling":      +1.787,  # Presença de funil cervical     OR=5.971  p=0.026
    "index_le_0200":  +0.427,  # Índice CL/Ângulo ≤ 0.200       OR=1.533  p=0.470
    "no_antibiotic":  +0.530,  # Sem antibiótico na gestação     OR=1.698  p=0.297
    "prev_ptb_37":    +1.579,  # PPT anterior < 37 semanas       OR=4.851  p=0.004
    "one_curettage":  -1.245,  # Uma curetagem prévia            OR=0.288  p=0.254
    "weight_le_58":   +0.699,  # Peso ≤ 58 kg                    OR=2.012  p=0.178
    "no_smoking":     -0.949,  # Sem tabagismo                   OR=0.387  p=0.148
    "cl_lt_30_9":     +0.622,  # CL < 30.9 mm                    OR=1.863  p=0.266
}

OR_ANDRADE = {
    "Funil cervical (Funneling)":       {"or": 5.971, "ci_lo": 1.234, "ci_hi": 28.891, "p": 0.026},
    "Índice CL/Ângulo ≤ 0,200":         {"or": 1.533, "ci_lo": 0.481, "ci_hi": 4.887,  "p": 0.470},
    "Sem antibiótico na gestação":      {"or": 1.698, "ci_lo": 0.628, "ci_hi": 4.594,  "p": 0.297},
    "PPT anterior < 37 semanas":        {"or": 4.851, "ci_lo": 1.661, "ci_hi": 14.172, "p": 0.004},
    "Uma curetagem prévia":             {"or": 0.288, "ci_lo": 0.034, "ci_hi": 2.447,  "p": 0.254},
    "Peso ≤ 58 kg":                     {"or": 2.012, "ci_lo": 0.727, "ci_hi": 5.568,  "p": 0.178},
    "Sem tabagismo":                    {"or": 0.387, "ci_lo": 0.107, "ci_hi": 1.399,  "p": 0.148},
    "CL < 30,9 mm":                     {"or": 1.863, "ci_lo": 0.622, "ci_hi": 5.580,  "p": 0.266},
}

# Performance metrics — Tabela 5 do artigo (FPR fixo = 10%)
PERF_ANDRADE = {
    "auc": 0.749, "sensitivity": 38.9, "specificity": 92.2,
    "accuracy": 88.5, "ppv": 26.9, "npv": 95.3, "fpr_fixed": 10
}


def predict_andrade_lr(funneling: int, index_le_0200: int, no_antibiotic: int,
                       prev_ptb_37: int, one_curettage: int, weight_le_58: int,
                       no_smoking: int, cl_lt_30_9: int) -> tuple:
    """
    Calcula probabilidade de sPTB < 35 semanas usando o modelo LR de Andrade Júnior et al.

    Todos os inputs são binários (0 = não, 1 = sim).
    Returns: (probability: float, logit: float)
    """
    c = COEFF_ANDRADE
    logit = (c["constant"]
             + c["funneling"]     * funneling
             + c["index_le_0200"] * index_le_0200
             + c["no_antibiotic"] * no_antibiotic
             + c["prev_ptb_37"]   * prev_ptb_37
             + c["one_curettage"] * one_curettage
             + c["weight_le_58"]  * weight_le_58
             + c["no_smoking"]    * no_smoking
             + c["cl_lt_30_9"]    * cl_lt_30_9)
    probability = 1.0 / (1.0 + np.exp(-logit))
    return probability, logit


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 2: Regressão Logística — França MS, Andrade Júnior VL, Hatanaka AR et al.
# Banco de Dados UNIFESP — inclui gestações únicas e múltiplas
# ─────────────────────────────────────────────────────────────────────────────

COEFF_FRANCA = {
    "constant":         +0.724,  # Intercepto (β₀)
    "white":            +0.929,  # Raça branca          OR=2.532  p=0.019
    "no_curettage":     -2.182,  # Ausência curetagem   OR=0.113  p=0.000
    "prev_ptb_37":      +1.294,  # PPT anterior <37s    OR=3.647  p=0.001
    "singleton":        -2.005,  # Gestação única       OR=0.135  p=0.000
    "ga_lt_19w":        +1.216,  # IG < 19 sem ao US    OR=3.373  p=0.008
    "cl_straight_5_14": +1.404,  # CL reto 5,2-14,7mm  OR=4.072  p=0.006
    "cl_curve_gt_21":   -1.533,  # CL curvo > 21mm      OR=0.216  p=0.000
}

OR_FRANCA = {
    "Raça branca":                   {"or": 2.532, "ci_lo": 1.164, "ci_hi": 5.508,  "p": 0.019},
    "Ausência de curetagem prévia":  {"or": 0.113, "ci_lo": 0.049, "ci_hi": 0.258,  "p": 0.000},
    "PPT anterior < 37 semanas":     {"or": 3.647, "ci_lo": 1.650, "ci_hi": 8.058,  "p": 0.001},
    "Gestação única (singleton)":    {"or": 0.135, "ci_lo": 0.052, "ci_hi": 0.349,  "p": 0.000},
    "IG < 19 semanas ao US":         {"or": 3.373, "ci_lo": 1.379, "ci_hi": 8.248,  "p": 0.008},
    "CL reto entre 5,2–14,7 mm":    {"or": 4.072, "ci_lo": 1.506, "ci_hi": 11.006, "p": 0.006},
    "CL curvo > 21 mm":              {"or": 0.216, "ci_lo": 0.094, "ci_hi": 0.498,  "p": 0.000},
}


def predict_franca_lr(white: int, no_curettage: int, prev_ptb_37: int,
                      singleton: int, ga_lt_19w: int,
                      cl_straight_5_14: int, cl_curve_gt_21: int) -> tuple:
    """
    Calcula probabilidade de PPT usando o modelo LR de França MS et al. (UNIFESP).

    Todos os inputs são binários (0 = não, 1 = sim).
    Returns: (probability: float, logit: float)
    """
    c = COEFF_FRANCA
    logit = (c["constant"]
             + c["white"]            * white
             + c["no_curettage"]     * no_curettage
             + c["prev_ptb_37"]      * prev_ptb_37
             + c["singleton"]        * singleton
             + c["ga_lt_19w"]        * ga_lt_19w
             + c["cl_straight_5_14"] * cl_straight_5_14
             + c["cl_curve_gt_21"]   * cl_curve_gt_21)
    probability = 1.0 / (1.0 + np.exp(-logit))
    return probability, logit


# ─────────────────────────────────────────────────────────────────────────────
# RASTREAMENTO PADRÃO: CL < 25 mm (gold standard para comparação)
# AUC = 0.318 | Sens = 33.3% | Espec = 91.8% @ 10% FPR
# ─────────────────────────────────────────────────────────────────────────────

PERF_CL25 = {
    "auc": 0.318, "sensitivity": 33.3, "specificity": 91.8,
    "accuracy": 87.8, "ppv": 23.1, "npv": 94.9, "fpr_fixed": 10
}


def screen_cl_25mm(cervical_length_mm: float) -> bool:
    """Rastreamento clássico: positivo se CL < 25 mm."""
    return cervical_length_mm < 25.0


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADE: categorização de risco
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_category(probability: float) -> tuple:
    """
    Categoriza o risco com base na probabilidade calculada.
    Prevalência na coorte: 6.9% (36/524).

    Returns: (category: str, icon: str, color: str)
    """
    if probability < 0.05:
        return "BAIXO", "🟢", "#28a745"
    elif probability < 0.10:
        return "LIMÍTROFE", "🟡", "#ffc107"
    elif probability < 0.20:
        return "ALTO", "🟠", "#fd7e14"
    else:
        return "MUITO ALTO", "🔴", "#dc3545"
