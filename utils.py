"""
utils.py — Funções auxiliares para o app de rastreamento de parto prematuro
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd


def format_probability(p: float) -> str:
    """Formata probabilidade como percentual com 1 casa decimal."""
    return f"{p * 100:.1f}%"


def logistic(x: float) -> float:
    """Função logística (sigmoide)."""
    return 1.0 / (1.0 + np.exp(-x))


def build_forest_plot(or_dict: dict, title: str = "Forest Plot — Odds Ratios (IC 95%)") -> go.Figure:
    """
    Gera um forest plot interativo a partir de um dicionário de Odds Ratios.

    or_dict: {nome_variável: {"or": float, "ci_lo": float, "ci_hi": float, "p": float}}
    """
    names = list(or_dict.keys())
    ors   = [or_dict[n]["or"]    for n in names]
    lo    = [or_dict[n]["ci_lo"] for n in names]
    hi    = [or_dict[n]["ci_hi"] for n in names]
    pvals = [or_dict[n]["p"]     for n in names]

    colors = ["#e53935" if o > 1 else "#43a047" for o in ors]

    fig = go.Figure()

    # Linha de referência OR=1
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", line_width=1.5,
                  annotation_text="OR = 1", annotation_position="top right",
                  annotation_font_size=10)

    for i, (name, or_val, lo_val, hi_val, pv, col) in enumerate(
            zip(names, ors, lo, hi, pvals, colors)):

        # Linha de IC
        fig.add_trace(go.Scatter(
            x=[lo_val, hi_val], y=[name, name],
            mode="lines",
            line=dict(color="#1a237e", width=2),
            showlegend=False,
            hoverinfo="skip"
        ))
        # Ponto central (OR)
        sig_marker = "star" if pv < 0.05 else "square"
        fig.add_trace(go.Scatter(
            x=[or_val], y=[name],
            mode="markers",
            marker=dict(size=12, color=col, symbol=sig_marker,
                        line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"OR = {or_val:.3f}<br>"
                f"IC 95%: {lo_val:.3f} – {hi_val:.3f}<br>"
                f"p-valor = {pv:.3f}<br>"
                f"{'★ Significativo (p<0,05)' if pv < 0.05 else 'Não significativo'}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Odds Ratio (escala log)",
        xaxis_type="log",
        xaxis=dict(gridcolor="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
        height=max(300, 50 * len(names) + 80),
        margin=dict(l=10, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        font=dict(family="Arial", size=11),
    )
    return fig


def build_roc_comparison() -> go.Figure:
    """
    Gera gráfico comparativo das curvas ROC aproximadas com base nos AUCs publicados.
    Pontos operacionais reais (a 10% FPR) são plotados sobre as curvas.
    """
    models = [
        {"name": "CL < 25 mm (AUC = 0,318)", "auc": 0.318,
         "point_fpr": 0.082, "point_tpr": 0.333, "color": "#43a047"},
        {"name": "LR — Andrade Júnior (AUC = 0,749)", "auc": 0.749,
         "point_fpr": 0.078, "point_tpr": 0.389, "color": "#e53935"},
        {"name": "XGBoost (AUC = 0,788)*", "auc": 0.788,
         "point_fpr": 0.074, "point_tpr": 0.444, "color": "#1565c0"},
        {"name": "SBELM Neural Network (AUC = 0,808)*", "auc": 0.808,
         "point_fpr": 0.072, "point_tpr": 0.472, "color": "#7b1fa2"},
    ]

    fig = go.Figure()

    # Linha de acaso
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="lightgray", dash="dash", width=1.5),
        name="Acaso (AUC = 0,500)",
        hoverinfo="skip"
    ))

    x_base = np.linspace(0, 1, 200)

    for m in models:
        # Aproximação paramétrica da curva ROC dado o AUC
        auc = m["auc"]
        if auc > 0.5:
            # parâmetro de discriminação (dist. binormal)
            d = np.sqrt(2) * np.abs(np.log(auc / (1 - auc)))
        else:
            d = 0.0

        from scipy.special import ndtri, ndtr  # type: ignore
        try:
            y_roc = ndtr(ndtri(x_base) + d)
            y_roc = np.clip(y_roc, 0, 1)
        except Exception:
            y_roc = x_base ** (1 / max(auc, 0.01))

        fig.add_trace(go.Scatter(
            x=x_base, y=y_roc,
            mode="lines",
            name=m["name"],
            line=dict(color=m["color"], width=2.5),
        ))

        # Ponto operacional a 10% FPR (valores reais da Tabela 5)
        fig.add_trace(go.Scatter(
            x=[m["point_fpr"]], y=[m["point_tpr"]],
            mode="markers",
            marker=dict(size=12, color=m["color"], symbol="star",
                        line=dict(color="white", width=1.5)),
            name=f"★ Ponto 10% FPR — {m['name'].split('(')[0].strip()}",
            showlegend=False,
            hovertemplate=(
                f"<b>{m['name'].split('(')[0].strip()}</b><br>"
                f"FPR = {m['point_fpr']:.1%}<br>"
                f"Sensibilidade = {m['point_tpr']:.1%}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text="Curvas ROC — Comparação dos Modelos"
                 "<br><sup>★ = ponto operacional com FPR fixo em 10% (valores exatos da Tabela 5)</sup>",
            font=dict(size=13)
        ),
        xaxis_title="1 − Especificidade (Taxa de Falso Positivo)",
        yaxis_title="Sensibilidade (Taxa de Verdadeiro Positivo)",
        xaxis=dict(range=[0, 1], gridcolor="#e0e0e0"),
        yaxis=dict(range=[0, 1], gridcolor="#e0e0e0"),
        height=480,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.85)", font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        font=dict(family="Arial"),
    )
    return fig


def performance_table() -> pd.DataFrame:
    """Retorna DataFrame com métricas de performance dos modelos (Tabela 5 do artigo)."""
    return pd.DataFrame({
        "Modelo":        ["CL < 25 mm", "LR (Andrade Jr.)", "XGBoost *", "SBELM *"],
        "AUC":           [0.318,        0.749,               0.788,        0.808],
        "Sensib. (%)":   [33.3,         38.9,                44.4,         47.2],
        "Espec. (%)":    [91.8,         92.2,                92.6,         92.8],
        "Acurácia (%)":  [87.8,         88.5,                89.3,         89.7],
        "VPP (%)":       [23.1,         26.9,                30.8,         32.7],
        "VPN (%)":       [94.9,         95.3,                95.8,         96.0],
        "p-valor":       ["—",          "< 0,0001",          "< 0,0001",   "< 0,0001"],
    })
