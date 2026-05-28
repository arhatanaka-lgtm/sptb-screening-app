"""
app.py — Rastreamento de Parto Prematuro Espontâneo < 35 semanas
Modelos de Inteligência Artificial — UNIFESP (EPM)

Referência principal:
Andrade Júnior VL, França MS, Hatanaka AR et al.
J Matern Fetal Neonatal Med. 2023;36(2):2241100.
DOI: 10.1080/14767058.2023.2241100
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from models import (
    predict_andrade_lr, OR_ANDRADE, PERF_ANDRADE,
    predict_franca_lr,  OR_FRANCA,
    screen_cl_25mm, PERF_CL25,
    get_risk_category,
)
from utils import (
    format_probability, build_forest_plot,
    build_roc_comparison, performance_table,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rastreamento PPT < 35s — UNIFESP AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .header-banner {
    background: linear-gradient(135deg, #0d1b5e 0%, #1a237e 55%, #283593 100%);
    color: white; padding: 1.8rem 2rem 1.4rem; border-radius: 14px;
    margin-bottom: 1.2rem; text-align: center;
  }
  .header-banner h1 { font-size: 1.75rem; margin: 0 0 0.3rem; }
  .header-banner .subtitle { font-size: 0.9rem; opacity: 0.88; margin: 0.2rem 0; }
  .header-banner .warning { font-size: 0.75rem; opacity: 0.7; margin-top: 0.5rem; }

  .risk-card {
    border-radius: 14px; border: 2.5px solid; padding: 1.6rem;
    text-align: center; margin: 0.8rem 0;
  }
  .risk-prob { font-size: 3rem; font-weight: 800; line-height: 1; }
  .risk-label { font-size: 1.15rem; font-weight: 700; margin-top: 0.4rem; }
  .risk-sub { font-size: 0.82rem; color: #555; margin-top: 0.3rem; }

  .info-card {
    background: #f0f2f6; border-left: 4px solid #1a237e;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    font-size: 0.88rem;
  }
  .eq-box {
    background: #e8eaf6; border-radius: 8px; padding: 1rem;
    font-family: "Courier New", monospace; font-size: 0.83rem; line-height: 1.7;
  }
  .warn-box {
    background: #fff8e1; border: 1px solid #ffb300; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.83rem; margin: 0.6rem 0;
  }
  .separator { border-top: 1px solid #e0e0e0; margin: 1.2rem 0; }
  div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🔬 Rastreamento de Parto Prematuro Espontâneo &lt; 35 semanas</h1>
  <div class="subtitle">
    Modelos de Inteligência Artificial — UNIFESP · Setor de Predição de Parto Prematuro
  </div>
  <div class="subtitle">
    <strong>Andrade Júnior VL, França MS, Hatanaka AR et al.</strong>
    · J Matern Fetal Neonatal Med. 2023;36(2):2241100
    · DOI: 10.1080/14767058.2023.2241100
  </div>
  <div class="warning">
    ⚠️ Ferramenta educacional e de pesquisa · Não substitui o julgamento clínico individualizado
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Abas principais
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧮  Modelo LR — Andrade Júnior et al. (2023)",
    "🧮  Modelo LR — França MS et al. (UNIFESP)",
    "📈  Comparação dos Modelos",
    "ℹ️  Metodologia & Sobre",
])


# ═════════════════════════════════════════════════════════════════════════════
# ABA 1 — MODELO ANDRADE JÚNIOR LR
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Modelo de Regressão Logística — Andrade Júnior et al. (2023)")
    st.markdown(
        "**8 variáveis clínicas e ultrassonográficas** selecionadas por stepwise forward LR "
        "a partir de 33 candidatas em 524 gestações únicas (18–24 semanas).  **AUC = 0,749** · "
        "Sensibilidade 38,9% · Especificidade 92,2% @ FPR 10%."
    )

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    col_in, col_out = st.columns([1, 1], gap="large")

    # ── Entradas ──────────────────────────────────────────────────────────────
    with col_in:
        st.markdown("#### 📋 Dados da Paciente")

        st.markdown("**🔬 Ultrassonografia Transvaginal do Colo (18–24 semanas)**")

        funneling_sel = st.radio(
            "Presença de funil cervical (funneling)?",
            ["Não", "Sim"], horizontal=True, key="a1_fun"
        )
        funneling = 1 if funneling_sel == "Sim" else 0

        index_val = st.number_input(
            "Índice A-B / A-Y-B (CL reto ÷ ângulo interno do colo)",
            min_value=0.000, max_value=2.000, value=0.300,
            step=0.001, format="%.3f", key="a1_idx",
            help=(
                "Medida original do grupo UNIFESP. "
                "A-B = comprimento cervical reto (mm). "
                "A-Y-B = ângulo interno do colo (graus). "
                "Índice = A-B ÷ A-Y-B. Risco se ≤ 0,200."
            )
        )
        index_le_0200 = 1 if index_val <= 0.200 else 0
        st.caption(
            f"→ Índice **{'≤ 0,200 — fator de risco ✓' if index_le_0200 else '> 0,200 — fora da faixa de risco'}**"
        )

        cl_mm = st.number_input(
            "Comprimento cervical — medida reta (mm)",
            min_value=0.0, max_value=65.0, value=34.0,
            step=0.1, format="%.1f", key="a1_cl",
        )
        cl_lt_30_9 = 1 if cl_mm < 30.9 else 0
        cl_lt_25   = cl_mm < 25.0
        st.caption(f"→ CL {'< 30,9 mm — fator de risco no modelo LR ✓' if cl_lt_30_9 else '≥ 30,9 mm'}")
        st.caption(f"→ CL {'< 25 mm — rastreamento clássico **POSITIVO** ⚠️' if cl_lt_25 else '≥ 25 mm — rastreamento clássico negativo'}")

        st.markdown("**🏥 Antecedentes Obstétricos**")

        ptb_sel = st.radio(
            "Parto prematuro anterior < 37 semanas?",
            ["Não", "Sim"], horizontal=True, key="a1_ptb"
        )
        prev_ptb_37 = 1 if ptb_sel == "Sim" else 0

        cur_sel = st.selectbox(
            "Curetagens uterinas prévias",
            ["Nenhuma", "1 curetagem (fator protetor)", "2 ou mais"], key="a1_cur"
        )
        one_curettage = 1 if "1 curetagem" in cur_sel else 0

        st.markdown("**👤 Dados Clínicos e Hábitos**")

        weight = st.number_input(
            "Peso materno (kg)", min_value=30.0, max_value=160.0,
            value=65.0, step=0.5, format="%.1f", key="a1_wt"
        )
        weight_le_58 = 1 if weight <= 58.0 else 0
        st.caption(f"→ {'Peso ≤ 58 kg — fator de risco ✓' if weight_le_58 else 'Peso > 58 kg'}")

        smoke_sel = st.radio(
            "Tabagismo durante a gestação?",
            ["Não", "Sim"], horizontal=True, key="a1_smoke"
        )
        no_smoking = 1 if smoke_sel == "Não" else 0

        ab_sel = st.radio(
            "Uso de antibiótico durante a gestação atual?",
            ["Não", "Sim"], horizontal=True, key="a1_ab",
            help="Antibiótico prescrito antes da inclusão/coleta de dados."
        )
        no_antibiotic = 1 if ab_sel == "Não" else 0

    # ── Resultados ─────────────────────────────────────────────────────────
    with col_out:
        st.markdown("#### 📊 Resultado do Rastreamento")

        prob, logit = predict_andrade_lr(
            funneling, index_le_0200, no_antibiotic, prev_ptb_37,
            one_curettage, weight_le_58, no_smoking, cl_lt_30_9
        )
        cat, icon, color = get_risk_category(prob)

        st.markdown(f"""
        <div class="risk-card" style="border-color:{color}; background:{color}18;">
          <div class="risk-prob" style="color:{color};">{prob*100:.1f}%</div>
          <div class="risk-label" style="color:{color};">{icon} RISCO {cat}</div>
          <div class="risk-sub">Probabilidade estimada de sPTB &lt; 35 semanas</div>
        </div>
        """, unsafe_allow_html=True)

        # Rastreamento clássico
        cl_status = "🔴 POSITIVO" if cl_lt_25 else "🟢 NEGATIVO"
        st.markdown(f"""
        <div class="info-card">
          <strong>Rastreamento clássico (CL &lt; 25 mm):</strong>
          &nbsp; <b>{cl_status}</b>
          &emsp;|&emsp; CL informado: {cl_mm:.1f} mm
        </div>
        """, unsafe_allow_html=True)

        # Tabela de performance
        st.markdown("**Desempenho dos modelos @ FPR 10% (Tabela 5 do artigo):**")
        df_perf = performance_table()
        st.dataframe(df_perf, hide_index=True, use_container_width=True)
        st.caption("\\* XGBoost e SBELM não reproduzíveis (pesos da rede neural não publicados).")

        # Equação expandível
        with st.expander("🔢 Mostrar equação completa do modelo"):
            contrib = {
                "Funil":         funneling     * 1.787,
                "Índice≤0.200":  index_le_0200 * 0.427,
                "S/antibiótico": no_antibiotic * 0.530,
                "PPT<37s":       prev_ptb_37   * 1.579,
                "1 Curetagem":   one_curettage * (-1.245),
                "Peso≤58kg":     weight_le_58  * 0.699,
                "S/tabagismo":   no_smoking    * (-0.949),
                "CL<30.9mm":     cl_lt_30_9    * 0.622,
            }
            st.markdown(f"""
            <div class="eq-box">
            logit(p) = -3,116 (intercepto)<br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 1,787 × Funil ({funneling})
              = <b>{contrib['Funil']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 0,427 × Índice≤0,200 ({index_le_0200})
              = <b>{contrib['Índice≤0.200']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 0,530 × Sem antibiótico ({no_antibiotic})
              = <b>{contrib['S/antibiótico']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 1,579 × PPT&lt;37sem ({prev_ptb_37})
              = <b>{contrib['PPT<37s']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + (−1,245) × 1 Curetagem ({one_curettage})
              = <b>{contrib['1 Curetagem']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 0,699 × Peso≤58kg ({weight_le_58})
              = <b>{contrib['Peso≤58kg']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + (−0,949) × Sem tabagismo ({no_smoking})
              = <b>{contrib['S/tabagismo']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 0,622 × CL&lt;30,9mm ({cl_lt_30_9})
              = <b>{contrib['CL<30.9mm']:+.4f}</b><br><br>
            <b>logit = {logit:+.4f}</b><br>
            <b>p = 1 / (1 + e<sup>−{logit:.4f}</sup>) = {prob:.4f} → {prob*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

        # Forest plot
        with st.expander("📊 Forest Plot — Odds Ratios"):
            fig_fp = build_forest_plot(OR_ANDRADE, "Odds Ratios — Modelo LR Andrade Júnior et al. (2023)")
            st.plotly_chart(fig_fp, use_container_width=True)
            st.caption("★ = variável com p < 0,05. Tabela 2 do artigo. Escala logarítmica.")


# ═════════════════════════════════════════════════════════════════════════════
# ABA 2 — MODELO FRANÇA MS LR
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Modelo de Regressão Logística — França MS, Andrade Jr VL, Hatanaka AR et al.")
    st.markdown(
        "**7 variáveis** selecionadas por regressão logística a partir do banco de dados UNIFESP. "
        "Inclui gestações únicas e múltiplas. Introduz as novas medidas cervicais desenvolvidas pelo grupo "
        "(CL reto e CL curvo como preditores categóricos)."
    )
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    col_in2, col_out2 = st.columns([1, 1], gap="large")

    # ── Entradas ──────────────────────────────────────────────────────────────
    with col_in2:
        st.markdown("#### 📋 Dados da Paciente")

        st.markdown("**👤 Dados Demográficos e Obstétricos**")

        race_sel = st.radio("Raça / Etnia", ["Não branca", "Branca"],
                            horizontal=True, key="f2_race")
        white = 1 if race_sel == "Branca" else 0

        cur_f_sel = st.radio(
            "Curetagem uterina prévia?",
            ["Sim (tem curetagem prévia)", "Não (sem curetagem)"],
            horizontal=True, key="f2_cur"
        )
        no_curettage = 1 if "Não" in cur_f_sel else 0
        st.caption(
            "→ Ausência de curetagem = **fator protetor** (OR = 0,113): "
            "curetagem aumenta risco de PPT por lesão cervical."
            if no_curettage else
            "→ Curetagem prévia presente — fator de risco."
        )

        ptb_f_sel = st.radio(
            "Parto prematuro anterior < 37 semanas?",
            ["Não", "Sim"], horizontal=True, key="f2_ptb"
        )
        prev_ptb_37_f = 1 if ptb_f_sel == "Sim" else 0

        gest_sel = st.radio(
            "Tipo de gestação atual",
            ["Múltipla (gemelar ou mais)", "Única (singleton)"],
            horizontal=True, key="f2_sing"
        )
        singleton = 1 if "Única" in gest_sel else 0
        st.caption(
            "→ Gestação única = fator protetor (OR = 0,135): "
            "múltipla tem risco inerentemente maior de PPT."
        )

        st.markdown("**🔬 Dados do Ultrassom Transvaginal do Colo**")

        ga_us = st.number_input(
            "Idade gestacional ao US cervical (semanas)",
            min_value=14.0, max_value=28.0, value=20.0,
            step=0.5, format="%.1f", key="f2_ga",
            help="IG no momento da realização do US transvaginal para medida do colo."
        )
        ga_lt_19w = 1 if ga_us < 19.0 else 0
        st.caption(
            f"→ IG {'< 19 semanas — fator de risco ✓ (OR = 3,373)' if ga_lt_19w else '≥ 19 semanas'}"
        )

        cl_str = st.number_input(
            "CL reto — straight cervical length (mm)",
            min_value=0.0, max_value=65.0, value=30.0,
            step=0.1, format="%.1f", key="f2_clstr",
            help="Comprimento cervical em linha reta (medida A-B). Fator de risco: entre 5,2 e 14,7 mm."
        )
        cl_straight_5_14 = 1 if 5.2 <= cl_str <= 14.7 else 0
        st.caption(
            f"→ {'5,2–14,7 mm — fator de risco crítico ✓ (OR = 4,072)' if cl_straight_5_14 else 'Fora da faixa de risco crítico (5,2–14,7 mm)'}"
        )

        cl_cur = st.number_input(
            "CL curvo — curve cervical length (mm)",
            min_value=0.0, max_value=80.0, value=35.0,
            step=0.1, format="%.1f", key="f2_clcur",
            help=(
                "Comprimento cervical medido ao longo da curvatura (traçado A-Y-B). "
                "Fator protetor se > 21 mm (OR = 0,216)."
            )
        )
        cl_curve_gt_21 = 1 if cl_cur > 21.0 else 0
        st.caption(
            f"→ {'> 21 mm — fator PROTETOR ✓ (OR = 0,216)' if cl_curve_gt_21 else '≤ 21 mm — sem fator protetor'}"
        )

    # ── Resultados ─────────────────────────────────────────────────────────
    with col_out2:
        st.markdown("#### 📊 Resultado do Rastreamento")

        prob_f, logit_f = predict_franca_lr(
            white, no_curettage, prev_ptb_37_f, singleton,
            ga_lt_19w, cl_straight_5_14, cl_curve_gt_21
        )
        cat_f, icon_f, color_f = get_risk_category(prob_f)

        st.markdown(f"""
        <div class="risk-card" style="border-color:{color_f}; background:{color_f}18;">
          <div class="risk-prob" style="color:{color_f};">{prob_f*100:.1f}%</div>
          <div class="risk-label" style="color:{color_f};">{icon_f} RISCO {cat_f}</div>
          <div class="risk-sub">Probabilidade estimada de Parto Prematuro (Modelo França MS et al.)</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔢 Mostrar equação completa do modelo"):
            contribs_f = {
                "Raça branca":   white           * 0.929,
                "Sem curetagem": no_curettage    * (-2.182),
                "PPT<37s":       prev_ptb_37_f   * 1.294,
                "Singleton":     singleton        * (-2.005),
                "IG<19s":        ga_lt_19w        * 1.216,
                "CLstr5-14":     cl_straight_5_14 * 1.404,
                "CLcur>21":      cl_curve_gt_21   * (-1.533),
            }
            st.markdown(f"""
            <div class="eq-box">
            logit(p) = +0,724 (intercepto)<br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 0,929 × Raça branca ({white})
              = <b>{contribs_f['Raça branca']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + (−2,182) × Sem curetagem ({no_curettage})
              = <b>{contribs_f['Sem curetagem']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 1,294 × PPT&lt;37sem ({prev_ptb_37_f})
              = <b>{contribs_f['PPT<37s']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + (−2,005) × Singleton ({singleton})
              = <b>{contribs_f['Singleton']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 1,216 × IG&lt;19sem ({ga_lt_19w})
              = <b>{contribs_f['IG<19s']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + 1,404 × CL reto 5,2–14,7mm ({cl_straight_5_14})
              = <b>{contribs_f['CLstr5-14']:+.4f}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp; + (−1,533) × CL curvo &gt;21mm ({cl_curve_gt_21})
              = <b>{contribs_f['CLcur>21']:+.4f}</b><br><br>
            <b>logit = {logit_f:+.4f}</b><br>
            <b>p = 1 / (1 + e<sup>−{logit_f:.4f}</sup>) = {prob_f:.4f} → {prob_f*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📊 Forest Plot — Odds Ratios"):
            fig_fp2 = build_forest_plot(OR_FRANCA, "Odds Ratios — Modelo LR França MS et al. (UNIFESP)")
            st.plotly_chart(fig_fp2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ABA 3 — COMPARAÇÃO DOS MODELOS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Comparação de Desempenho dos Modelos")

    col_roc, col_bars = st.columns([1.3, 0.7], gap="large")

    with col_roc:
        st.plotly_chart(build_roc_comparison(), use_container_width=True)
        st.caption(
            "\\* Curvas ROC de XGBoost e SBELM são aproximações com base nos AUCs publicados. "
            "Pontos ★ = valores exatos da Tabela 5 do artigo (FPR fixo em 10%)."
        )

    with col_bars:
        df_perf = performance_table()

        fig_sens = go.Figure(go.Bar(
            x=df_perf["Modelo"], y=df_perf["Sensib. (%)"],
            marker_color=["#43a047", "#e53935", "#1565c0", "#7b1fa2"],
            text=[f"{v}%" for v in df_perf["Sensib. (%)"]],
            textposition="outside",
        ))
        fig_sens.update_layout(
            title="Sensibilidade @ FPR 10%", yaxis=dict(range=[0, 70]),
            height=230, margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,250,1)",
            showlegend=False,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        fig_auc = go.Figure(go.Bar(
            x=df_perf["Modelo"], y=df_perf["AUC"],
            marker_color=["#43a047", "#e53935", "#1565c0", "#7b1fa2"],
            text=df_perf["AUC"].astype(str), textposition="outside",
        ))
        fig_auc.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text="Acaso", annotation_position="right")
        fig_auc.update_layout(
            title="AUC (ROC)", yaxis=dict(range=[0, 1.0]),
            height=230, margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,250,1)",
            showlegend=False,
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # Tabela completa
    st.markdown("#### Tabela 5 — Métricas Completas (artigo original, FPR fixo = 10%)")
    st.dataframe(df_perf, hide_index=True, use_container_width=True)

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # A 20% FPR — SBELM vs CL < 28.5mm
    st.markdown("#### Análise a 20% FPR — SBELM vs CL < 28,5 mm (limiar equivalente)")
    col_20a, col_20b = st.columns(2)
    with col_20a:
        df_20 = pd.DataFrame({
            "Métrica":     ["Sensibilidade", "Especificidade", "Acurácia", "VPP", "VPN"],
            "SBELM (20% FPR)":       ["75,0%", "84,2%", "83,6%", "26,0%", "97,0%"],
            "CL < 28,5mm (20% FPR)": ["41,5%", "82,0%", "78,8%", "16,3%", "94,3%"],
            "Melhora relativa":       ["+80,7%", "+2,7%", "+6,1%", "+59,5%", "+2,9%"],
        })
        st.dataframe(df_20, hide_index=True, use_container_width=True)
    with col_20b:
        st.markdown("""
        <div class="warn-box">
        ℹ️ <b>Perspectiva clínica:</b> Com FPR = 20%, o SBELM alcança sensibilidade de 75%
        — ganho de ~83% vs. o comprimento cervical isolado. Este limiar de 20% poderia ser
        aplicado em uma <b>2ª avaliação entre 26–28 semanas</b> para maximizar a detecção
        em populações de alto risco.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ABA 4 — METODOLOGIA & SOBRE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ Metodologia e Informações sobre o Estudo")

    col_m1, col_m2 = st.columns(2, gap="large")

    with col_m1:
        st.markdown("""
#### 📚 Referências (Vancouver)

1. Andrade Júnior VL, França MS, Santos RAF, Hatanaka AR, Cruz JJ, Hamamoto TEK,
   Traina E, Sarmento SGP, Elito Júnior J, Pares DBS, Mattar R, Araujo Júnior E,
   Moron AF. A new model based on artificial intelligence to screening preterm birth.
   **J Matern Fetal Neonatal Med.** 2023;36(2):2241100.
   DOI: [10.1080/14767058.2023.2241100](https://doi.org/10.1080/14767058.2023.2241100)

2. França MS, Andrade Júnior VL, Hatanaka AR, Santos R, et al.
   Variáveis selecionadas por Regressão Logística — Banco de Dados UNIFESP.
   (Tabela 2 fornecida pelo grupo de pesquisa.)

---
#### 🏥 Caracterização da Coorte

| Característica | Valor |
|---|---|
| N total analisado | 524 gestantes |
| Tipo de gestação | 100% únicas (Modelo 1) |
| Janela gestacional | 18–24 semanas (US) |
| Período de coleta | Out/2010 – Ago/2018 |
| Instituição | UNIFESP — EPM, São Paulo |
| sPTB < 35 semanas | 36 (6,9%) |
| Partos ≥ 35 semanas | 488 (93,1%) |
| Validação | Cross-validation 70/30% |

---
#### ⚠️ Limitações do Calculador

- **SBELM e XGBoost não reproduzidos:** pesos da rede neural e da árvore de
  boosting não foram publicados.
- **Validado em centro terciário:** população referenciada para CL —
  desempenho pode ser inferior em população geral de baixo risco.
- **Estudo retrospectivo:** menor poder estatístico que RCT prospectivo;
  fase prospectiva está em andamento.
- **Uso educacional e de pesquisa:** não substitui avaliação clínica
  individualizada nem laudo ultrassonográfico formal.
""")

    with col_m2:
        st.markdown("""
#### 🤖 Como Funciona a Regressão Logística

A regressão logística estima a **probabilidade** de um evento binário
(parto prematuro: sim/não) a partir de múltiplos preditores:

```
logit(p) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ

p = 1 / (1 + e^(−logit))
```

Onde:
- **β₀** = intercepto (baseline log-odds)
- **βᵢ** = coeficiente de cada variável
- **OR = e^βᵢ** = Odds Ratio (razão de chances)

| OR | Interpretação |
|---|---|
| **> 1** | Variável **aumenta** risco de PPT |
| **= 1** | Sem efeito |
| **< 1** | Variável **diminui** risco de PPT (fator protetor) |

---
#### 🏗️ Arquitetura do SBELM

```
Entradas:  CL<25mm  ┐
           LR score ├─► [Neural Network] ─► P(sPTB<35s)
           XGBoost  ┘     1 hidden layer
                           3 neurônios
                           Tanh → SoftMax
```

**Importância relativa dos componentes (SHAP):**
- XGBoost: ~50%
- CL < 25 mm: ~30%
- Regressão Logística: ~20%

Algoritmo KS test vencedor: **XGBoost** (KS = 0,4899 no conjunto de teste)

---
#### 📐 Variáveis Cervicais Inovadoras (Grupo UNIFESP)

| Medida | Descrição |
|---|---|
| **A-B** | CL reto (functional length) |
| **A-Y-B** | CL traçado / dois-vezes / ângulo interno |
| **Índice A-B/A-Y-B** | CL reto ÷ ângulo interno do colo |
| **B-C** | Istmo cervical |
| **X-Z** | Diâmetro externo do colo |

---
#### 📧 Contato Institucional

**Setor de Rastreamento e Prevenção do Parto Prematuro**
Disciplina de Medicina Fetal — Depto. de Obstetrícia
EPM-UNIFESP · Rua Napoleão de Barros, 875 · Vila Clementino
CEP 04024-002 · São Paulo, SP · Brasil
Prof. Dr. Edward Araujo Júnior: araujojred@terra.com.br
""")

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box" style="font-size:0.8rem; text-align:center;">
    Este aplicativo foi desenvolvido para fins <strong>educacionais e de pesquisa</strong>.
    Os modelos reproduzem coeficientes publicados nos artigos referenciados.
    Resultados devem ser interpretados no contexto clínico completo pelo profissional de saúde habilitado.
    Desenvolvido com ❤️ a partir do banco de dados UNIFESP — 2024/2025.
    </div>
    """, unsafe_allow_html=True)
