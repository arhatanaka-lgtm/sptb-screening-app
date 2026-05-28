# 🔬 Rastreamento de Parto Prematuro Espontâneo < 35 semanas

**Modelos de Inteligência Artificial — UNIFESP**  
Setor de Rastreamento e Prevenção do Parto Prematuro — EPM-UNIFESP, São Paulo, Brasil

---

## 📚 Referência Principal

> Andrade Júnior VL, França MS, Santos RAF, **Hatanaka AR**, Cruz JJ, Hamamoto TEK,
> Traina E, Sarmento SGP, Elito Júnior J, Pares DBS, Mattar R, Araujo Júnior E, Moron AF.
> **A new model based on artificial intelligence to screening preterm birth.**
> *J Matern Fetal Neonatal Med.* 2023;36(2):2241100.
> DOI: [10.1080/14767058.2023.2241100](https://doi.org/10.1080/14767058.2023.2241100)

---

## 🎯 Sobre o App

Calculadora clínica interativa que implementa os modelos de Regressão Logística
publicados pelo grupo de Medicina Fetal da UNIFESP para rastreamento de **parto prematuro
espontâneo (sPTB) < 35 semanas** no segundo trimestre de gestação (18–24 semanas).

### Modelos Implementados

| Modelo | AUC | Sensib. (FPR 10%) | Espec. (FPR 10%) |
|---|---|---|---|
| CL < 25mm (gold standard) | 0,318 | 33,3% | 91,8% |
| **LR — Andrade Júnior et al. (2023)** | **0,749** | **38,9%** | **92,2%** |
| **LR — França MS et al. (UNIFESP)** | — | — | — |
| XGBoost *(não reproduzível)* | 0,788 | 44,4% | 92,6% |
| SBELM Neural Network *(não reproduzível)* | 0,808 | 47,2% | 92,8% |

> ⚠️ **Limitação:** Os modelos XGBoost e SBELM (rede neural) apresentaram o melhor desempenho
> no artigo, porém seus pesos internos não foram publicados e **não podem ser reproduzidos** neste
> calculador. Apenas os modelos de Regressão Logística (coeficientes β publicados) estão implementados.

---

## 🚀 Deploy Local

```bash
# Clonar repositório
git clone https://github.com/arhatanaka-lgtm/sptb-screening-app.git
cd sptb-screening-app

# Instalar dependências
pip install -r requirements.txt

# Executar app
streamlit run app.py
```

---

## 📁 Estrutura do Projeto

```
sptb-screening-app/
├── app.py           # Interface principal (Streamlit)
├── models.py        # Coeficientes e funções preditivas
├── utils.py         # Gráficos e funções auxiliares
├── requirements.txt # Dependências Python
├── README.md        # Este arquivo
└── .streamlit/
    └── config.toml  # Tema (azul UNIFESP)
```

---

## 🔬 Variáveis do Modelo (Andrade Júnior et al.)

| Variável | OR | IC 95% | p-valor |
|---|---|---|---|
| Presença de funil cervical | 5,971 | 1,234–28,891 | **0,026** |
| Índice CL/Ângulo ≤ 0,200 | 1,533 | 0,481–4,887 | 0,470 |
| Sem antibiótico na gestação | 1,698 | 0,628–4,594 | 0,297 |
| PPT anterior < 37 semanas | 4,851 | 1,661–14,172 | **0,004** |
| Uma curetagem prévia | 0,288 | 0,034–2,447 | 0,254 |
| Peso ≤ 58 kg | 2,012 | 0,727–5,568 | 0,178 |
| Sem tabagismo | 0,387 | 0,107–1,399 | 0,148 |
| CL < 30,9 mm | 1,863 | 0,622–5,580 | 0,266 |
| **Constante (β₀)** | **0,044** | | **< 0,001** |

---

## ⚠️ Aviso Legal

> Este aplicativo foi desenvolvido para fins **exclusivamente educacionais e de pesquisa**.
> Os resultados **não substituem** o julgamento clínico individualizado, o laudo
> ultrassonográfico formal, nem a consulta com especialista em Medicina Fetal.
> O uso clínico direto sem validação prospectiva em sua população não é recomendado.

---

## 📧 Contato

**Prof. Dr. Edward Araujo Júnior**  
Setor de Rastreamento e Prevenção do Parto Prematuro  
Disciplina de Medicina Fetal — Departamento de Obstetrícia  
EPM-UNIFESP · São Paulo, SP · Brasil  
📧 araujojred@terra.com.br
