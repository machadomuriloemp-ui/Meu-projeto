"""Geração do relatório (Markdown legível no GitHub, gráficos e Excel)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

CORES_RISCO = {"Alto": "#c0392b", "Médio": "#e67e22", "Baixo": "#27ae60"}


# ----------------------------------------------------------------- formatação
def pct(v, casas=0):
    return "–" if v is None or pd.isna(v) else f"{v:.{casas}f}%".replace(".", ",")


def num(v, casas=1):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def reais(v):
    if v is None or pd.isna(v) or v == 0:
        return "–"
    if abs(v) >= 1e9:
        return f"R$ {num(v / 1e9, 2)} bi"
    if abs(v) >= 1e6:
        return f"R$ {num(v / 1e6, 1)} mi"
    return f"R$ {num(v, 0)}"


def tabela(df: pd.DataFrame, colunas: dict[str, tuple[str, callable]]) -> str:
    if df is None or df.empty:
        return "_Sem dados suficientes._\n"
    cab = "| " + " | ".join(t for t, _ in colunas.values()) + " |"
    sep = "|" + "|".join("---" for _ in colunas) + "|"
    linhas = [cab, sep]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join(str(f(r[c])) if c in r else "–"
                                         for c, (_, f) in colunas.items()) + " |")
    return "\n".join(linhas) + "\n"


txt = str


def inteiro(v):
    return "–" if pd.isna(v) else f"{int(v)}"


# -------------------------------------------------------------------- gráficos
def grafico_tendencia(serie: pd.DataFrame, destino: Path) -> bool:
    if serie.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for conc, g in serie.groupby("concessionaria"):
        g = g.sort_values("t")
        movel = g["indice_cumprimento"].clip(0, 150).rolling(3, min_periods=1).mean()
        ax.plot(g["t"], movel, marker="o", ms=3, lw=1.8, label=conc)
    ax.axhline(config.TOLERANCIA_CUMPRIMENTO * 100, color="gray", ls="--", lw=1)
    ax.axhline(100, color="black", lw=0.6)
    ticks = serie.drop_duplicates("t").sort_values("t")
    passo = max(1, len(ticks) // 10)
    ax.set_xticks(ticks["t"].iloc[::passo])
    ax.set_xticklabels(ticks["periodo"].iloc[::passo], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Executado ÷ previsto (%)\nmédia móvel 3 meses")
    ax.set_title("Cumprimento do planejado mês a mês")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return True


def grafico_risco(risco: pd.DataFrame, destino: Path) -> bool:
    if risco.empty:
        return False
    r = risco.sort_values("prob_nao_cumprir_plano_pct")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(r) + 1.5))
    ax.barh(r["concessionaria"], r["prob_nao_cumprir_plano_pct"],
            color=[CORES_RISCO.get(x, "gray") for x in r["risco"]])
    for i, v in enumerate(r["prob_nao_cumprir_plano_pct"]):
        ax.text(v + 1, i, pct(v), va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probabilidade de não cumprir o plano do ano (%)")
    ax.set_title("Risco de atraso por concessionária")
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return True


def grafico_fatores(fatores: pd.DataFrame, destino: Path) -> bool:
    if fatores.empty:
        return False
    f = fatores[fatores["obras_com_fator"] >= 3].head(10).sort_values("lift")
    if f.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(f) + 1.5))
    ax.barh(f["fator"], f["lift"], color=np.where(f["lift"] > 1, "#c0392b", "#7f8c8d"))
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xlabel("Quantas vezes aumenta a chance de atraso (1 = neutro)")
    ax.set_title("Fatores associados ao atraso")
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return True


# ------------------------------------------------------------------- relatório
def gerar(r: dict, pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    img = pasta / "graficos"
    img.mkdir(exist_ok=True)
    tem_tend = grafico_tendencia(r["serie"], img / "tendencia.png")
    tem_risco = grafico_risco(r["risco"], img / "risco.png")
    tem_fat = grafico_fatores(r["fatores"], img / "fatores.png")

    md = [f"# Cumprimento do planejado — concessões rodoviárias (ANTT)\n",
          f"_Gerado em {datetime.now():%d/%m/%Y %H:%M}. Fonte: Portal de Dados Abertos da ANTT "
          f"(SIGICOR). {r['info_dados']}_\n"]

    # ---- Resumo
    md.append("## Resumo\n")
    bullets = []
    risco, tend, fat = r["risco"], r["tendencia"], r["fatores"]
    if not risco.empty:
        top = risco.iloc[0]
        bullets.append(f"**Maior risco de atraso:** {top['concessionaria']} — "
                       f"{pct(top['prob_nao_cumprir_plano_pct'])} de chance de não cumprir o plano de "
                       f"{top['ano']} ({top['obras_em_risco']} obra(s) em risco).")
        altos = risco[risco["risco"] == "Alto"]["concessionaria"].tolist()
        if altos:
            bullets.append(f"**Risco alto:** {', '.join(altos)}.")
    if not tend.empty:
        pior = tend[tend["tendencia"] == "Piorando"]["concessionaria"].tolist()
        melhor = tend[tend["tendencia"] == "Melhorando"]["concessionaria"].tolist()
        if pior:
            bullets.append(f"**Tendência piorando:** {', '.join(pior)}.")
        if melhor:
            bullets.append(f"**Tendência melhorando:** {', '.join(melhor)}.")
    if not fat.empty:
        f = fat[(fat["lift"] > 1) & (fat["obras_com_fator"] >= 3)].head(3)
        if not f.empty:
            bullets.append("**Motivos mais associados a atraso:** " + "; ".join(
                f"{x.fator} ({num(x.lift, 1)}×)" for x in f.itertuples()) + ".")
    md.append("\n".join(f"- {b}" for b in bullets) + "\n" if bullets else "_Sem destaques._\n")

    # ---- 1. Risco
    md.append("## 1. Quem tende a atrasar (ano corrente)\n")
    md.append("Probabilidade de a concessionária executar menos de "
              f"{pct(config.TOLERANCIA_CUMPRIMENTO * 100)} do que planejou para o ano, estimada "
              "a partir das probabilidades de atraso de cada obra.\n")
    md.append(tabela(risco, {
        "concessionaria": ("Concessionária", txt), "ano": ("Ano", inteiro),
        "risco": ("Risco", txt), "prob_nao_cumprir_plano_pct": ("Chance de não cumprir", pct),
        "cumprimento_esperado_pct": ("Cumprimento esperado", pct),
        "obras_em_risco": ("Obras em risco", inteiro)}))
    if tem_risco:
        md.append("![Risco](graficos/risco.png)\n")

    # ---- 2. Tendência
    md.append("## 2. Histórico e tendência mês a mês\n")
    md.append("Índice = soma do % executado ÷ soma do % previsto das obras no mês. "
              "Tendência = inclinação da reta ao longo dos meses (pontos percentuais por ano).\n")
    md.append(tabela(tend, {
        "concessionaria": ("Concessionária", txt),
        "cumprimento_medio": ("Cumprimento médio", pct),
        "cumprimento_ult_6m": ("Últimos 6 meses", pct),
        "pct_meses_cumpridos": ("Meses cumpridos", pct),
        "tendencia": ("Tendência", txt),
        "inclinacao_pp_ano": ("pp/ano", lambda v: num(v, 1))}))
    if tem_tend:
        md.append("![Tendência](graficos/tendencia.png)\n")
    if not r["anual"].empty:
        md.append("### Cumprimento por ano encerrado\n")
        cols = {"concessionaria": ("Concessionária", txt), "ano": ("Ano", inteiro),
                "obras": ("Obras", inteiro), "cumprimento_pct": ("Executado ÷ previsto", pct),
                "pct_obras_cumpriram": ("Obras que cumpriram", pct)}
        if "cumprimento_vs_plano_original_pct" in r["anual"]:
            cols["cumprimento_vs_plano_original_pct"] = ("vs. plano original", pct)
        md.append(tabela(r["anual"], cols))

    # ---- 3. Motivos
    md.append("## 3. Motivos de atraso\n")
    md.append("**Lift** = quantas vezes a presença do fator aumenta a chance de a obra atrasar, "
              "comparado à média. Baseado nas obras de anos encerrados.\n")
    md.append(tabela(fat[fat["obras_com_fator"] >= 1].head(12) if not fat.empty else fat, {
        "fator": ("Fator", txt), "obras_com_fator": ("Obras", inteiro),
        "taxa_atraso_com_fator_pct": ("Atraso com fator", pct),
        "taxa_atraso_sem_fator_pct": ("Sem fator", pct),
        "lift": ("Lift", lambda v: num(v, 1) + "×")}))
    if tem_fat:
        md.append("![Fatores](graficos/fatores.png)\n")
    md.append("### Por concessionária\n")
    md.append(tabela(r["motivos_conc"], {
        "concessionaria": ("Concessionária", txt), "obras_atrasadas": ("Obras atrasadas", inteiro),
        "principais_fatores": ("Principais fatores", txt)}))
    md.append("### Tipos de obra que mais atrasam\n")
    md.append(tabela(r["motivos_tipo"].head(10), {
        "tipo": ("Tipo de obra", txt), "obras": ("Obras", inteiro),
        "taxa_atraso_pct": ("Taxa de atraso", pct)}))
    if not r["importancia"].empty:
        md.append("### Peso de cada fator no modelo\n")
        md.append(tabela(r["importancia"].head(10), {
            "fator": ("Fator", txt), "efeito": ("Efeito", txt),
            "peso": ("Peso", lambda v: num(v, 2))}))

    # ---- 4. Impacto
    md.append("## 4. Impacto\n")
    md.append("Déficit = % previsto − % executado. Km e R$ = déficit aplicado à extensão e ao "
              "valor contratual (quando informados). Fator D = desconto na tarifa aplicado pela ANTT "
              "por inexecução.\n")
    md.append(tabela(r["impacto"], {
        "concessionaria": ("Concessionária", txt), "obras_atrasadas": ("Obras atrasadas", inteiro),
        "km_nao_entregues": ("Km não entregues", lambda v: num(v, 1)),
        "valor_nao_executado_rs": ("Valor não executado", reais),
        "postergacao_media_meses": ("Postergação média (meses)", lambda v: num(v, 1)),
        "inexecucao_media_pct": ("Inexecução oficial", pct),
        "fator_d_total_pct": ("Fator D", lambda v: pct(v, 2))}))
    if not risco.empty:
        md.append("**Impacto esperado no ano corrente** (probabilidade × o que falta executar):\n")
        md.append(tabela(risco, {
            "concessionaria": ("Concessionária", txt),
            "valor_em_risco_rs": ("Valor em risco", reais),
            "km_em_risco": ("Km em risco", lambda v: num(v, 1)),
            "cumprimento_p10_p90": ("Faixa provável de cumprimento", txt)}))

    # ---- 5. Obras
    md.append("## 5. Obras com maior probabilidade de atraso\n")
    prev = r["previsoes"]
    if not prev.empty:
        prev = prev.sort_values("prob_atraso", ascending=False).head(15)
    if not prev.empty:
        prev = prev.assign(rotulo=prev.apply(_rotulo_obra, axis=1))
    md.append(tabela(prev, {
        "concessionaria": ("Concessionária", txt),
        "rotulo": ("Obra", txt),
        "prob_atraso": ("Chance de atraso", lambda v: pct(v * 100)),
        "motivos_provaveis": ("Motivos prováveis", txt)}))
    md.append("_Lista completa no Excel (aba `previsao_obras`)._\n")

    # ---- 6. Modelo
    md.append("## 6. Confiabilidade do modelo\n")
    m = r["metricas"]
    linhas = [f"- Obras-ano usadas no treino: {m.get('obras_treino', 0)} "
              f"(taxa histórica de atraso: {pct(m.get('taxa_atraso_historica_pct'))})"]
    if "modelo_escolhido" in m:
        linhas.append(f"- Modelo escolhido: {m['modelo_escolhido']} (comparação AUC: {m['comparacao']})")
        linhas.append(f"- AUC na validação cruzada: {num(m['auc_validacao_cruzada'], 2)} "
                      "(0,5 = acaso; 0,7 = razoável; 0,8+ = bom)")
        if "auc_teste_ultimo_ano" in m:
            linhas.append(f"- AUC treinando só com anos anteriores e testando em {m['ano_teste']}: "
                          f"{num(m['auc_teste_ultimo_ano'], 2)}")
    if "observacao" in m:
        linhas.append(f"- {m['observacao']}")
    md.append("\n".join(linhas) + "\n")

    md.append("## Como ler e limitações\n")
    md.append(
        f"- Uma obra **atrasou** quando executou menos de {pct(config.TOLERANCIA_CUMPRIMENTO * 100)} "
        "do % previsto para o ano.\n"
        "- Meses ainda não informados pela concessionária não contam como descumprimento.\n"
        "- As bases são preenchidas pelas próprias concessionárias no SIGICOR; nem todas as "
        "concessões estão no sistema e campos como observações e valor contratual podem estar vazios.\n"
        "- Os motivos são **associações** estatísticas e registros das bases, não prova de causa.\n"
        "- O índice mensal trata cada obra com o mesmo peso (% de avanço físico).\n")

    destino = pasta / "RELATORIO.md"
    destino.write_text("\n".join(md), encoding="utf-8")

    # ---- Excel e CSV
    abas = {"risco_concessionarias": risco, "tendencia": tend, "serie_mensal": r["serie"],
            "cumprimento_anual": r["anual"], "fatores_atraso": fat,
            "motivos_concessionaria": r["motivos_conc"], "motivos_tipo_obra": r["motivos_tipo"],
            "impacto": r["impacto"], "impacto_por_ano": r["impacto_ano"],
            "inexecucao_oficial": r["inexecucao"], "previsao_obras": _colunas_obras(r["previsoes"]),
            "peso_fatores": r["importancia"]}
    with pd.ExcelWriter(pasta / "analise_concessoes.xlsx") as xls:
        for nome, df in abas.items():
            if df is not None and not df.empty:
                df.to_excel(xls, sheet_name=nome[:31], index=False)
                df.to_csv(pasta / f"{nome}.csv", index=False, sep=";", decimal=",",
                          encoding="utf-8-sig")
    return destino


def _rotulo_obra(r) -> str:
    desc = str(r.get("descricao")) if pd.notna(r.get("descricao")) else "–"
    desc = desc if len(desc) <= 55 else desc[:52] + "..."
    local = [str(r["rodovia"])] if pd.notna(r.get("rodovia")) else []
    if pd.notna(r.get("km_inicial")):
        local.append(f"km {num(r['km_inicial'], 1)}")
    return f"{desc} ({', '.join(local)})" if local else desc


def _colunas_obras(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["concessionaria", "ano", "situacao", "id_sigicor", "item_per", "descricao", "rodovia",
            "km_inicial",
            "tipo", "extensao", "valor_contratual", "previsto_anual", "mes_referencia",
            "previsto_ytd", "executado_ytd", "prob_atraso", "em_risco", "motivos_provaveis",
            "observacao"]
    return df[[c for c in cols if c in df]].sort_values("prob_atraso", ascending=False)
