"""Monta a tabela "obra x ano" com resultado (atrasou?) e todas as variáveis
explicativas usadas pelo modelo de previsão."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .carregar import MESES
from .cumprimento import chave_obra
from .motivos import concentracao_fim_ano

SUAVIZACAO = 5  # peso da média geral nas taxas históricas (evita extremos com poucas obras)


def _ytd(df: pd.DataFrame, meses: pd.Series) -> tuple[pd.Series, pd.Series]:
    prev = pd.Series(0.0, index=df.index)
    exe = pd.Series(0.0, index=df.index)
    for m in MESES:
        dentro = meses >= m
        prev += np.where(dentro, df.get(f"prev_{m}", 0).fillna(0) if f"prev_{m}" in df else 0, 0)
        exe += np.where(dentro, df.get(f"exec_{m}", 0).fillna(0) if f"exec_{m}" in df else 0, 0)
    return prev, exe


def _taxa_historica(obras: pd.DataFrame, grupo: str) -> pd.Series:
    """Taxa de atraso do grupo usando só anos ANTERIORES ao da linha (sem vazamento)."""
    hist = obras.dropna(subset=["atrasou"])
    geral = hist["atrasou"].mean() if len(hist) else 0.5
    resultado = pd.Series(np.nan, index=obras.index)
    if grupo not in obras:
        return resultado.fillna(geral)
    for ano in obras["ano"].dropna().unique():
        anteriores = hist[hist["ano"] < ano]
        if anteriores.empty:
            resultado[obras["ano"] == ano] = geral
            continue
        est = anteriores.groupby(grupo)["atrasou"].agg(["sum", "count"])
        taxa = (est["sum"] + SUAVIZACAO * geral) / (est["count"] + SUAVIZACAO)
        linhas = obras["ano"] == ano
        resultado[linhas] = obras.loc[linhas, grupo].map(taxa).fillna(geral).to_numpy()
    return resultado.fillna(geral)


def mes_corte(fechados: pd.DataFrame) -> int:
    """Mês do ano corrente até onde já há execução informada (mediana entre concessionárias)."""
    abertos = fechados.loc[fechados["meses_fechados"] < 12, "meses_fechados"]
    return int(abertos.median()) if len(abertos) else 0


def montar(acomp: pd.DataFrame, fechados: pd.DataFrame, fat_plan: pd.DataFrame,
           fat_cad: pd.DataFrame, plan_ultima: pd.DataFrame,
           acomp_inicial: pd.DataFrame | None = None) -> pd.DataFrame:
    obras = acomp.merge(fechados, on=["concessionaria", "ano"], how="left")
    obras["situacao"] = np.where(obras["meses_fechados"] == 12, "encerrado", "em andamento")
    if acomp_inicial is not None and "data_fim_prevista" in acomp_inicial:
        obras = obras.merge(acomp_inicial[["obra", "ano", "data_fim_prevista"]]
                            .rename(columns={"data_fim_prevista": "data_fim_inicial"}),
                            on=["obra", "ano"], how="left")

    # Obras planejadas para anos ainda sem acompanhamento (ex.: ano seguinte)
    if not plan_ultima.empty:
        ult_acomp = acomp.groupby("concessionaria")["ano"].max()
        fut = plan_ultima.copy()
        fut["obra"] = chave_obra(fut)
        limite = fut["concessionaria"].map(ult_acomp).fillna(-1)
        fut = fut[fut["ano"] > limite].copy()
        if not fut.empty:
            fut["meses_fechados"] = 0
            fut["situacao"] = "planejado"
            obras = pd.concat([obras, fut[[c for c in fut.columns if c in obras.columns or
                                           c in ("obra", "meses_fechados", "situacao")]]],
                              ignore_index=True)

    obras = obras.merge(fat_plan.drop(columns=[c for c in ["concessionaria"] if c in fat_plan]),
                        on=["obra", "ano"], how="left")
    obras = obras.merge(fat_cad, on="obra", how="left")

    if "extensao" not in obras:
        obras["extensao"] = np.nan
    if "extensao_cadastro" in obras:
        obras["extensao"] = obras["extensao"].fillna(obras["extensao_cadastro"])
    if "valor_contratual" not in obras:
        obras["valor_contratual"] = np.nan
    obras["log_valor"] = np.log1p(obras["valor_contratual"])

    prev_ano = obras["previsto_anual"].fillna(0)
    encerrado = obras["situacao"] == "encerrado"
    obras["atrasou"] = np.where(
        encerrado & (prev_ano > 0),
        (obras["executado_anual"].fillna(0) < config.TOLERANCIA_CUMPRIMENTO * prev_ano).astype(float),
        np.nan)

    obras["meses_com_meta"] = sum((obras.get(f"prev_{m}", 0).fillna(0) > 0).astype(int)
                                  if f"prev_{m}" in obras else 0 for m in MESES)
    obras["concentracao_fim_ano"] = concentracao_fim_ano(obras)

    # Data de término conhecida no momento da previsão (sem olhar o futuro):
    # anos encerrados -> plano disponível até o mês de corte, ou 1ª versão do acompanhamento;
    # ano corrente/futuro -> informação mais recente.
    vazio = pd.Series(pd.NaT, index=obras.index, dtype="datetime64[ns]")
    fim_plan = obras.get("data_fim_prevista_plan", vazio)
    fim_atual = obras.get("data_fim_prevista", vazio)
    fim_inicial = obras.get("data_fim_inicial", vazio)
    encerrado_ = obras["situacao"] == "encerrado"
    fim = fim_plan.fillna(fim_inicial.where(encerrado_, fim_atual))
    ini = obras.get("data_inicio_prevista", vazio)
    if "data_inicio_prevista_plan" in obras:
        ini = obras["data_inicio_prevista_plan"].fillna(ini)
    obras["duracao_meses"] = (fim - ini).dt.days / 30.44
    if "data_fim_contrato" in obras:
        obras["meses_postergacao"] = ((fim - obras["data_fim_contrato"]).dt.days / 30.44).clip(lower=0)
    else:
        obras["meses_postergacao"] = np.nan

    # Execução acumulada no ano até o mês de referência
    corte_treino = mes_corte(fechados)
    meses_ref = np.where(obras["situacao"] == "em andamento", obras["meses_fechados"],
                         np.where(obras["situacao"] == "encerrado", corte_treino, 0))
    prev_ytd, exec_ytd = _ytd(obras, pd.Series(meses_ref, index=obras.index))
    obras["mes_referencia"] = meses_ref
    obras["previsto_ytd"] = prev_ytd
    obras["executado_ytd"] = exec_ytd
    obras["desvio_ytd_pp"] = np.where(prev_ytd > 0, prev_ytd - exec_ytd, 0.0)
    obras["razao_ytd"] = np.where(prev_ytd > 0, (exec_ytd / prev_ytd).clip(0, 2), 1.0)
    obras.attrs["mes_corte_treino"] = corte_treino

    obras["hist_atraso_concessionaria"] = _taxa_historica(obras, "concessionaria")
    obras["hist_atraso_tipo"] = _taxa_historica(obras, "tipo")
    return obras
