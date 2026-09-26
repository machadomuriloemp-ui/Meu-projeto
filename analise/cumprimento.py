"""Medição do cumprimento do planejado: mês a mês, ano a ano e tendência."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .carregar import MESES

TETO_INDICE = 150.0       # executar 3x o previsto num mês não deve dominar a média
MIN_MESES_TENDENCIA = 6


def chave_obra(df: pd.DataFrame) -> pd.Series:
    """Identificador único da obra: concessionária + id SIGICOR (ou item/descrição)."""
    if "id_sigicor" in df and df["id_sigicor"].notna().any():
        ident = df["id_sigicor"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    else:
        ident = pd.Series("", index=df.index)
    reserva = (df.get("item_per", pd.Series("", index=df.index)).fillna("").astype(str) + "|"
               + df.get("descricao", pd.Series("", index=df.index)).fillna("").astype(str) + "|"
               + df.get("km_inicial", pd.Series("", index=df.index)).astype(str))
    ident = ident.where(ident != "", reserva)
    return df["concessionaria"].astype(str) + "#" + ident


def ultima_versao(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só a versão mais recente de cada obra em cada ano."""
    if df.empty:
        return df
    df = df.copy()
    df["obra"] = chave_obra(df)
    ordem = [c for c in ["versao", "data_planejamento"] if c in df]
    df = df.sort_values(ordem) if ordem else df
    return df.drop_duplicates(["obra", "ano"], keep="last").reset_index(drop=True)


def primeira_versao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["obra"] = chave_obra(df)
    ordem = [c for c in ["versao", "data_planejamento"] if c in df]
    df = df.sort_values(ordem) if ordem else df
    return df.drop_duplicates(["obra", "ano"], keep="first").reset_index(drop=True)


def meses_fechados(acomp: pd.DataFrame, hoje: pd.Timestamp | None = None) -> pd.DataFrame:
    """Quantos meses de cada (concessionária, ano) já estão encerrados.

    O ano está encerrado (12 meses) quando: já existe um ano posterior da mesma
    concessionária; ou o relatório foi enviado a partir de março do ano seguinte
    (o ano de concessão já terminou); ou já se passou mais de um ano inteiro.
    Nesses casos, meses zerados no fim do ano CONTAM como não execução.

    Só no ano realmente em curso considera-se fechado até o último mês com
    alguma execução registrada (meses futuros vêm zerados).
    """
    hoje = pd.Timestamp.today().normalize() if hoje is None else pd.Timestamp(hoje)
    linhas = []
    for (conc, ano), g in acomp.groupby(["concessionaria", "ano"]):
        ultimo_ano = acomp.loc[acomp["concessionaria"] == conc, "ano"].max()
        relatorio = g["data_planejamento"].max() if "data_planejamento" in g else pd.NaT
        encerrado = (ano < ultimo_ano
                     or hoje.year > ano + 1
                     or (pd.notna(relatorio) and relatorio >= pd.Timestamp(int(ano) + 1, 3, 1)))
        if encerrado:
            fechados = 12
        else:
            fechados = 0
            for m in MESES:
                col = f"exec_{m}"
                if col in g and g[col].fillna(0).gt(0).any():
                    fechados = m
        linhas.append({"concessionaria": conc, "ano": ano, "meses_fechados": fechados})
    return pd.DataFrame(linhas)


def serie_mensal(acomp: pd.DataFrame, fechados: pd.DataFrame) -> pd.DataFrame:
    """Previsto x executado por concessionária e mês (apenas meses encerrados)."""
    partes = []
    for m in MESES:
        pc, ec = f"prev_{m}", f"exec_{m}"
        if pc not in acomp or ec not in acomp:
            continue
        p = acomp[["concessionaria", "ano", "obra", pc, ec]].rename(
            columns={pc: "previsto", ec: "executado"})
        p["mes"] = m
        partes.append(p)
    if not partes:
        return pd.DataFrame()
    longo = pd.concat(partes, ignore_index=True)
    longo[["previsto", "executado"]] = longo[["previsto", "executado"]].fillna(0)
    longo = longo.merge(fechados, on=["concessionaria", "ano"])
    longo = longo[longo["mes"] <= longo["meses_fechados"]]

    com_meta = longo["previsto"] > 0
    longo["cumpriu"] = np.where(
        com_meta, longo["executado"] >= config.TOLERANCIA_CUMPRIMENTO * longo["previsto"], np.nan)

    serie = longo.groupby(["concessionaria", "ano", "mes"]).agg(
        previsto=("previsto", "sum"),
        executado=("executado", "sum"),
        obras_com_meta=("cumpriu", "count"),
        obras_cumpriram=("cumpriu", "sum"),
    ).reset_index()
    serie["indice_cumprimento"] = np.where(
        serie["previsto"] > 0, serie["executado"] / serie["previsto"] * 100, np.nan)
    serie["pct_obras_cumpriram"] = np.where(
        serie["obras_com_meta"] > 0, serie["obras_cumpriram"] / serie["obras_com_meta"] * 100, np.nan)
    serie = serie.sort_values(["concessionaria", "ano", "mes"])
    g = serie.groupby(["concessionaria", "ano"])
    serie["previsto_acum"] = g["previsto"].cumsum()
    serie["executado_acum"] = g["executado"].cumsum()
    serie["desvio_acum_pp"] = serie["executado_acum"] - serie["previsto_acum"]
    serie["periodo"] = serie["ano"].astype(int).astype(str) + "-M" + serie["mes"].astype(int).astype(str).str.zfill(2)
    serie["t"] = serie["ano"] * 12 + serie["mes"]
    return serie.reset_index(drop=True)


def tendencia(serie: pd.DataFrame) -> pd.DataFrame:
    """Tendência do índice de cumprimento ao longo dos meses, por concessionária.

    Usa a reta de Theil-Sen (mediana das inclinações), que não é arrastada por
    meses isolados com percentuais extremos, e o teste de Kendall para dizer se
    a tendência é estatisticamente relevante. O índice é limitado a 150%.
    """
    linhas = []
    for conc, g in serie.groupby("concessionaria"):
        g = g.dropna(subset=["indice_cumprimento"]).sort_values("t")
        y = g["indice_cumprimento"].clip(0, TETO_INDICE).to_numpy()
        x = g["t"].to_numpy(dtype=float)
        linha = {"concessionaria": conc, "meses_com_meta": len(g),
                 "cumprimento_medio": float(np.nanmean(y)) if len(y) else np.nan,
                 "cumprimento_ult_6m": float(np.nanmean(y[-6:])) if len(y) else np.nan,
                 "pct_meses_cumpridos": float(np.mean(y >= config.TOLERANCIA_CUMPRIMENTO * 100) * 100)
                 if len(y) else np.nan}
        if len(g) >= MIN_MESES_TENDENCIA and np.ptp(x) > 0 and np.ptp(y) > 0:
            linha["inclinacao_pp_ano"] = stats.theilslopes(y, x)[0] * 12
            linha["p_valor"] = stats.kendalltau(x, y).pvalue
        else:
            linha["inclinacao_pp_ano"] = np.nan
            linha["p_valor"] = np.nan
        inc, p = linha["inclinacao_pp_ano"], linha["p_valor"]
        if np.isnan(inc):
            linha["tendencia"] = "Dados insuficientes"
        elif p < 0.10 and inc >= config.LIMIAR_TENDENCIA_PP_ANO:
            linha["tendencia"] = "Melhorando"
        elif p < 0.10 and inc <= -config.LIMIAR_TENDENCIA_PP_ANO:
            linha["tendencia"] = "Piorando"
        else:
            linha["tendencia"] = "Estável"
        linhas.append(linha)
    return pd.DataFrame(linhas)


def resumo_anual(acomp: pd.DataFrame, fechados: pd.DataFrame,
                 plano_original: pd.DataFrame | None) -> pd.DataFrame:
    """Cumprimento anual (anos encerrados) contra o plano vigente e o plano original."""
    df = acomp.merge(fechados, on=["concessionaria", "ano"])
    df = df[(df["meses_fechados"] == 12) & (df["previsto_anual"].fillna(0) > 0)].copy()
    if df.empty:
        return pd.DataFrame()
    df["cumpriu"] = df["executado_anual"].fillna(0) >= config.TOLERANCIA_CUMPRIMENTO * df["previsto_anual"]
    res = df.groupby(["concessionaria", "ano"]).agg(
        obras=("obra", "nunique"),
        previsto_pp=("previsto_anual", "sum"),
        executado_pp=("executado_anual", "sum"),
        obras_cumpriram=("cumpriu", "sum"),
    ).reset_index()
    res["cumprimento_pct"] = res["executado_pp"] / res["previsto_pp"] * 100
    res["pct_obras_cumpriram"] = res["obras_cumpriram"] / res["obras"] * 100
    if plano_original is not None and not plano_original.empty and "previsto_anual" in plano_original:
        orig = plano_original[["obra", "ano", "previsto_anual"]].rename(
            columns={"previsto_anual": "previsto_original"})
        m = df[["concessionaria", "ano", "obra", "executado_anual"]].merge(orig, on=["obra", "ano"])
        o = m.groupby(["concessionaria", "ano"]).agg(
            previsto_original_pp=("previsto_original", "sum"),
            exec_obras_orig=("executado_anual", "sum")).reset_index()
        res = res.merge(o, on=["concessionaria", "ano"], how="left")
        res["cumprimento_vs_plano_original_pct"] = np.where(
            res["previsto_original_pp"] > 0,
            res["exec_obras_orig"] / res["previsto_original_pp"] * 100, np.nan)
        res = res.drop(columns=["exec_obras_orig"])
    return res
