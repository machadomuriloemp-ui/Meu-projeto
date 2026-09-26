"""Impacto dos atrasos: o que deixou de ser entregue (%, km, R$, meses) e o
que a ANTT registrou como inexecução e Fator D (desconto na tarifa)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cumprimento import ultima_versao


def impacto_historico(obras: pd.DataFrame) -> pd.DataFrame:
    base = obras[(obras["situacao"] == "encerrado") & (obras["previsto_anual"].fillna(0) > 0)].copy()
    if base.empty:
        return pd.DataFrame()
    base["deficit_pp"] = (base["previsto_anual"] - base["executado_anual"].fillna(0)).clip(lower=0)
    base["km_nao_entregues"] = base["deficit_pp"] / 100 * base["extensao"].fillna(0)
    base["valor_nao_executado_rs"] = base["deficit_pp"] / 100 * base["valor_contratual"].fillna(0)
    res = base.groupby(["concessionaria", "ano"]).agg(
        obras_atrasadas=("atrasou", "sum"),
        deficit_total_pp=("deficit_pp", "sum"),
        km_nao_entregues=("km_nao_entregues", "sum"),
        valor_nao_executado_rs=("valor_nao_executado_rs", "sum"),
        postergacao_media_meses=("meses_postergacao", "mean"),
    ).reset_index()
    return res


def inexecucao_oficial(inex: pd.DataFrame) -> pd.DataFrame:
    """Resumo da base oficial de inexecução (quando disponível)."""
    if inex.empty or "inexecucao" not in inex:
        return pd.DataFrame()
    inex = ultima_versao(inex)
    agg = {"obras": ("obra", "nunique"), "inexecucao_media_pct": ("inexecucao", "mean"),
           "obras_com_inexecucao": ("inexecucao", lambda s: int((s.fillna(0) > 0).sum()))}
    if "fator_d" in inex:
        agg["fator_d_total_pct"] = ("fator_d", "sum")
    return inex.groupby(["concessionaria", "ano"]).agg(**agg).reset_index()


def resumo_impacto(hist: pd.DataFrame, oficial: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame()
    res = hist.groupby("concessionaria").agg(
        anos=("ano", "nunique"),
        obras_atrasadas=("obras_atrasadas", "sum"),
        km_nao_entregues=("km_nao_entregues", "sum"),
        valor_nao_executado_rs=("valor_nao_executado_rs", "sum"),
        postergacao_media_meses=("postergacao_media_meses", "mean"),
    ).reset_index()
    if not oficial.empty:
        o = oficial.groupby("concessionaria").agg(
            inexecucao_media_pct=("inexecucao_media_pct", "mean"),
            **({"fator_d_total_pct": ("fator_d_total_pct", "sum")}
               if "fator_d_total_pct" in oficial else {})).reset_index()
        res = res.merge(o, on="concessionaria", how="left")
    return res.sort_values("obras_atrasadas", ascending=False)
