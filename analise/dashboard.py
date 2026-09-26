"""Exporta os resultados em JSON para o dashboard web (docs/index.html)."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TETO = 150.0


def _limpo(v):
    """Converte valores do pandas/numpy para JSON (NaN -> null)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) or math.isinf(v) else round(float(v), 3)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _linhas(df: pd.DataFrame | None, colunas: list[str]) -> list[dict]:
    if df is None or df.empty:
        return []
    cols = [c for c in colunas if c in df.columns]
    return [{c: _limpo(v) for c, v in zip(cols, linha)} for linha in df[cols].itertuples(index=False)]


def _serie(serie: pd.DataFrame) -> dict:
    if serie is None or serie.empty:
        return {"por_concessionaria": {}, "geral": []}
    s = serie.copy()
    s["indice"] = s["indice_cumprimento"].clip(0, TETO)
    por = {conc: _linhas(g.sort_values("t"), ["periodo", "t", "indice", "previsto", "executado"])
           for conc, g in s.groupby("concessionaria")}
    geral = (s.groupby(["t", "periodo"]).agg(previsto=("previsto", "sum"), executado=("executado", "sum"),
                                              concessionarias=("concessionaria", "nunique"))
             .reset_index().sort_values("t"))
    geral["indice"] = np.where(geral["previsto"] > 0,
                               (geral["executado"] / geral["previsto"] * 100).clip(0, TETO), np.nan)
    return {"por_concessionaria": por,
            "geral": _linhas(geral, ["periodo", "t", "indice", "previsto", "executado", "concessionarias"])}


def obras_para_o_painel(prev: pd.DataFrame, anos_validos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lista de obras mostrada no painel.

    * Só entram os anos que estão sendo avaliados para cada concessionária (o ano
      do ranking e, se houver plano, o seguinte) — planos de anos que já passaram
      sem acompanhamento não são "previsões".
    * A ANTT cadastra vários itens idênticos (mesma descrição, rodovia e km, ids
      diferentes — ex.: cada acesso de um segmento). Eles viram um card só, com a
      quantidade de itens.
    """
    if prev is None or prev.empty:
        return pd.DataFrame()
    p = prev.copy()
    p["ano"] = p["ano"].astype(int)
    if anos_validos is not None and not anos_validos.empty:
        chaves = set(zip(anos_validos["concessionaria"], anos_validos["ano"].astype(int)))
        p = p[[(c, a) in chaves for c, a in zip(p["concessionaria"], p["ano"])]]
    if p.empty:
        return p
    p["prob"] = (p["prob_atraso"] * 100).round(1)
    for c in ["descricao", "tipo", "rodovia"]:
        if c not in p:
            p[c] = ""
        p[c] = p[c].astype("string").fillna("")
    if "km_inicial" not in p:
        p["km_inicial"] = np.nan
    p["_km"] = p["km_inicial"].round(3).fillna(-1)
    # "BR-364" e "BR-364/RO" são a mesma rodovia escrita de dois jeitos
    p["_rod"] = (p["rodovia"].str.upper().str.replace(r"\s+", "", regex=True)
                 .str.replace(r"/[A-Z]{2}$", "", regex=True))
    p["_desc"] = p["descricao"].str.strip().str.upper()
    p = p.sort_values("prob", ascending=False)
    chave = ["concessionaria", "ano", "_desc", "_rod", "_km", "tipo"]
    grupos = p.groupby(chave, sort=False, dropna=False)
    agrup = grupos.first().reset_index()   # a linha de maior probabilidade de cada grupo
    agrup["itens"] = grupos.size().to_numpy()
    agrup["prob_media"] = grupos["prob"].mean().round(1).to_numpy()
    # mostra a forma mais completa do nome da rodovia (ex.: "BR-364/RO")
    agrup["rodovia"] = grupos["rodovia"].agg(lambda s: max(s, key=len)).to_numpy()
    agrup = agrup.drop(columns=["_km", "_rod", "_desc"], errors="ignore")
    return agrup.sort_values("prob", ascending=False).reset_index(drop=True)


def _obras(prev: pd.DataFrame, anos_validos: pd.DataFrame | None = None) -> list[dict]:
    p = obras_para_o_painel(prev, anos_validos)
    return _linhas(p, ["concessionaria", "ano", "situacao", "descricao", "tipo", "rodovia", "km_inicial",
                       "previsto_anual", "executado_ytd", "prob", "prob_media", "itens", "motivos_provaveis"])


COLS_RISCO = ["concessionaria", "ano", "risco", "prob_nao_cumprir_plano_pct", "cumprimento_esperado_pct",
              "cumprimento_p10_p90", "obras_no_plano", "obras_em_risco", "km_em_risco",
              "valor_em_risco_rs", "obras_para_proximo_ano", "pp_para_proximo_ano",
              "km_para_proximo_ano"]


def _backtest(bt: dict) -> dict:
    if not bt:
        return {}
    lista = lambda val: [{c: _limpo(v) for c, v in linha.items()} for linha in val]  # noqa: E731
    saida = {}
    for k, val in bt.items():
        if isinstance(val, list):
            saida[k] = lista(val)
        elif isinstance(val, dict):
            saida[k] = {c: lista(v) for c, v in val.items()}
        else:
            saida[k] = _limpo(val)
    return saida


def _agrupar(df: pd.DataFrame | None, colunas: list[str]) -> dict:
    """{concessionaria: [linhas]} para os detalhes de cada concessionária no painel."""
    if df is None or df.empty:
        return {}
    return {c: _linhas(g, colunas) for c, g in df.groupby("concessionaria")}


def _resumo_concessionarias(obras: pd.DataFrame) -> dict:
    if obras is None or obras.empty:
        return {}
    res = {}
    for c, g in obras.groupby("concessionaria"):
        hist = g.dropna(subset=["atrasou"])
        res[c] = {"obras": int(g["obra"].nunique()),
                  "anos": [int(g["ano"].min()), int(g["ano"].max())],
                  "obras_ano_historico": int(len(hist)),
                  "atrasadas_historico": int(hist["atrasou"].sum()) if len(hist) else 0,
                  "taxa_atraso_historica_pct": _limpo(hist["atrasou"].mean() * 100) if len(hist) else None}
    return res


def exportar(r: dict, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    m = r.get("metricas", {})
    obras = r.get("obras", pd.DataFrame())
    risco = r.get("risco", pd.DataFrame())
    hist = obras.dropna(subset=["atrasou"]) if not obras.empty else obras
    dados = {
        "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fonte": "Portal de Dados Abertos da ANTT (SIGICOR)",
        "tolerancia_pct": config.TOLERANCIA_CUMPRIMENTO * 100,
        "resumo": {
            "concessionarias": int(obras["concessionaria"].nunique()) if not obras.empty else 0,
            "obras": int(obras["obra"].nunique()) if not obras.empty else 0,
            "anos": [int(obras["ano"].min()), int(obras["ano"].max())] if not obras.empty else [],
            "obras_ano_historico": int(len(hist)),
            "taxa_atraso_historica_pct": _limpo(hist["atrasou"].mean() * 100) if len(hist) else None,
            "risco_alto": int((risco["risco"] == "Alto").sum()) if not risco.empty else 0,
            "ano_referencia": int(risco["ano"].mode().iloc[0]) if not risco.empty else None,
        },
        "modelo": {
            "nome": m.get("modelo_escolhido", "Regra baseada no histórico"),
            "auc_cv": _limpo(m.get("auc_validacao_cruzada")),
            "auc_teste": _limpo(m.get("auc_teste_ultimo_ano")),
            "ano_teste": m.get("ano_teste"),
            "comparacao": m.get("comparacao", {}),
            "obras_treino": m.get("obras_treino"),
        },
        "risco": _linhas(risco, COLS_RISCO),
        "risco_proximo": _linhas(r.get("risco_proximo"), COLS_RISCO),
        "backtest": _backtest(r.get("backtest", {})),
        "tendencia": _linhas(r.get("tendencia"), ["concessionaria", "cumprimento_medio", "cumprimento_ult_6m",
                                                   "pct_meses_cumpridos", "tendencia", "inclinacao_pp_ano",
                                                   "meses_com_meta"]),
        "serie": _serie(r.get("serie")),
        "anual": _linhas(r.get("anual"), ["concessionaria", "ano", "obras", "cumprimento_pct",
                                          "pct_obras_cumpriram", "cumprimento_vs_plano_original_pct"]),
        "fatores": _linhas(r["fatores"][r["fatores"]["obras_com_fator"] >= 3]
                           if not r["fatores"].empty else r["fatores"],
                           ["fator", "obras_com_fator", "taxa_atraso_com_fator_pct",
                            "taxa_atraso_sem_fator_pct", "lift"]),
        "tipos": _linhas(r["motivos_tipo"].head(12) if not r["motivos_tipo"].empty else r["motivos_tipo"],
                         ["tipo", "obras", "taxa_atraso_pct"]),
        "por_concessionaria": _resumo_concessionarias(obras),
        "qualidade": _linhas(r.get("qualidade"), [
            "concessionaria", "ano", "status", "rotulo", "nivel", "excluido_das_metricas",
            "investimento_rs", "obras_planejadas", "obras_acompanhadas", "obras_com_execucao",
            "explicacao"]),
        "sem_par_investimentos": list(r.get("sem_par_investimentos") or []),
        "fatores_por_concessionaria": _agrupar(r.get("fatores_conc"), [
            "fator", "obras_atrasadas_com_fator", "pct_das_atrasadas", "taxa_atraso_com_fator_pct",
            "obras_com_fator"]),
        "tipos_por_concessionaria": _agrupar(r.get("tipos_conc"), ["tipo", "obras", "taxa_atraso_pct"]),
        "impacto_por_ano": _agrupar(r.get("impacto_ano"), [
            "ano", "obras_atrasadas", "deficit_total_pp", "km_nao_entregues", "postergacao_media_meses"]),
        "motivos_concessionaria": _linhas(r.get("motivos_conc"),
                                          ["concessionaria", "obras_atrasadas", "principais_fatores"]),
        "impacto": _linhas(r.get("impacto"), ["concessionaria", "obras_atrasadas", "km_nao_entregues",
                                              "valor_nao_executado_rs", "postergacao_media_meses",
                                              "inexecucao_media_pct", "fator_d_total_pct"]),
        "obras": _obras(r.get("previsoes"), pd.concat(
            [x[["concessionaria", "ano"]] for x in (r.get("risco"), r.get("risco_proximo"))
             if x is not None and not x.empty], ignore_index=True) if not risco.empty else None),
    }
    destino.write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return destino
