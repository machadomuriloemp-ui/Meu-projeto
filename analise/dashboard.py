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


def _obras(prev: pd.DataFrame) -> list[dict]:
    if prev is None or prev.empty:
        return []
    p = prev.copy()
    p["prob"] = (p["prob_atraso"] * 100).round(1)
    p["ano"] = p["ano"].astype(int)
    p = p.sort_values("prob", ascending=False)
    return _linhas(p, ["concessionaria", "ano", "situacao", "descricao", "tipo", "rodovia", "km_inicial",
                       "previsto_anual", "executado_ytd", "prob", "motivos_provaveis"])


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
        "risco": _linhas(risco, ["concessionaria", "ano", "risco", "prob_nao_cumprir_plano_pct",
                                 "cumprimento_esperado_pct", "cumprimento_p10_p90", "obras_no_plano",
                                 "obras_em_risco", "km_em_risco", "valor_em_risco_rs"]),
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
        "motivos_concessionaria": _linhas(r.get("motivos_conc"),
                                          ["concessionaria", "obras_atrasadas", "principais_fatores"]),
        "impacto": _linhas(r.get("impacto"), ["concessionaria", "obras_atrasadas", "km_nao_entregues",
                                              "valor_nao_executado_rs", "postergacao_media_meses",
                                              "inexecucao_media_pct", "fator_d_total_pct"]),
        "obras": _obras(r.get("previsoes")),
    }
    destino.write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return destino
