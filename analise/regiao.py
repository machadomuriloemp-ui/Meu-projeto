"""Recorte por estado (UF) e região.

A conta é feita OBRA A OBRA: cada obra vai para o estado onde está. Assim, uma
concessão que atravessa dois estados ou duas regiões não é problema — cada obra
conta no seu lugar.

Como o estado de cada obra é identificado (nesta ordem):
1. Sufixo da rodovia: "BR-381/MG" -> MG
2. Rodovia estadual: "PR-423" -> PR
3. Código SNV do cadastro da ANTT: "101BSC4125" -> SC
4. Concessionária que atua em um único estado (pelas demais obras dela)
O que não se encaixa fica como "não identificado" — nunca por palpite.

O investimento em R$ NÃO entra aqui: a base da ANTT só tem o total por
concessionária, sem divisão por estado.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

UFS = ("AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO").split()
REGIAO = {
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_REGIAO = {uf: reg for reg, ufs in REGIAO.items() for uf in ufs}
NOME_UF = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul", "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina",
    "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins",
}
MIN_OBRAS_CONFIAVEL = 10


def uf_pela_rodovia(rodovia) -> str | None:
    if rodovia is None or (isinstance(rodovia, float) and np.isnan(rodovia)):
        return None
    r = str(rodovia).upper().strip()
    m = re.search(r"/\s*([A-Z]{2})\b", r)
    if m and m.group(1) in UFS:
        return m.group(1)
    m = re.match(r"^([A-Z]{2})\s*-\s*\d", r)           # rodovia estadual: PR-423
    if m and m.group(1) in UFS:
        return m.group(1)
    m = re.match(r"^([A-Z]{2})-\1-\d", r)              # grafia repetida: PR-PR-323
    if m and m.group(1) in UFS:
        return m.group(1)
    return None


def uf_pelo_snv(codigo) -> str | None:
    """Código do Sistema Nacional de Viação: 3 dígitos + tipo + UF + trecho (101BSC4125)."""
    if codigo is None or (isinstance(codigo, float) and np.isnan(codigo)):
        return None
    m = re.match(r"^\s*\d{3}[A-Z]([A-Z]{2})\d", str(codigo).upper())
    return m.group(1) if m and m.group(1) in UFS else None


def atribuir_uf(obras: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as colunas 'uf', 'regiao' e 'uf_origem' (como o estado foi identificado)."""
    o = obras.copy()
    uf = pd.Series(None, index=o.index, dtype="object")
    origem = pd.Series("não identificado", index=o.index, dtype="object")
    for col, fonte, fn in [("rodovia", "rodovia", uf_pela_rodovia),
                           ("rodovia_cadastro", "rodovia (cadastro)", uf_pela_rodovia),
                           ("snv_cadastro", "código SNV", uf_pelo_snv)]:
        if col in o:
            achado = o[col].map(fn)
            novo = uf.isna() & achado.notna()
            uf[novo] = achado[novo]
            origem[novo] = fonte
    # Concessionária que atua em um só estado: as obras sem UF herdam esse estado
    conhecidas = pd.DataFrame({"c": o["concessionaria"], "uf": uf}).dropna()
    unico = conhecidas.groupby("c")["uf"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).dropna()
    falta = uf.isna() & o["concessionaria"].isin(unico.index)
    uf[falta] = o.loc[falta, "concessionaria"].map(unico)
    origem[falta] = "concessionária de um só estado"
    o["uf"] = uf
    o["uf_origem"] = origem
    o["regiao"] = o["uf"].map(UF_REGIAO)
    return o


def _agrega(hist: pd.DataFrame, atual: pd.DataFrame, chave: str) -> pd.DataFrame:
    h = (hist.groupby(chave)
         .agg(metas_encerradas=("atrasou", "size"), metas_atrasadas=("atrasou", "sum"),
              concessionarias_hist=("concessionaria", lambda s: sorted(s.unique())))
         .reset_index())
    h["taxa_atraso_pct"] = h["metas_atrasadas"] / h["metas_encerradas"] * 100
    a = (atual.groupby(chave)
         .agg(obras_ano_atual=("prob_atraso", "size"),
              obras_em_risco=("prob_atraso", lambda s: int((s >= config.LIMIAR_RISCO).sum())),
              chance_media_pct=("prob_atraso", lambda s: float(s.mean() * 100)),
              concessionarias_atual=("concessionaria", lambda s: sorted(s.unique())))
         .reset_index())
    if not a.empty:
        a["pct_em_risco"] = a["obras_em_risco"] / a["obras_ano_atual"] * 100
    r = h.merge(a, on=chave, how="outer")
    r["concessionarias"] = [sorted(set((x if isinstance(x, list) else []) + (y if isinstance(y, list) else [])))
                            for x, y in zip(r.get("concessionarias_hist", []), r.get("concessionarias_atual", []))]
    r = r.drop(columns=[c for c in ["concessionarias_hist", "concessionarias_atual"] if c in r])
    r["poucos_dados"] = r["metas_encerradas"].fillna(0) < MIN_OBRAS_CONFIAVEL
    return r


def resumo(obras: pd.DataFrame, previsoes: pd.DataFrame, risco: pd.DataFrame) -> dict:
    """Atraso histórico e risco no ano avaliado, por UF e por região (geral e por concessionária)."""
    o = atribuir_uf(obras)
    hist = o.dropna(subset=["atrasou"])
    # Ano avaliado de cada concessionária (o mesmo do ranking)
    if previsoes is not None and not previsoes.empty and risco is not None and not risco.empty:
        p = atribuir_uf(previsoes)
        chaves = set(zip(risco["concessionaria"], risco["ano"].astype(int)))
        atual = p[[(c, int(a)) in chaves for c, a in zip(p["concessionaria"], p["ano"])]]
    else:
        atual = pd.DataFrame(columns=["concessionaria", "uf", "regiao", "prob_atraso"])

    sem_uf_hist = int(hist["uf"].isna().sum())
    sem_uf_atual = int(atual["uf"].isna().sum()) if "uf" in atual else 0
    por_uf = _agrega(hist.dropna(subset=["uf"]), atual.dropna(subset=["uf"]), "uf")
    por_uf["regiao"] = por_uf["uf"].map(UF_REGIAO)
    por_uf["nome"] = por_uf["uf"].map(NOME_UF)
    por_regiao = _agrega(hist.dropna(subset=["regiao"]), atual.dropna(subset=["regiao"]), "regiao")
    por_regiao["ufs_com_dados"] = por_regiao["regiao"].map(
        lambda rg: sorted(set(por_uf.loc[por_uf["regiao"] == rg, "uf"])))

    por_conc = {}
    for c in sorted(set(o["concessionaria"].dropna())):
        hc, ac = hist[hist["concessionaria"] == c], atual[atual["concessionaria"] == c]
        pu = _agrega(hc.dropna(subset=["uf"]), ac.dropna(subset=["uf"]), "uf")
        if not pu.empty:
            pu["regiao"] = pu["uf"].map(UF_REGIAO)
            pu["nome"] = pu["uf"].map(NOME_UF)
        por_conc[c] = {"por_uf": pu,
                       "sem_uf": int(hc["uf"].isna().sum() + (ac["uf"].isna().sum() if len(ac) else 0))}

    origem = o["uf_origem"].value_counts().to_dict()
    return {"por_uf": por_uf, "por_regiao": por_regiao, "por_concessionaria": por_conc,
            "sem_uf_hist": sem_uf_hist, "sem_uf_atual": sem_uf_atual,
            "total_hist": int(len(hist)), "total_atual": int(len(atual)),
            "origem_uf": {k: int(v) for k, v in origem.items()}}
