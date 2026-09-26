"""Checagem de consistência entre as bases da ANTT.

As bases de planejamento e acompanhamento (execução física, obra a obra) são
preenchidas pelas concessionárias separadamente da base de investimentos (R$
investido por concessionária e ano). Quando uma concessionária declara
investimento num ano, mas não informa nenhuma execução física nas obras desse
ano, o mais provável é falta de preenchimento — não obra parada.

Esses anos são sinalizados e ficam FORA das métricas de atraso e do modelo,
para não contar como atraso algo que pode ser só dado não informado.

Limitação: a base de investimentos é anual e por concessionária, sem ligação
com cada obra. Por isso a checagem é feita concessionária por concessionária,
ano a ano — não obra a obra, nem mês a mês.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .carregar import MESES
from .cumprimento import chave_obra

# Mesma concessionária com grafias diferentes entre as bases (conferidas uma a uma).
# Chave = nome na base de investimentos, normalizado; valor = nome nas demais bases.
APELIDOS = {
    "EPRLITORALPIONEIRO": "LITORALPIONEIRO",
    "VIADOSCRISTAIS": "VIACRISTAIS",
    "ROTAVERDE": "ROTAVERDEGOIAS",
}

STATUS = {
    "investimento_sem_acompanhamento": {
        "rotulo": "Investimento declarado, execução não enviada",
        "nivel": "alerta", "exclui": True,
    },
    "investimento_sem_execucao_informada": {
        "rotulo": "Investimento declarado, execução física zerada",
        "nivel": "alerta", "exclui": True,
    },
    "execucao_sem_investimento": {
        "rotulo": "Execução informada, investimento não declarado",
        "nivel": "atencao", "exclui": False,
    },
    "sem_correspondencia": {
        "rotulo": "Concessionária não encontrada na base de investimentos",
        "nivel": "atencao", "exclui": False,
    },
    "investimento_nao_publicado": {
        "rotulo": "Investimento do ano ainda não publicado",
        "nivel": "info", "exclui": False,
    },
    "sem_investimento_sem_execucao": {
        "rotulo": "Sem investimento e sem execução",
        "nivel": "info", "exclui": False,
    },
    "consistente": {
        "rotulo": "Consistente", "nivel": "ok", "exclui": False,
    },
}


def chave_nome(nome: str) -> str:
    s = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]", "", s)


def _reais(v: float) -> str:
    if v is None or pd.isna(v):
        return "–"
    if abs(v) >= 1e9:
        return f"R$ {v / 1e9:.2f} bi".replace(".", ",")
    if abs(v) >= 1e6:
        return f"R$ {v / 1e6:.1f} mi".replace(".", ",")
    return f"R$ {v:,.0f}".replace(",", ".")


def investimentos_por_nome(invest: pd.DataFrame, nomes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Liga a base de investimentos aos nomes usados nas outras bases.

    Devolve (concessionaria, ano, investimento_rs) e a lista de nomes da base de
    investimentos que não corresponderam a nenhuma concessionária analisada.
    """
    if invest is None or invest.empty or "investimento_rs" not in invest:
        return pd.DataFrame(columns=["concessionaria", "ano", "investimento_rs"]), []
    por_chave = {chave_nome(n): n for n in nomes}
    inv = invest.dropna(subset=["concessionaria", "ano"]).copy()
    inv["_k"] = inv["concessionaria"].map(chave_nome).replace(APELIDOS)
    inv["nome_base"] = inv["concessionaria"]
    inv["concessionaria"] = inv["_k"].map(por_chave)
    sem_par = sorted(inv.loc[inv["concessionaria"].isna(), "nome_base"].unique().tolist())
    inv = inv.dropna(subset=["concessionaria"])
    inv["ano"] = inv["ano"].astype(int)
    inv = (inv.groupby(["concessionaria", "ano"], as_index=False)
           .agg(investimento_rs=("investimento_rs", lambda s: s.sum(min_count=1)),
                nome_na_base_investimentos=("nome_base", "first")))
    return inv, sem_par


def _execucao_informada(acomp: pd.DataFrame) -> pd.Series:
    cols = [f"exec_{m}" for m in MESES if f"exec_{m}" in acomp]
    total = acomp.get("executado_anual", pd.Series(0.0, index=acomp.index)).fillna(0)
    if cols:
        total = np.maximum(total, acomp[cols].fillna(0).sum(axis=1))
    return pd.Series(total, index=acomp.index)


def checar(acomp: pd.DataFrame, plan: pd.DataFrame, invest: pd.DataFrame,
           hoje: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Uma linha por concessionária e ano, com o status de consistência."""
    ano_atual = (pd.Timestamp.today() if hoje is None else pd.Timestamp(hoje)).year
    a = acomp.copy()
    if "obra" not in a:
        a["obra"] = chave_obra(a)
    a["_exec"] = _execucao_informada(a)
    ac = (a.groupby(["concessionaria", "ano"])
          .agg(obras_acompanhadas=("obra", "nunique"),
               obras_com_meta=("previsto_anual", lambda s: int((s.fillna(0) > 0).sum())),
               obras_com_execucao=("_exec", lambda s: int((s > 0).sum())),
               executado_pp=("_exec", "sum"))
          .reset_index())
    if plan is not None and not plan.empty:
        p = plan.copy()
        if "obra" not in p:
            p["obra"] = chave_obra(p)
        pl = p.groupby(["concessionaria", "ano"]).agg(obras_planejadas=("obra", "nunique")).reset_index()
    else:
        pl = pd.DataFrame(columns=["concessionaria", "ano", "obras_planejadas"])
    base = ac.merge(pl, on=["concessionaria", "ano"], how="outer")
    base = base[base["ano"] <= ano_atual]
    for c in ["obras_acompanhadas", "obras_com_meta", "obras_com_execucao", "obras_planejadas"]:
        base[c] = base[c].fillna(0).astype(int)
    base["executado_pp"] = base["executado_pp"].fillna(0.0)
    base["ano"] = base["ano"].astype(int)

    nomes = sorted(set(acomp["concessionaria"].dropna()) | set(plan["concessionaria"].dropna()
                                                                 if plan is not None and not plan.empty else []))
    inv, sem_par_invest = investimentos_por_nome(invest, nomes)
    base = base.merge(inv, on=["concessionaria", "ano"], how="left")
    tem_base_invest = not inv.empty
    ultimo_ano_invest = int(inv["ano"].max()) if tem_base_invest else None
    com_invest = set(inv["concessionaria"]) if tem_base_invest else set()

    status, textos = [], []
    for r in base.itertuples(index=False):
        c, ano, v = r.concessionaria, r.ano, r.investimento_rs
        investiu = pd.notna(v) and v > 0
        if not tem_base_invest:
            st = "sem_correspondencia"
            txt = "Base de investimentos indisponível nesta atualização; não foi possível checar."
        elif ultimo_ano_invest is not None and ano > ultimo_ano_invest:
            st = "investimento_nao_publicado"
            txt = (f"A ANTT ainda não publicou o investimento de {ano} (a base é anual; o último ano "
                   f"disponível é {ultimo_ano_invest}). A checagem será feita quando sair.")
        elif c not in com_invest:
            st = "sem_correspondencia"
            txt = (f"A {c} não aparece com esse nome na base de investimentos da ANTT; "
                   "não foi possível comparar.")
        elif investiu and r.obras_acompanhadas == 0 and r.obras_planejadas > 0:
            st = "investimento_sem_acompanhamento"
            txt = (f"Em {ano}, a {c} declarou {_reais(v)} de investimento, mas não enviou o "
                   f"acompanhamento de execução de nenhuma das {r.obras_planejadas} obras planejadas. "
                   "Isso indica falta de preenchimento, não necessariamente obra parada. "
                   "Esse ano não entra no cálculo de atraso.")
        elif investiu and r.obras_com_meta > 0 and r.obras_com_execucao == 0:
            st = "investimento_sem_execucao_informada"
            txt = (f"Em {ano}, a {c} declarou {_reais(v)} de investimento, mas informou execução "
                   f"zero em todas as {r.obras_acompanhadas} obras acompanhadas. O mais provável é "
                   "falta de preenchimento. Esse ano foi retirado do cálculo de atraso para não "
                   "contar como atraso algo que pode ser só dado não informado.")
        elif not investiu and r.obras_com_execucao > 0:
            st = "execucao_sem_investimento"
            txt = (f"Em {ano}, a {c} informou execução física em {r.obras_com_execucao} obra(s), "
                   "mas não declarou investimento na base da ANTT. Os dados de execução foram mantidos.")
        elif not investiu:
            st = "sem_investimento_sem_execucao"
            txt = f"Em {ano}, a {c} não declarou investimento nem informou execução física."
        else:
            st = "consistente"
            txt = (f"Em {ano}, a {c} declarou {_reais(v)} de investimento e informou execução "
                   f"física em {r.obras_com_execucao} de {r.obras_acompanhadas} obras acompanhadas.")
        status.append(st)
        textos.append(txt)
    base["status"] = status
    base["rotulo"] = base["status"].map(lambda s: STATUS[s]["rotulo"])
    base["nivel"] = base["status"].map(lambda s: STATUS[s]["nivel"])
    base["excluido_das_metricas"] = base["status"].map(lambda s: STATUS[s]["exclui"])
    base["explicacao"] = textos
    base = base.sort_values(["concessionaria", "ano"]).reset_index(drop=True)
    return base, sem_par_invest


def pares_excluidos(qual: pd.DataFrame) -> set[tuple[str, int]]:
    if qual is None or qual.empty:
        return set()
    q = qual[qual["excluido_das_metricas"]]
    return set(zip(q["concessionaria"], q["ano"].astype(int)))


def remover_pares(df: pd.DataFrame, pares: set[tuple[str, int]]) -> pd.DataFrame:
    if df is None or df.empty or not pares:
        return df
    chave = list(zip(df["concessionaria"], df["ano"].astype(int)))
    return df[[k not in pares for k in chave]].reset_index(drop=True)
