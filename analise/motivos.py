"""Identificação dos motivos (fatores) associados ao atraso das obras.

Fontes de motivos:
* Status de pré-requisitos no planejamento: projeto executivo, licenciamento
  ambiental e desapropriação.
* Texto livre de observações, classificado por palavras-chave.
* Dinâmica do plano: número de revisões, corte do previsto entre versões,
  obra herdada do ano anterior, concentração da meta no fim do ano.
* Postergação da data de término em relação ao cronograma do cadastro/contrato.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .carregar import MESES, normalizar_nome
from .cumprimento import chave_obra

STATUS_RESOLVIDO = {"concluido", "concluida", "nao_objecao", "aprovado", "aprovada",
                    "emitida", "emitido", "obtida", "obtido", "dispensado", "dispensada",
                    "nao_se_aplica", "n_a", "na", "nao_aplicavel", "sim", "ok", "finalizado"}

CATEGORIAS_TEXTO = {
    "licenciamento": r"licenc|ambiental|ibama|fepam|inea|icmbio|\basv\b|outorga|condicionante",
    "desapropriacao": r"desaprop|\bdup\b|indeniza|imovel|imoveis|invas|ocupac|reassent|faixa de dominio",
    "projeto": r"projeto|aprovac|nao objec|anteprojeto|revisao de|analise da antt",
    "clima": r"chuva|climat|enchente|inunda|pluvio|cheia|calamidade|intemper|evento extremo",
    "interferencias": r"interferen|remanej|\brede\b|energia|saneamento|\bgas\b|fibra|prefeitura|municipio|dnit",
    "contratacao_suprimentos": r"fornec|contrata|licita|insumo|materia|mao de obra|empreiteir|equipamento",
    "judicial_regulatorio": r"judic|liminar|embargo|\btcu\b|\bmpf\b|decisao|reequilibr|aditivo|repactua|reprogram",
    "financeiro": r"financ|recurso|\bcaixa\b|credito|bndes|pagamento|orcament",
}

# Nomes neutros (usados na tabela de pesos do modelo)
NOMES = {
    "pend_projeto": "Projeto executivo pendente",
    "pend_licenciamento": "Licenciamento ambiental pendente",
    "pend_desapropriacao": "Desapropriação pendente",
    "herdada_ano_anterior": "Obra herdada do ano anterior",
    "n_versoes": "Nº de revisões do plano",
    "corte_plano_pp": "Redução da meta entre versões (pp)",
    "concentracao_fim_ano": "Fração da meta nos meses 10–12",
    "meses_postergacao": "Meses de postergação vs. cronograma original",
    "hist_atraso_concessionaria": "Taxa histórica de atraso da concessionária",
    "hist_atraso_tipo": "Taxa histórica de atraso do tipo de obra",
    "razao_ytd": "Executado ÷ previsto no ano até agora",
    "previsto_anual": "% da obra previsto para o ano",
    "exec_acum_anterior": "% executado antes do ano",
    "extensao": "Extensão (km)",
    "log_valor": "Valor contratual",
    "duracao_meses": "Prazo previsto (meses)",
    "meses_com_meta": "Nº de meses com meta",
}

# Frases no sentido do risco (usadas para explicar por que uma obra está em risco)
ROTULOS = {
    "pend_projeto": "Projeto executivo não concluído",
    "pend_licenciamento": "Licenciamento ambiental pendente",
    "pend_desapropriacao": "Desapropriação pendente",
    "herdada_ano_anterior": "Obra herdada do ano anterior (não executada)",
    "n_versoes": "Muitas revisões do plano no ano",
    "corte_plano_pp": "Meta reduzida entre versões do plano",
    "concentracao_fim_ano": "Meta concentrada no fim do ano",
    "meses_postergacao": "Término postergado vs. cronograma original",
    "hist_atraso_concessionaria": "Histórico de atraso da concessionária",
    "hist_atraso_tipo": "Tipo de obra que costuma atrasar",
    "razao_ytd": "Execução no ano acima do previsto",
    "previsto_anual": "Meta anual alta (muito % em um ano)",
    "exec_acum_anterior": "Obra já avançada em anos anteriores",
    "extensao": "Obra extensa (km)",
    "log_valor": "Obra de alto valor",
    "duracao_meses": "Prazo previsto longo",
    "meses_com_meta": "Meta espalhada por muitos meses",
}
for _cat in CATEGORIAS_TEXTO:
    ROTULOS[f"motivo_{_cat}"] = {
        "licenciamento": "Observação cita licenciamento/ambiental",
        "desapropriacao": "Observação cita desapropriação/imóveis",
        "projeto": "Observação cita projeto/aprovação",
        "clima": "Observação cita chuvas/clima",
        "interferencias": "Observação cita interferências (redes, prefeitura)",
        "contratacao_suprimentos": "Observação cita contratação/suprimentos",
        "judicial_regulatorio": "Observação cita questão judicial/regulatória",
        "financeiro": "Observação cita questão financeira",
    }[_cat]
    NOMES[f"motivo_{_cat}"] = ROTULOS[f"motivo_{_cat}"]


def rotulo_risco(variavel: str, coeficiente: float) -> str | None:
    """Frase que descreve o lado "arriscado" da variável (None se não fizer sentido)."""
    if variavel in ROTULOS and coeficiente > 0:
        return ROTULOS[variavel]
    opostos = {"razao_ytd": "Baixa execução no ano até agora",
               "meses_com_meta": "Poucos meses com meta (execução concentrada)",
               "previsto_anual": "Meta anual pequena (obra em fase inicial/final)",
               "exec_acum_anterior": "Obra no começo (pouco avanço anterior)",
               "extensao": "Obra curta (km)", "log_valor": "Obra de menor valor",
               "duracao_meses": "Prazo previsto curto"}
    return opostos.get(variavel)


def _pendente(serie: pd.Series) -> pd.Series:
    norm = serie.fillna("").map(normalizar_nome)
    return ((norm != "") & ~norm.isin(STATUS_RESOLVIDO)).astype(int)


def classificar_texto(textos: pd.Series) -> pd.DataFrame:
    norm = textos.fillna("").map(lambda t: normalizar_nome(t).replace("_", " "))
    return pd.DataFrame({f"motivo_{cat}": norm.str.contains(padrao, regex=True).astype(int)
                         for cat, padrao in CATEGORIAS_TEXTO.items()}, index=textos.index)


def filtrar_versoes_disponiveis(plan: pd.DataFrame, encerrados: pd.DataFrame,
                                mes_corte: int) -> pd.DataFrame:
    """Evita "olhar o futuro" nos anos encerrados.

    Para prever o ano corrente só conhecemos as versões do plano feitas até agora
    (mês `mes_corte`). Para o histórico ser comparável, nos anos encerrados usamos
    apenas as versões feitas até o mesmo ponto do ano — revisões feitas depois,
    quando o atraso já era conhecido, ficam de fora.
    """
    if plan.empty or "data_planejamento" not in plan or encerrados.empty:
        return plan
    chave = plan["concessionaria"].astype(str) + "#" + plan["ano"].astype(str)
    fechado = chave.isin(encerrados["concessionaria"].astype(str) + "#" + encerrados["ano"].astype(str))
    limite = pd.to_datetime(plan["ano"].fillna(1900).astype(int).astype(str) + "-01-01") \
        + pd.DateOffset(months=int(mes_corte) + 1)
    disponivel = plan["data_planejamento"].isna() | (plan["data_planejamento"] <= limite)
    filtrado = plan[~fechado | disponivel]
    # Se nenhuma versão existia até o corte, mantém a primeira versão (plano original)
    faltando = plan[fechado].copy()
    faltando["obra_tmp"] = chave_obra(faltando) + "|" + faltando["ano"].astype(str)
    presentes = set(chave_obra(filtrado) + "|" + filtrado["ano"].astype(str))
    faltando = faltando[~faltando["obra_tmp"].isin(presentes)]
    if not faltando.empty:
        ordem = [c for c in ["versao", "data_planejamento"] if c in faltando]
        faltando = faltando.sort_values(ordem).drop_duplicates("obra_tmp").drop(columns="obra_tmp")
        filtrado = pd.concat([filtrado, faltando], ignore_index=True)
    return filtrado


def fatores_planejamento(plan: pd.DataFrame) -> pd.DataFrame:
    """Um registro por (obra, ano) com os fatores extraídos da base de planejamento."""
    if plan.empty:
        return pd.DataFrame(columns=["obra", "ano"])
    plan = plan.copy()
    plan["obra"] = chave_obra(plan)
    ordem = [c for c in ["versao", "data_planejamento"] if c in plan]
    plan = plan.sort_values(ordem)

    g = plan.groupby(["obra", "ano"])
    res = g.size().rename("n_versoes").reset_index()

    if "previsto_anual" in plan:
        prim = g["previsto_anual"].first()
        ult = g["previsto_anual"].last()
        res = res.merge((prim - ult).rename("corte_plano_pp").reset_index(), on=["obra", "ano"])
        res = res.merge(prim.rename("previsto_plano_original").reset_index(), on=["obra", "ano"])

    ultima = plan.drop_duplicates(["obra", "ano"], keep="last").set_index(["obra", "ano"])
    for col, nome in [("projeto_executivo", "pend_projeto"),
                      ("licenciamento_ambiental", "pend_licenciamento"),
                      ("desapropriacao", "pend_desapropriacao")]:
        if col in ultima:
            res = res.merge(_pendente(ultima[col]).rename(nome).reset_index(), on=["obra", "ano"])
    if "tipo_planejamento" in ultima:
        herd = ultima["tipo_planejamento"].fillna("").map(normalizar_nome).str.contains("anterior")
        res = res.merge(herd.astype(int).rename("herdada_ano_anterior").reset_index(), on=["obra", "ano"])
    if "observacao" in plan:
        texto = g["observacao"].agg(lambda s: " ".join(s.dropna().astype(str)))
        res = res.merge(pd.concat([texto.rename("observacao"), classificar_texto(texto)], axis=1)
                        .reset_index(), on=["obra", "ano"])
    for col in ["data_fim_prevista", "data_inicio_prevista"]:
        if col in ultima:
            res = res.merge(ultima[col].rename(f"{col}_plan").reset_index(), on=["obra", "ano"])
    return res


def fatores_cadastro(cad: pd.DataFrame) -> pd.DataFrame:
    """Valor contratual e cronograma de referência (cadastro) por obra."""
    if cad.empty:
        return pd.DataFrame(columns=["obra"])
    cad = cad.copy()
    cad["obra"] = chave_obra(cad)
    cols = {"valor_contratual": "valor_contratual",
            "data_fim_prevista": "data_fim_contrato",
            "data_inicio_prevista": "data_inicio_contrato",
            "extensao": "extensao_cadastro",
            "codigo_snv_inicial": "snv_cadastro",
            "rodovia": "rodovia_cadastro"}
    manter = ["obra"] + [c for c in cols if c in cad]
    return cad[manter].rename(columns=cols).drop_duplicates("obra", keep="last")


def concentracao_fim_ano(df: pd.DataFrame) -> pd.Series:
    fim = sum(df.get(f"prev_{m}", 0).fillna(0) for m in (10, 11, 12))
    total = sum(df.get(f"prev_{m}", 0).fillna(0) for m in MESES)
    return pd.Series(np.where(total > 0, fim / np.maximum(total, 1e-9), np.nan), index=df.index)


def fatores_de_atraso(obras: pd.DataFrame, fatores: list[str]) -> pd.DataFrame:
    """Para cada fator binário: quanto ele aumenta a chance de atraso (lift).

    lift = P(atraso | fator presente) / P(atraso). Acima de 1 indica associação
    com atraso. Só considera obras de anos encerrados (com resultado conhecido).
    """
    base = obras.dropna(subset=["atrasou"])
    if base.empty:
        return pd.DataFrame()
    taxa_geral = base["atrasou"].mean()
    linhas = []
    for f in fatores:
        if f not in base:
            continue
        presente = base[f].fillna(0) > 0
        n = int(presente.sum())
        if n == 0:
            continue
        taxa = base.loc[presente, "atrasou"].mean()
        linhas.append({
            "fator": ROTULOS.get(f, f), "coluna": f, "obras_com_fator": n,
            "taxa_atraso_com_fator_pct": taxa * 100,
            "taxa_atraso_sem_fator_pct": base.loc[~presente, "atrasou"].mean() * 100
            if (~presente).any() else np.nan,
            "lift": taxa / taxa_geral if taxa_geral > 0 else np.nan,
            "pct_dos_atrasos": base.loc[presente, "atrasou"].sum() / max(base["atrasou"].sum(), 1) * 100,
        })
    return pd.DataFrame(linhas).sort_values("lift", ascending=False) if linhas else pd.DataFrame()


def motivos_por_tipo(obras: pd.DataFrame, minimo: int = 3) -> pd.DataFrame:
    base = obras.dropna(subset=["atrasou"])
    if base.empty or "tipo" not in base:
        return pd.DataFrame()
    r = base.groupby("tipo").agg(obras=("atrasou", "size"), taxa_atraso_pct=("atrasou", "mean"))
    r["taxa_atraso_pct"] *= 100
    return r[r["obras"] >= minimo].sort_values("taxa_atraso_pct", ascending=False).reset_index()


def motivos_por_concessionaria(obras: pd.DataFrame, fatores: list[str]) -> pd.DataFrame:
    """Fatores mais frequentes entre as obras atrasadas de cada concessionária."""
    base = obras[obras["atrasou"] == 1]
    linhas = []
    for conc, g in base.groupby("concessionaria"):
        contagem = {f: int((g[f].fillna(0) > 0).sum()) for f in fatores if f in g}
        top = sorted(((v, k) for k, v in contagem.items() if v > 0), reverse=True)[:3]
        linhas.append({
            "concessionaria": conc, "obras_atrasadas": len(g),
            "principais_fatores": "; ".join(f"{ROTULOS.get(k, k)} ({v})" for v, k in top)
            or "Sem motivo registrado nas bases",
        })
    return pd.DataFrame(linhas)


def fatores_por_concessionaria(obras: pd.DataFrame, fatores: list[str]) -> pd.DataFrame:
    """Para cada concessionária: em quantas das obras atrasadas cada fator aparece."""
    base = obras.dropna(subset=["atrasou"])
    linhas = []
    for conc, g in base.groupby("concessionaria"):
        atrasadas = g[g["atrasou"] == 1]
        if atrasadas.empty:
            continue
        for f in fatores:
            if f not in g:
                continue
            com = atrasadas[f].fillna(0) > 0
            if not com.any():
                continue
            todas_com = g[f].fillna(0) > 0
            linhas.append({
                "concessionaria": conc, "fator": ROTULOS.get(f, f),
                "obras_atrasadas_com_fator": int(com.sum()),
                "pct_das_atrasadas": com.mean() * 100,
                "taxa_atraso_com_fator_pct": g.loc[todas_com, "atrasou"].mean() * 100,
                "obras_com_fator": int(todas_com.sum()),
            })
    df = pd.DataFrame(linhas)
    return df.sort_values(["concessionaria", "pct_das_atrasadas"], ascending=[True, False]) if not df.empty else df


def tipos_por_concessionaria(obras: pd.DataFrame, minimo: int = 2) -> pd.DataFrame:
    base = obras.dropna(subset=["atrasou"])
    if base.empty or "tipo" not in base:
        return pd.DataFrame()
    r = (base.groupby(["concessionaria", "tipo"])
         .agg(obras=("atrasou", "size"), taxa_atraso_pct=("atrasou", "mean")).reset_index())
    r["taxa_atraso_pct"] *= 100
    return r[r["obras"] >= minimo].sort_values(["concessionaria", "taxa_atraso_pct"], ascending=[True, False])
