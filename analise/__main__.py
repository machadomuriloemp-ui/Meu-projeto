"""Execução completa: python -m analise [--offline] [--dados PASTA] [--saida PASTA]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import (baixar, carregar, config, cumprimento, dashboard, impacto, modelo, motivos, obras,
               relatorio)


def executar(offline: bool = False, pasta_dados: Path | None = None,
             pasta_saida: Path | None = None, hoje: str | None = None,
             pasta_dashboard: Path | None = None) -> dict:
    """hoje: data de referência (padrão: data atual). Usado nos testes."""
    if pasta_dados is not None:
        config.PASTA_DADOS = Path(pasta_dados)
    if pasta_dashboard is None:
        # Com pasta de saída própria (ex.: testes), não mexe no dashboard publicado
        pasta_dashboard = Path(pasta_saida) / "docs" if pasta_saida else config.PASTA_DASHBOARD
    pasta_saida = Path(pasta_saida) if pasta_saida else config.PASTA_SAIDA

    print("== 1/5 Dados")
    arquivos = baixar.baixar_tudo(offline=offline)
    bases = carregar.carregar(arquivos)
    acomp_bruto = bases.get("acompanhamento", pd.DataFrame())
    plan_bruto = bases.get("planejamento", pd.DataFrame())
    if acomp_bruto.empty:
        raise SystemExit("A base de acompanhamento está vazia; não há como medir o cumprimento.")

    print("== 2/5 Cumprimento")
    acomp = cumprimento.ultima_versao(acomp_bruto)
    plan_ult = cumprimento.ultima_versao(plan_bruto)
    plan_orig = cumprimento.primeira_versao(plan_bruto)
    fechados = cumprimento.meses_fechados(acomp, hoje=hoje)
    serie = cumprimento.serie_mensal(acomp, fechados)
    tend = cumprimento.tendencia(serie) if not serie.empty else pd.DataFrame()
    anual = cumprimento.resumo_anual(acomp, fechados, plan_orig)

    print("== 3/5 Motivos")
    corte = obras.mes_corte(fechados)
    encerrados = fechados[fechados["meses_fechados"] == 12]
    plan_disponivel = motivos.filtrar_versoes_disponiveis(plan_bruto, encerrados, corte)
    fat_plan = motivos.fatores_planejamento(plan_disponivel)
    fat_cad = motivos.fatores_cadastro(bases.get("cadastro", pd.DataFrame()))
    acomp_inicial = cumprimento.primeira_versao(acomp_bruto)
    tabela = obras.montar(acomp, fechados, fat_plan, fat_cad, plan_ult, acomp_inicial)
    lista_fatores = [v for v in modelo.VARIAVEIS if v in tabela]
    binarios = [v for v in lista_fatores if v.startswith(("pend_", "motivo_", "herdada"))]
    # Variáveis contínuas viram "presente" quando acima de um limiar interpretável
    derivados = {"meses_postergacao": tabela.get("meses_postergacao", 0) > 1,
                 "corte_plano_pp": tabela.get("corte_plano_pp", 0) > 1,
                 "n_versoes": tabela.get("n_versoes", 0) >= 3,
                 "concentracao_fim_ano": tabela.get("concentracao_fim_ano", 0) > 0.5}
    tab_fatores = tabela.copy()
    for k, v in derivados.items():
        if k in tabela:
            tab_fatores[k] = pd.Series(v, index=tabela.index).astype(int)
            binarios.append(k)
    fatores = motivos.fatores_de_atraso(tab_fatores, binarios)
    motivos_conc = motivos.motivos_por_concessionaria(tab_fatores, binarios)
    motivos_tipo = motivos.motivos_por_tipo(tabela)
    fatores_conc = motivos.fatores_por_concessionaria(tab_fatores, binarios)
    tipos_conc = motivos.tipos_por_concessionaria(tabela)

    print("== 4/5 Modelo de previsão")
    m = modelo.ModeloAtraso().treinar(tabela)
    previsoes = m.prever(tabela)
    risco = modelo.risco_concessionarias(previsoes, tabela, hoje=hoje)
    # Próximo ano: só existe quando a ANTT já publicou os planos dele
    risco_proximo = pd.DataFrame()
    if not previsoes.empty and not risco.empty:
        ano_ref = int(risco["ano"].mode().iloc[0])
        futuras = previsoes[previsoes["ano"] == ano_ref + 1]
        if not futuras.empty:
            risco_proximo = modelo.risco_concessionarias(futuras, tabela, hoje=hoje)
    backtest = modelo.resumo_backtest(m.backtest)
    print(f"  {m.metricas}")

    print("== 5/5 Impacto e relatório")
    imp_ano = impacto.impacto_historico(tabela)
    inex = impacto.inexecucao_oficial(bases.get("inexecucao", pd.DataFrame()))
    imp = impacto.resumo_impacto(imp_ano, inex)

    info = (f"{acomp_bruto['concessionaria'].nunique()} concessionária(s), "
            f"anos {int(acomp['ano'].min())}–{int(acomp['ano'].max())}, "
            f"{acomp['obra'].nunique()} obras acompanhadas.")
    resultado = {
        "serie": serie, "tendencia": tend, "anual": anual, "fatores": fatores,
        "motivos_conc": motivos_conc, "motivos_tipo": motivos_tipo,
        "fatores_conc": fatores_conc, "tipos_conc": tipos_conc,
        "importancia": m.importancia(), "metricas": m.metricas, "previsoes": previsoes,
        "risco": risco, "impacto": imp, "impacto_ano": imp_ano, "inexecucao": inex,
        "obras": tabela, "info_dados": info, "risco_proximo": risco_proximo, "backtest": backtest,
    }
    destino = relatorio.gerar(resultado, pasta_saida)
    print(f"Relatório: {destino}")
    print(f"Dashboard: {dashboard.exportar(resultado, Path(pasta_dashboard) / 'dados.json')}")
    return resultado


def main():
    p = argparse.ArgumentParser(description="Análise de cumprimento do planejado - ANTT")
    p.add_argument("--offline", action="store_true", help="não baixa; usa CSVs já salvos")
    p.add_argument("--dados", type=Path, help="pasta com subpastas acompanhamento/, planejamento/...")
    p.add_argument("--saida", type=Path, help="pasta onde salvar o relatório")
    a = p.parse_args()
    executar(offline=a.offline, pasta_dados=a.dados, pasta_saida=a.saida)


if __name__ == "__main__":
    main()
