"""Teste de ponta a ponta: a análise precisa reencontrar o "gabarito" embutido
nas bases simuladas (ver testes/dados_simulados.py)."""
import pandas as pd


def test_arquivos_de_saida(resultado_simulado):
    saida = resultado_simulado["_saida"]
    assert (saida / "RELATORIO.md").stat().st_size > 2000
    assert (saida / "analise_concessoes.xlsx").exists()
    for g in ["tendencia.png", "risco.png", "fatores.png"]:
        assert (saida / "graficos" / g).exists()
    abas = pd.ExcelFile(saida / "analise_concessoes.xlsx").sheet_names
    for aba in ["risco_concessionarias", "tendencia", "previsao_obras", "fatores_atraso", "impacto"]:
        assert aba in abas


def test_ranking_de_risco_reencontra_o_gabarito(resultado_simulado):
    risco = resultado_simulado["risco"]
    assert set(risco["concessionaria"]) == {"VIA SUL", "VIA COSTEIRA", "RIOSP", "WAY 262"}
    assert risco.iloc[0]["concessionaria"] == "VIA SUL"          # pior
    assert risco.iloc[-1]["concessionaria"] == "VIA COSTEIRA"    # melhor
    assert risco.iloc[-1]["risco"] in {"Baixo", "Médio"}
    assert risco["prob_nao_cumprir_plano_pct"].between(0, 100).all()
    assert (risco["ano"] == 2026).all()


def test_tendencia_detectada(resultado_simulado):
    t = resultado_simulado["tendencia"].set_index("concessionaria")
    assert t.loc["VIA SUL", "tendencia"] == "Piorando"
    assert t.loc["VIA COSTEIRA", "tendencia"] == "Melhorando"
    assert t.loc["RIOSP", "tendencia"] != "Piorando"


def test_meses_futuros_nao_contam_como_atraso(resultado_simulado):
    serie = resultado_simulado["serie"]
    assert serie.loc[serie["ano"] == 2026, "mes"].max() == 6
    assert serie.loc[serie["ano"] == 2025, "mes"].max() == 12


def test_previsoes_por_obra(resultado_simulado):
    p = resultado_simulado["previsoes"]
    assert not p.empty
    assert (p["situacao"] != "encerrado").all()
    assert p["prob_atraso"].between(0, 1).all()
    assert p["motivos_provaveis"].str.len().gt(0).all()
    # Obras do ano corrente que já estão atrasadas no mês 6 devem ter risco maior
    corr = p[p["ano"] == 2026]
    atrasadas = corr["razao_ytd"] < 0.8
    assert corr.loc[atrasadas, "prob_atraso"].mean() > corr.loc[~atrasadas, "prob_atraso"].mean() + 0.3


def test_qualidade_do_modelo(resultado_simulado):
    m = resultado_simulado["metricas"]
    assert m["auc_validacao_cruzada"] >= 0.75
    assert m["auc_teste_ultimo_ano"] >= 0.75


def test_motivos_reais_sao_identificados(resultado_simulado):
    f = resultado_simulado["fatores"].set_index("coluna")
    assert f.loc["pend_licenciamento", "lift"] > 1.2
    assert f.loc["pend_desapropriacao", "lift"] > 1.1


def test_sem_vazamento_de_informacao_futura(resultado_simulado):
    """Revisões do plano feitas depois do corte (quando o atraso já é conhecido)
    não podem ser usadas. Nos dados simulados, o corte da meta só aparece nessas
    versões tardias; se ele virar um preditor forte, há vazamento."""
    f = resultado_simulado["fatores"].set_index("coluna")
    if "corte_plano_pp" in f.index:
        assert f.loc["corte_plano_pp", "lift"] < 1.3


def test_impacto(resultado_simulado):
    imp = resultado_simulado["impacto"].set_index("concessionaria")
    assert imp.loc["VIA SUL", "obras_atrasadas"] > imp.loc["VIA COSTEIRA", "obras_atrasadas"]
    assert imp.loc["VIA SUL", "valor_nao_executado_rs"] > 0
    assert imp.loc["VIA SUL", "km_nao_entregues"] > 0
    assert imp.loc["VIA SUL", "fator_d_total_pct"] > 0      # veio da base de inexecução
    assert pd.isna(imp.loc["RIOSP", "fator_d_total_pct"])   # RIOSP não está na inexecução
