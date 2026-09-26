"""Consistência entre a base de investimentos e a execução física informada."""
import json

import pandas as pd

import dados_simulados
from analise import qualidade


def test_ano_com_investimento_e_execucao_zerada_e_detectado(resultado_simulado):
    q = resultado_simulado["qualidade"].set_index(["concessionaria", "ano"])
    conc, ano = dados_simulados.CASO_SEM_PREENCHIMENTO
    linha = q.loc[(conc, ano)]
    assert linha["status"] == "investimento_sem_execucao_informada"
    assert linha["excluido_das_metricas"]
    assert "falta de preenchimento" in linha["explicacao"]


def test_ano_suspeito_fica_fora_das_metricas_e_do_modelo(resultado_simulado):
    conc, ano = dados_simulados.CASO_SEM_PREENCHIMENTO
    obras = resultado_simulado["obras"]
    assert obras[(obras["concessionaria"] == conc) & (obras["ano"] == ano)].empty
    serie = resultado_simulado["serie"]
    assert serie[(serie["concessionaria"] == conc) & (serie["ano"] == ano)].empty
    anual = resultado_simulado["anual"]
    assert anual[(anual["concessionaria"] == conc) & (anual["ano"] == ano)].empty


def test_grafias_diferentes_do_nome_sao_reconhecidas(resultado_simulado):
    """Na base de investimentos a VIA COSTEIRA aparece como "VIACOSTEIRA"."""
    q = resultado_simulado["qualidade"]
    vc = q[(q["concessionaria"] == "VIA COSTEIRA") & (q["ano"] == 2023)].iloc[0]
    assert vc["status"] == "consistente" and vc["investimento_rs"] > 0


def test_nome_sem_correspondencia_e_informado(resultado_simulado):
    assert "CONCESSIONÁRIA ANTIGA ENCERRADA" in resultado_simulado["sem_par_investimentos"]


def test_ano_ainda_nao_publicado_nao_e_julgado(resultado_simulado):
    q = resultado_simulado["qualidade"]
    corrente = q[q["ano"] == dados_simulados.ANO_CORRENTE]
    assert (corrente["status"] == "investimento_nao_publicado").all()
    assert not corrente["excluido_das_metricas"].any()


def test_execucao_sem_investimento_e_mantida(resultado_simulado):
    q = resultado_simulado["qualidade"].set_index(["concessionaria", "ano"])
    linha = q.loc[("RIOSP", 2021)]
    assert linha["status"] == "execucao_sem_investimento"
    assert not linha["excluido_das_metricas"]


def test_apelidos_conferidos():
    assert qualidade.chave_nome("ECOVIAS RIOMINAS") == qualidade.chave_nome("ECOVIAS RIO MINAS")
    assert qualidade.chave_nome("ECOVIASPONTE") == qualidade.chave_nome("ECOVIAS PONTE")
    assert qualidade.chave_nome("Ecovias Minas Goiás") == qualidade.chave_nome("ECOVIAS MINAS GOIAS")
    inv = pd.DataFrame({"concessionaria": ["EPR LITORAL PIONEIRO", "VIA DOS CRISTAIS", "ROTA VERDE", "PR VIAS",
                                           "ECOVIAS 101", "MSVIA", "CRO", "CONCESSIONÁRIA DESCONHECIDA"],
                        "ano": [2025] * 8, "investimento_rs": [float(i) for i in range(1, 9)]})
    ligado, sem_par = qualidade.investimentos_por_nome(
        inv, ["LITORAL PIONEIRO", "VIA CRISTAIS", "ROTA VERDE GOIÁS", "MOTIVA PARANÁ",
              "ECOVIAS CAPIXABA", "PANTANAL", "NOVA ROTA DO OESTE"])
    assert set(ligado["concessionaria"]) == {"LITORAL PIONEIRO", "VIA CRISTAIS", "ROTA VERDE GOIÁS",
                                             "MOTIVA PARANÁ", "ECOVIAS CAPIXABA", "PANTANAL",
                                             "NOVA ROTA DO OESTE"}
    assert sem_par == ["CONCESSIONÁRIA DESCONHECIDA"]   # nome não conferido não é ligado por palpite


def test_investimento_sem_acompanhamento():
    """Como a ECOVIAS RIO MINAS em 2025: plano e investimento, nenhum acompanhamento."""
    acomp = pd.DataFrame({"concessionaria": ["X"], "ano": [2024], "id_sigicor": [1.0],
                          "previsto_anual": [50.0], "executado_anual": [40.0]})
    plan = pd.DataFrame({"concessionaria": ["X"] * 3, "ano": [2024, 2025, 2025],
                         "id_sigicor": [1.0, 2.0, 3.0]})
    inv = pd.DataFrame({"concessionaria": ["X", "X"], "ano": [2024, 2025],
                        "investimento_rs": [1e8, 1.49e9]})
    q, _ = qualidade.checar(acomp, plan, inv, hoje="2026-09-26")
    l25 = q[q["ano"] == 2025].iloc[0]
    assert l25["status"] == "investimento_sem_acompanhamento"
    assert "R$ 1,49 bi" in l25["explicacao"] and "2 obras planejadas" in l25["explicacao"]
    assert q[q["ano"] == 2024].iloc[0]["status"] == "consistente"


def test_painel_recebe_a_checagem(resultado_simulado):
    d = json.loads((resultado_simulado["_saida"] / "docs" / "dados.json").read_text(encoding="utf-8"))
    assert d["qualidade"] and {"status", "explicacao", "excluido_das_metricas"} <= set(d["qualidade"][0])
    assert any(r["excluido_das_metricas"] for r in d["qualidade"])
