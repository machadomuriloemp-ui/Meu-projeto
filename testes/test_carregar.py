"""Testes das conversões de formato (números, %, datas, codificação, colunas)."""
import numpy as np
import pandas as pd

import dados_simulados
from analise.carregar import (ler_csv, normalizar_nome, padronizar, para_data, para_numero,
                              para_percentual)


def test_numeros_no_formato_brasileiro():
    s = pd.Series(["1.234,56", "16.44", " 337,700", "R$ 12.345.678,90", "", "abc", "340,2"])
    r = para_numero(s)
    assert r.iloc[0] == 1234.56
    assert r.iloc[1] == 16.44
    assert r.iloc[2] == 337.7
    assert r.iloc[3] == 12345678.90
    assert np.isnan(r.iloc[4]) and np.isnan(r.iloc[5])
    assert r.iloc[6] == 340.2


def test_percentuais():
    assert list(para_percentual(pd.Series(["55,7%", "0,0%", "100%"]))) == [55.7, 0.0, 100.0]
    # Sem "%" e com valores entre 0 e 1: interpretado como fração
    assert list(para_percentual(pd.Series(["0,5", "0,25"]))) == [50.0, 25.0]


def test_datas():
    r = para_data(pd.Series(["2026-06-25-03:00", "01/04/2022", "14/02/2025", ""]))
    assert r.iloc[0] == pd.Timestamp("2026-06-25")
    assert r.iloc[1] == pd.Timestamp("2022-04-01")  # dia primeiro, não mês
    assert r.iloc[2] == pd.Timestamp("2025-02-14")
    assert pd.isna(r.iloc[3])


def test_nomes_de_colunas():
    assert normalizar_nome("Ano do Planejamento") == "ano_do_planejamento"
    assert normalizar_nome("Extensão(km)") == "extensao_km"
    assert normalizar_nome("% Executado Acumulado") == "executado_acumulado"


def test_arquivo_windows_1252(tmp_path):
    arq = tmp_path / "inex.csv"
    arq.write_bytes("concessionaria;descricao\nWAY 262;Praça de pedágio 1 - Reforma\n".encode("cp1252"))
    df = ler_csv(arq)
    assert df.loc[0, "descricao"] == "Praça de pedágio 1 - Reforma"


def test_acentos_quebrados_sao_corrigidos(tmp_path):
    arq = tmp_path / "x.csv"
    arq.write_text("concessionaria;tipo\nVIA SUL;InterseÃ§Ã£o em DesnÃ­vel\n", encoding="utf-8")
    assert ler_csv(arq).loc[0, "tipo"] == "Interseção em Desnível"


def test_linha_mal_formatada_nao_derruba_leitura(tmp_path):
    arq = tmp_path / "x.csv"
    arq.write_text("a;b;c\n1;2;3\n4;5;6;7\n8;9;10\n", encoding="utf-8")
    df = ler_csv(arq)
    assert len(df) == 2


def test_cabecalhos_reais_da_antt_sao_reconhecidos(tmp_path):
    """Os cabeçalhos copiados dos arquivos publicados viram os nomes padronizados."""
    esperado = {
        "CAB_ACOMP": ["concessionaria", "id_sigicor", "ano", "versao", "data_planejamento",
                      "prev_1", "exec_1", "prev_12", "exec_12", "previsto_anual",
                      "executado_anual", "exec_acum_anterior", "data_fim_prevista"],
        "CAB_PLAN": ["ano", "projeto_executivo", "licenciamento_ambiental", "desapropriacao",
                     "observacao", "prev_6", "previsto_anual", "tipo_planejamento",
                     "exec_acum_anterior"],
        "CAB_CAD": ["valor_contratual", "data_fim_prevista", "extensao", "id_sigicor"],
        "CAB_INEX": ["inexecucao", "fator_d", "executado_total", "data_fim_prevista", "ano"],
    }
    for nome, colunas in esperado.items():
        cab = getattr(dados_simulados, nome)
        arq = tmp_path / f"{nome}.csv"
        n = cab.count(";") + 1
        arq.write_text(cab + "\n" + ";".join(["1"] * n) + "\n", encoding="utf-8")
        df = padronizar(ler_csv(arq))
        faltando = [c for c in colunas if c not in df.columns]
        assert not faltando, f"{nome}: colunas não reconhecidas {faltando}"
