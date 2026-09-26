"""Recorte por estado e região (obra a obra)."""
import json

import pandas as pd

from analise import regiao


def test_uf_pela_rodovia():
    casos = {"BR-381/MG": "MG", "BR-116/RJ": "RJ", "br-101 / sc": "SC", "PR-423": "PR", "PR-PR-323": "PR",
             "BR-364": None, "ACESSO PORTO NOVO": None, "ACESSO 2": None, None: None, "": None}
    for entrada, esperado in casos.items():
        assert regiao.uf_pela_rodovia(entrada) == esperado, entrada


def test_uf_pelo_codigo_snv():
    assert regiao.uf_pelo_snv("101BSC4125") == "SC"
    assert regiao.uf_pelo_snv("163BMT0450") == "MT"
    assert regiao.uf_pelo_snv("20180A") is None
    assert regiao.uf_pelo_snv(None) is None


def test_ordem_de_identificacao_e_fallback():
    obras = pd.DataFrame({
        "concessionaria": ["NOVA 364", "NOVA 364", "MULTI", "MULTI", "MULTI", "NOVA ROTA"],
        "rodovia": ["BR-364/RO", "BR-364", "BR-381/MG", "BR-381/SP", "BR-381", "BR-163"],
        "snv_cadastro": [None, None, None, None, None, "163BMT0450"],
    })
    o = regiao.atribuir_uf(obras)
    assert [u if isinstance(u, str) else None for u in o["uf"]] == ["RO", "RO", "MG", "SP", None, "MT"]
    assert o.loc[1, "uf_origem"] == "concessionária de um só estado"
    assert o.loc[4, "uf_origem"] == "não identificado"   # atua em 2 estados: não chuta
    assert o.loc[5, "uf_origem"] == "código SNV"
    assert o.loc[0, "regiao"] == "Norte" and o.loc[3, "regiao"] == "Sudeste"


def test_concessao_em_dois_estados_conta_obra_a_obra():
    obras = pd.DataFrame({"concessionaria": ["X"] * 4, "rodovia": ["BR-116/PR", "BR-116/PR", "BR-116/SC", "BR-116/SC"],
                          "atrasou": [1.0, 1.0, 0.0, 1.0], "ano": [2025] * 4})
    r = regiao.resumo(obras, pd.DataFrame(), pd.DataFrame())
    uf = r["por_uf"].set_index("uf")
    assert uf.loc["PR", "taxa_atraso_pct"] == 100 and uf.loc["SC", "taxa_atraso_pct"] == 50
    assert r["por_regiao"].set_index("regiao").loc["Sul", "metas_encerradas"] == 4


def test_totais_batem_com_o_historico(resultado_simulado):
    rg = resultado_simulado["regioes"]
    hist = resultado_simulado["obras"].dropna(subset=["atrasou"])
    assert rg["por_uf"]["metas_encerradas"].sum() + rg["sem_uf_hist"] == len(hist)
    assert rg["por_regiao"]["metas_encerradas"].sum() == rg["por_uf"]["metas_encerradas"].sum()
    assert set(rg["por_uf"]["uf"]) == {"RS", "SC", "RJ", "MG"}
    rs = rg["por_uf"].set_index("uf").loc["RS"]
    sc = rg["por_uf"].set_index("uf").loc["SC"]
    assert rs["taxa_atraso_pct"] > sc["taxa_atraso_pct"]   # VIA SUL (RS) atrasa mais que VIA COSTEIRA (SC)


def test_painel_recebe_regioes(resultado_simulado):
    d = json.loads((resultado_simulado["_saida"] / "docs" / "dados.json").read_text(encoding="utf-8"))
    rg = d["regioes"]
    assert {"por_uf", "por_regiao", "por_concessionaria"} <= set(rg)
    assert rg["por_concessionaria"]["VIA SUL"]["por_uf"][0]["uf"] == "RS"
    assert "NaN" not in json.dumps(rg)
