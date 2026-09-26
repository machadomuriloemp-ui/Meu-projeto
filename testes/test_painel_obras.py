"""Lista de obras do painel: sem repetições e sem anos que já passaram."""
import collections
import json

import pandas as pd

from analise.dashboard import obras_para_o_painel


def _dados(resultado):
    return json.loads((resultado["_saida"] / "docs" / "dados.json").read_text(encoding="utf-8"))


def test_itens_identicos_viram_um_card_so(resultado_simulado):
    d = _dados(resultado_simulado)
    chaves = collections.Counter((o["concessionaria"], o["ano"], o["descricao"], o["rodovia"], o["km_inicial"])
                                 for o in d["obras"])
    repetidas = [k for k, n in chaves.items() if n > 1]
    assert not repetidas, f"cards repetidos: {repetidas[:3]}"
    # O caso simulado (vários acessos iguais da RIOSP) aparece agrupado, com a contagem
    acessos = [o for o in d["obras"] if o["descricao"] == "L. Acessos - SEGMENTO 1 - RJ"]
    assert acessos and all(o["itens"] > 1 for o in acessos)


def test_nenhum_item_se_perde_no_agrupamento(resultado_simulado):
    d = _dados(resultado_simulado)
    prev = resultado_simulado["previsoes"]
    for r in d["risco"]:
        itens = sum(o["itens"] for o in d["obras"]
                    if o["concessionaria"] == r["concessionaria"] and o["ano"] == r["ano"])
        no_plano = len(prev[(prev["concessionaria"] == r["concessionaria"]) & (prev["ano"] == r["ano"])])
        assert itens == no_plano == r["obras_no_plano"], r["concessionaria"]


def test_so_anos_avaliados_aparecem(resultado_simulado):
    """Cada concessionária mostra apenas o ano do ranking e, se houver, o ano seguinte."""
    d = _dados(resultado_simulado)
    validos = {(r["concessionaria"], r["ano"]) for r in d["risco"] + d["risco_proximo"]}
    for o in d["obras"]:
        assert (o["concessionaria"], o["ano"]) in validos, (o["concessionaria"], o["ano"])


def test_ano_passado_sem_acompanhamento_nao_vira_previsao():
    """Como na ECOVIAS RIO MINAS: plano de 2025 (já passou) e de 2026, sem acompanhamento.
    Só 2026 deve aparecer."""
    prev = pd.DataFrame({
        "concessionaria": ["X"] * 4, "ano": [2025, 2025, 2026, 2026],
        "situacao": ["planejado"] * 4, "descricao": ["A", "B", "A", "C"], "tipo": ["t"] * 4,
        "rodovia": ["BR-1"] * 4, "km_inicial": [1.0, 2.0, 1.0, 3.0], "prob_atraso": [0.9, 0.8, 0.7, 0.2],
        "motivos_provaveis": ["m"] * 4,
    })
    validos = pd.DataFrame({"concessionaria": ["X"], "ano": [2026]})
    p = obras_para_o_painel(prev, validos)
    assert set(p["ano"]) == {2026}
    assert len(p) == 2


def test_agrupamento_guarda_a_maior_chance_e_a_media():
    prev = pd.DataFrame({
        "concessionaria": ["X"] * 3, "ano": [2026] * 3, "situacao": ["em andamento"] * 3,
        "descricao": ["Acesso"] * 3, "tipo": ["t"] * 3, "rodovia": ["BR-1"] * 3,
        "km_inicial": [5.0, 5.0, 5.0], "prob_atraso": [0.9, 0.5, 0.1], "motivos_provaveis": ["a", "b", "c"],
    })
    p = obras_para_o_painel(prev)
    assert len(p) == 1
    assert p.loc[0, "itens"] == 3
    assert p.loc[0, "prob"] == 90.0
    assert p.loc[0, "prob_media"] == 50.0
    assert p.loc[0, "motivos_provaveis"] == "a"
