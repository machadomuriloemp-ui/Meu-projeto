"""Situações difíceis: bases faltando, pouco histórico, portal fora do ar."""
import pytest
import requests

import dados_simulados
from analise import baixar, config
from analise.__main__ import executar


def test_funciona_so_com_acompanhamento(tmp_path):
    dados_simulados.gerar(tmp_path / "dados", incluir_opcionais=False)
    r = executar(offline=True, pasta_dados=tmp_path / "dados", pasta_saida=tmp_path / "saida",
                 hoje=dados_simulados.HOJE)
    assert not r["risco"].empty
    assert (tmp_path / "saida" / "RELATORIO.md").exists()


def test_pouco_historico_usa_regra_simples(tmp_path):
    dados_simulados.gerar(tmp_path / "dados", obras_por_ano=2, anos=[2025, 2026])
    r = executar(offline=True, pasta_dados=tmp_path / "dados", pasta_saida=tmp_path / "saida",
                 hoje=dados_simulados.HOJE)
    assert "observacao" in r["metricas"]            # não treinou modelo estatístico
    assert r["previsoes"]["prob_atraso"].between(0, 1).all()


def test_portal_fora_do_ar_usa_arquivos_locais(tmp_path, monkeypatch):
    dados_simulados.gerar(tmp_path / "dados")
    monkeypatch.setattr(config, "PASTA_DADOS", tmp_path / "dados")
    monkeypatch.setattr(baixar.time, "sleep", lambda s: None)

    def falha(*a, **k):
        raise requests.ConnectionError("sem rede")

    monkeypatch.setattr(baixar.requests, "get", falha)
    arquivos = baixar.baixar_tudo(offline=False)
    assert len(arquivos["acompanhamento"]) == 1
    assert len(arquivos["inexecucao"]) == 1


def test_erro_claro_sem_base_obrigatoria(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_DADOS", tmp_path / "vazio")
    with pytest.raises(SystemExit, match="acompanhamento"):
        baixar.baixar_tudo(offline=True)


def test_descoberta_de_arquivos_pela_api(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"resources": [
                {"format": "CSV", "url": "https://x/a.csv"},
                {"format": "PDF", "url": "https://x/dic.pdf"},
                {"format": "", "url": "https://x/b.csv"}]}}

    monkeypatch.setattr(baixar.requests, "get", lambda *a, **k: Resp())
    assert baixar._descobrir_csvs("inexecucao") == ["https://x/a.csv", "https://x/b.csv"]


def test_ano_encerrado_com_meses_zerados_conta_como_nao_execucao():
    """Um ano já terminado (relatório enviado no ano seguinte) com execução só até
    o mês 8 NÃO pode ser tratado como "em andamento" — isso esconderia o atraso."""
    import pandas as pd
    from analise.cumprimento import meses_fechados

    def linha(conc, ano, relatorio, ultimo_mes):
        d = {"concessionaria": conc, "ano": ano, "data_planejamento": pd.Timestamp(relatorio)}
        for m in range(1, 13):
            d[f"exec_{m}"] = 5.0 if m <= ultimo_mes else 0.0
        return d

    acomp = pd.DataFrame([
        linha("A", 2025, "2026-06-25", 8),   # relatório de 2025 enviado em 2026 -> encerrado
        linha("B", 2026, "2026-07-10", 6),   # ano corrente -> fechado até o mês 6
        linha("C", 2024, "2024-11-30", 9),   # ano muito antigo -> encerrado pelo calendário
    ])
    r = meses_fechados(acomp, hoje="2026-09-26").set_index("concessionaria")["meses_fechados"]
    assert r["A"] == 12
    assert r["B"] == 6
    assert r["C"] == 12


def test_tendencia_nao_e_distorcida_por_mes_extremo():
    import numpy as np
    import pandas as pd
    from analise.cumprimento import tendencia

    t = np.arange(24)
    indice = np.full(24, 80.0)
    indice[5] = 900.0      # um mês com execução 9x o previsto
    serie = pd.DataFrame({"concessionaria": "X", "t": t, "indice_cumprimento": indice})
    r = tendencia(serie).iloc[0]
    assert r["tendencia"] == "Estável"
    assert abs(r["inclinacao_pp_ano"]) < 1
