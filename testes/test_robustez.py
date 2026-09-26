"""Situações difíceis: bases faltando, pouco histórico, portal fora do ar."""
import pytest
import requests

import dados_simulados
from analise import baixar, config
from analise.__main__ import executar


def test_funciona_so_com_acompanhamento(tmp_path):
    dados_simulados.gerar(tmp_path / "dados", incluir_opcionais=False)
    r = executar(offline=True, pasta_dados=tmp_path / "dados", pasta_saida=tmp_path / "saida")
    assert not r["risco"].empty
    assert (tmp_path / "saida" / "RELATORIO.md").exists()


def test_pouco_historico_usa_regra_simples(tmp_path):
    dados_simulados.gerar(tmp_path / "dados", obras_por_ano=2, anos=[2025, 2026])
    r = executar(offline=True, pasta_dados=tmp_path / "dados", pasta_saida=tmp_path / "saida")
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
