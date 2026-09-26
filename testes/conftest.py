import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "testes"))

import dados_simulados  # noqa: E402

from analise.__main__ import executar  # noqa: E402


@pytest.fixture(scope="session")
def resultado_simulado(tmp_path_factory):
    """Roda a análise completa uma vez sobre as bases simuladas."""
    base = tmp_path_factory.mktemp("sim")
    dados_simulados.gerar(base / "dados")
    saida = base / "saida"
    r = executar(offline=True, pasta_dados=base / "dados", pasta_saida=saida, hoje=dados_simulados.HOJE)
    r["_saida"] = saida
    return r
