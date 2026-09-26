"""Configurações gerais: fontes de dados, pastas e parâmetros da análise."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "dados" / "brutos"
PASTA_SAIDA = RAIZ / "saida"
PASTA_DASHBOARD = RAIZ / "docs"  # publicada pelo GitHub Pages

PORTAL = "https://dados.antt.gov.br"

# Conjuntos de dados do Portal de Dados Abertos da ANTT (sistema SIGICOR).
# "slug" é usado para descobrir automaticamente todos os arquivos CSV do conjunto
# pela API do portal (CKAN). As URLs fixas são usadas se a API não responder.
FONTES = {
    "acompanhamento": {
        "slug": "acompanhamento-anual",
        "obrigatoria": True,
        "urls": [
            f"{PORTAL}/dataset/b4ef619c-71a1-48a5-86c8-1109964753c5/resource/"
            "ac0da108-a79b-4bd9-b770-6c36c0778804/download/acompanhamento-anual-investimentos.csv",
        ],
    },
    "planejamento": {
        "slug": "planejamento-anual",
        "obrigatoria": False,
        "urls": [
            f"{PORTAL}/dataset/1dda2275-1777-46eb-8dcc-013497656a93/resource/"
            "5493f281-08ed-4bd8-a0d0-1486af3a3487/download/"
            "planejamento-anual-de-investimentos-dos-contratos-de-concessao.csv",
        ],
    },
    "cadastro": {
        "slug": "cadastros-investimentos",
        "obrigatoria": False,
        "urls": [
            f"{PORTAL}/dataset/6a42582f-5481-458d-bf21-247d91a60af5/resource/"
            "5ae179e0-e7a2-4a65-9a9f-40b987f66701/download/"
            "cadastros-de-investimentos-dos-contratos-de-concessao.csv",
        ],
    },
    "investimentos": {
        # Valor investido por concessionária e ano (R$), sem detalhe por obra.
        # Usado para checar se a execução física foi de fato informada.
        "slug": "investimentos",
        "obrigatoria": False,
        "urls": [
            f"{PORTAL}/dataset/a133da64-1e03-4832-909d-e1eb835eec2e/resource/"
            "a08292ac-41a6-4220-bfe2-35c7b3abfbf0/download/investimentos.csv",
        ],
    },
    "inexecucao": {
        "slug": "inexecucao",
        "obrigatoria": False,
        "urls": [
            f"{PORTAL}/dataset/4e09e6b8-dcdf-4bc2-bf91-0a46e04920b5/resource/"
            "39f5386d-45d7-44bc-b6cb-c5d210e2a0eb/download/litoral-pioneiro-inexecucao.csv",
            f"{PORTAL}/dataset/4e09e6b8-dcdf-4bc2-bf91-0a46e04920b5/resource/"
            "335dd879-5941-4034-b312-3724f9b820f8/download/way-262-inexecucao.csv",
            f"{PORTAL}/dataset/4e09e6b8-dcdf-4bc2-bf91-0a46e04920b5/resource/"
            "42174316-358c-4681-b980-3c54b2eb5e1f/download/via-cristais-inexecucao.csv",
        ],
    },
}

# Uma obra "atrasou" no ano quando executou menos que esta fração do previsto.
TOLERANCIA_CUMPRIMENTO = 0.90

# Probabilidade a partir da qual a obra é classificada como "em risco".
LIMIAR_RISCO = 0.50

# Faixas de classificação do risco da concessionária (probabilidade de NÃO
# cumprir ao menos TOLERANCIA_CUMPRIMENTO do plano do ano).
FAIXAS_RISCO = [(0.66, "Alto"), (0.33, "Médio"), (0.0, "Baixo")]

# Variação da tendência (pontos percentuais por ano) considerada relevante.
LIMIAR_TENDENCIA_PP_ANO = 5.0

SEMENTE = 42
