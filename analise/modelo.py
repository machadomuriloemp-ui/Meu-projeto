"""Modelo de probabilidade de atraso por obra e agregação por concessionária."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .motivos import CATEGORIAS_TEXTO, NOMES, ROTULOS, rotulo_risco

VARIAVEIS = [
    "hist_atraso_concessionaria", "hist_atraso_tipo", "razao_ytd",
    "previsto_anual", "exec_acum_anterior", "meses_com_meta", "concentracao_fim_ano",
    "duracao_meses", "meses_postergacao", "extensao", "log_valor",
    "n_versoes", "corte_plano_pp", "herdada_ano_anterior",
    "pend_projeto", "pend_licenciamento", "pend_desapropriacao",
] + [f"motivo_{c}" for c in CATEGORIAS_TEXTO]

MIN_OBRAS_MODELO = 20
MIN_CLASSE = 5
CORRELACAO_OBRAS = 0.3  # obras da mesma concessionária tendem a atrasar juntas
SIMULACOES = 5000


def _variaveis_uteis(df: pd.DataFrame) -> list[str]:
    """Descarta variáveis ausentes ou constantes (não ajudam a prever)."""
    uteis = []
    for v in VARIAVEIS:
        if v in df and df[v].notna().any() and df[v].nunique(dropna=True) > 1:
            uteis.append(v)
    return uteis


def _logistica():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))


def _arvores():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                        max_iter=200, l2_regularization=1.0,
                                                        random_state=config.SEMENTE))


class ModeloAtraso:
    def __init__(self):
        self.tipo = "heuristico"
        self.variaveis: list[str] = []
        self.modelo = None
        self.explicador = None
        self.metricas: dict = {}
        self.backtest = pd.DataFrame()   # previsões feitas "às cegas" no último ano encerrado

    # ------------------------------------------------------------------ treino
    def treinar(self, obras: pd.DataFrame) -> "ModeloAtraso":
        treino = obras.dropna(subset=["atrasou"])
        y = treino["atrasou"].astype(int)
        self.variaveis = _variaveis_uteis(treino)
        self.metricas = {"obras_treino": len(treino),
                         "taxa_atraso_historica_pct": float(y.mean() * 100) if len(y) else np.nan}
        menor_classe = int(min(y.sum(), len(y) - y.sum())) if len(y) else 0
        if len(treino) < MIN_OBRAS_MODELO or menor_classe < MIN_CLASSE or not self.variaveis:
            self.metricas["observacao"] = ("Histórico insuficiente para treinar modelo estatístico; "
                                           "usando regra baseada em histórico e execução no ano.")
            return self

        X = treino[self.variaveis].astype(float)
        cv = StratifiedKFold(n_splits=min(5, menor_classe), shuffle=True, random_state=config.SEMENTE)
        candidatos = {"Regressão logística": _logistica(), "Gradient boosting": _arvores()}
        resultados = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for nome, m in candidatos.items():
                p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
                resultados[nome] = {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)}
        melhor = max(resultados, key=lambda k: resultados[k]["auc"])
        self.tipo = melhor
        self.modelo = candidatos[melhor].fit(X, y)
        self.explicador = _logistica().fit(X, y)
        self.metricas.update({
            "modelo_escolhido": melhor,
            "auc_validacao_cruzada": resultados[melhor]["auc"],
            "brier_validacao_cruzada": resultados[melhor]["brier"],
            "comparacao": {k: round(v["auc"], 3) for k, v in resultados.items()},
        })
        self.metricas.update(self._validacao_temporal(treino, melhor))
        return self

    def _validacao_temporal(self, treino: pd.DataFrame, nome: str) -> dict:
        """Treina com anos antigos e testa no último ano encerrado (simula o uso real)."""
        anos = sorted(treino["ano"].unique())
        if len(anos) < 2:
            return {}
        antes, teste = treino[treino["ano"] < anos[-1]], treino[treino["ano"] == anos[-1]]
        if antes["atrasou"].nunique() < 2 or teste["atrasou"].nunique() < 2 or len(antes) < 10:
            return {}
        m = _logistica() if nome == "Regressão logística" else _arvores()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(antes[self.variaveis].astype(float), antes["atrasou"].astype(int))
            p = m.predict_proba(teste[self.variaveis].astype(float))[:, 1]
        self.backtest = teste.assign(prob_atraso=p)
        return {"auc_teste_ultimo_ano": roc_auc_score(teste["atrasou"].astype(int), p),
                "ano_teste": int(anos[-1])}

    # --------------------------------------------------------------- previsão
    def prever(self, obras: pd.DataFrame) -> pd.DataFrame:
        alvo = obras[(obras["situacao"] != "encerrado") & (obras["previsto_anual"].fillna(0) > 0)].copy()
        if alvo.empty:
            return alvo
        if self.modelo is not None:
            X = alvo[self.variaveis].astype(float)
            alvo["prob_atraso"] = self.modelo.predict_proba(X)[:, 1]
            alvo["motivos_provaveis"] = self._explicar(X)
        else:
            hist = alvo["hist_atraso_concessionaria"].fillna(0.5)
            execucao = 1 - alvo["razao_ytd"].fillna(1).clip(0, 1)
            peso_exec = np.where(alvo["previsto_ytd"] > 0, 0.6, 0.0)
            alvo["prob_atraso"] = (1 - peso_exec) * hist + peso_exec * execucao
            alvo["motivos_provaveis"] = alvo.apply(_motivos_regra, axis=1)
        alvo["em_risco"] = alvo["prob_atraso"] >= config.LIMIAR_RISCO
        return alvo

    def _explicar(self, X: pd.DataFrame) -> list[str]:
        imputador, escala, reg = self.explicador.named_steps.values()
        z = escala.transform(imputador.transform(X))
        contrib = z * reg.coef_[0]
        textos = []
        for linha in contrib:
            ordem = np.argsort(-linha)
            top = []
            for i in ordem:
                if linha[i] <= 0.1 or len(top) == 3:
                    break
                frase = rotulo_risco(self.variaveis[i], reg.coef_[0][i])
                if frase:
                    top.append(frase)
            textos.append("; ".join(top) if top else "Sem fator de risco destacado")
        return textos

    def importancia(self) -> pd.DataFrame:
        """Peso de cada variável (coeficiente padronizado da regressão logística)."""
        if self.explicador is None:
            return pd.DataFrame()
        coef = self.explicador.named_steps["logisticregression"].coef_[0]
        df = pd.DataFrame({"variavel": self.variaveis,
                           "fator": [NOMES.get(v, v) for v in self.variaveis],
                           "peso": coef})
        df["efeito"] = np.where(df["peso"] > 0, "quanto maior, MAIS risco",
                                "quanto maior, MENOS risco")
        return df.reindex(df["peso"].abs().sort_values(ascending=False).index).reset_index(drop=True)


FAIXAS_OBRA = [(0.66, "Alto"), (0.33, "Médio"), (0.0, "Baixo")]


def _faixa_obra(p: float) -> str:
    for limite, nome in FAIXAS_OBRA:
        if p >= limite:
            return nome
    return "Baixo"


def resumo_backtest(bt: pd.DataFrame) -> dict:
    """Confere as previsões feitas sem conhecer o resultado do ano de teste.

    O modelo foi treinado só com anos anteriores e previu o ano de teste a partir
    do que se sabia no meio dele. Aqui comparamos com o que realmente aconteceu.
    """
    if bt is None or bt.empty:
        return {}
    bt = bt.copy()
    bt["faixa"] = bt["prob_atraso"].map(_faixa_obra)
    bt["previu_atraso"] = bt["prob_atraso"] >= config.LIMIAR_RISCO
    faixas = []
    for nome in ["Alto", "Médio", "Baixo"]:
        g = bt[bt["faixa"] == nome]
        if len(g):
            faixas.append({"faixa": nome, "obras": int(len(g)),
                           "atrasaram": int(g["atrasou"].sum()),
                           "pct_atrasaram": float(g["atrasou"].mean() * 100),
                           "prob_media_prevista": float(g["prob_atraso"].mean() * 100)})
    conc = (bt.groupby("concessionaria")
            .agg(obras=("atrasou", "size"), previsto_pct=("prob_atraso", "mean"),
                 real_pct=("atrasou", "mean"))
            .reset_index())
    conc = conc[conc["obras"] >= 3]
    conc[["previsto_pct", "real_pct"]] *= 100
    conc["erro_pp"] = (conc["previsto_pct"] - conc["real_pct"]).abs()
    faixas_conc = {}
    for c, g in bt.groupby("concessionaria"):
        faixas_conc[c] = [{"faixa": nome, "obras": int(len(h)), "atrasaram": int(h["atrasou"].sum()),
                           "pct_atrasaram": float(h["atrasou"].mean() * 100),
                           "prob_media_prevista": float(h["prob_atraso"].mean() * 100)}
                          for nome in ["Alto", "Médio", "Baixo"]
                          for h in [g[g["faixa"] == nome]] if len(h)]
    acertos = (bt["previu_atraso"] == (bt["atrasou"] == 1))
    return {
        "ano": int(bt["ano"].iloc[0]),
        "obras": int(len(bt)),
        "acerto_pct": float(acertos.mean() * 100),
        "atrasos_reais": int(bt["atrasou"].sum()),
        "atrasos_detectados": int((bt["previu_atraso"] & (bt["atrasou"] == 1)).sum()),
        "faixas": faixas,
        "concessionarias": conc.sort_values("real_pct", ascending=False).to_dict("records"),
        "faixas_por_concessionaria": faixas_conc,
    }


def _motivos_regra(linha: pd.Series) -> str:
    motivos = []
    if linha.get("previsto_ytd", 0) > 0 and linha.get("razao_ytd", 1) < config.TOLERANCIA_CUMPRIMENTO:
        motivos.append("Baixa execução no ano até agora")
    for f in ["pend_licenciamento", "pend_desapropriacao", "pend_projeto", "herdada_ano_anterior"]:
        if linha.get(f, 0) == 1:
            motivos.append(ROTULOS[f])
    if linha.get("hist_atraso_concessionaria", 0) > 0.5:
        motivos.append(ROTULOS["hist_atraso_concessionaria"])
    return "; ".join(motivos[:3]) or "Sem fator de risco destacado"


def _faixa(p: float) -> str:
    for limite, nome in config.FAIXAS_RISCO:
        if p >= limite:
            return nome
    return config.FAIXAS_RISCO[-1][1]


def ano_referencia(anos: pd.Series, hoje: pd.Timestamp | None = None) -> int:
    """Ano a avaliar: o ano atual, se houver plano para ele; senão o mais recente até hoje."""
    ano_atual = (pd.Timestamp.today() if hoje is None else pd.Timestamp(hoje)).year
    anos = sorted(int(a) for a in anos.dropna().unique())
    if ano_atual in anos:
        return ano_atual
    passados = [a for a in anos if a <= ano_atual]
    return passados[-1] if passados else anos[0]


def risco_concessionarias(previsoes: pd.DataFrame, obras: pd.DataFrame,
                          hoje: pd.Timestamp | None = None) -> pd.DataFrame:
    """Probabilidade de cada concessionária NÃO cumprir o plano do ano.

    Simulação de Monte Carlo: cada obra atrasa com sua probabilidade prevista;
    atrasos são correlacionados dentro da concessionária (choque comum). Quando
    atrasa, a obra entrega uma fração do previsto sorteada do histórico real.
    """
    if previsoes.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(config.SEMENTE)
    hist = obras[obras["atrasou"] == 1].copy()
    hist["fracao"] = (hist["executado_anual"].fillna(0) / hist["previsto_anual"]).clip(0, 1)
    fracao_geral = hist["fracao"].to_numpy() if len(hist) else np.array([0.5])

    linhas = []
    for conc, g in previsoes.groupby("concessionaria"):
        ano = ano_referencia(g["ano"], hoje)
        g = g[g["ano"] == ano]
        p = g["prob_atraso"].clip(1e-4, 1 - 1e-4).to_numpy()
        prev = g["previsto_anual"].to_numpy()
        frac_conc = hist.loc[hist["concessionaria"] == conc, "fracao"].to_numpy()
        fracoes = frac_conc if len(frac_conc) >= 5 else fracao_geral

        comum = rng.standard_normal((SIMULACOES, 1))
        idio = rng.standard_normal((SIMULACOES, len(p)))
        z = np.sqrt(CORRELACAO_OBRAS) * comum + np.sqrt(1 - CORRELACAO_OBRAS) * idio
        atrasa = norm.cdf(z) < p
        entregue = np.where(atrasa, rng.choice(fracoes, size=atrasa.shape), 1.0)
        cumprimento = (entregue * prev).sum(axis=1) / prev.sum()
        prob_nao_cumprir = float((cumprimento < config.TOLERANCIA_CUMPRIMENTO).mean())

        valor = g["valor_contratual"].fillna(0).to_numpy() if "valor_contratual" in g else np.zeros(len(g))
        ext = g["extensao"].fillna(0).to_numpy()
        restante = np.clip(prev - g["executado_ytd"].fillna(0).to_numpy(), 0, None) / 100
        linhas.append({
            "concessionaria": conc, "ano": int(ano), "situacao": g["situacao"].iloc[0],
            "obras_no_plano": len(g),
            "prob_nao_cumprir_plano_pct": prob_nao_cumprir * 100,
            "risco": _faixa(prob_nao_cumprir),
            "cumprimento_esperado_pct": float(cumprimento.mean() * 100),
            "cumprimento_p10_p90": f"{np.percentile(cumprimento, 10) * 100:.0f}%–"
                                   f"{np.percentile(cumprimento, 90) * 100:.0f}%",
            "prob_media_atraso_obra_pct": float(p.mean() * 100),
            "obras_em_risco": int(g["em_risco"].sum()),
            "valor_em_risco_rs": float((p * restante * valor).sum()),
            # O que deve escorregar para o ano seguinte (valor esperado)
            "obras_para_proximo_ano": float(p.sum()),
            "pp_para_proximo_ano": float((p * prev * (1 - fracoes.mean())).sum()),
            "km_para_proximo_ano": float((p * prev / 100 * (1 - fracoes.mean()) * ext).sum()),
            "km_em_risco": float((p * prev / 100 * ext).sum()),
        })
    return (pd.DataFrame(linhas)
            .sort_values(["prob_nao_cumprir_plano_pct", "cumprimento_esperado_pct"],
                         ascending=[False, True])
            .reset_index(drop=True))
