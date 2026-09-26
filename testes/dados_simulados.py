"""Gera bases SIMULADAS com o mesmo formato dos CSVs reais da ANTT.

Os cabeçalhos são cópias exatas dos arquivos publicados (ponto e vírgula,
"%" com vírgula decimal, datas dd/mm/aaaa, codificação Windows-1252 na
inexecução). Os dados embutem um "gabarito" conhecido, para que os testes
confiram se a análise o encontra:

* VIA SUL      : atrasa muito e está piorando
* VIA COSTEIRA : atrasa pouco e está melhorando
* RIOSP e WAY 262: intermediárias e estáveis
* Licenciamento ou desapropriação pendentes aumentam fortemente o atraso
* 2026 é o ano corrente, com execução informada só até o mês 6
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CAB_ACOMP = ("concessionaria;id_sigicor;ano_do_planejamento;versao;data_do_planejamento;origem_PER;"
             "item_do_PER;descricao;rodovia;tipo;km_Inicial;km_Final;extensao;data_inicio_Prevista;"
             "data_Fim_Prevista;seguro_de_Risco_de_Engenharia;executado_acumulado_anterior;"
             + ";".join(f"mes_{m}_previsto;mes_{m}_executado" for m in range(1, 13))
             + ";previsto_anual;executado_anual;executado_acumulado_anterior_executado_anual")
CAB_PLAN = ("concessionaria;id_sigicor;Ano_do_planejamento;versao;data_do_planejamento;Origem_PER;"
            "Item_do_PER;Descricao;Rodovia;Tipo;Km_Inicial;Km_Final;extensao;projeto_executivo;"
            "licenciamento_ambiental;desapropriacao;observacao;Data_inicio_Prevista;Data_Fim_Prevista;"
            "ano_inicio;ano_fim;execucao_acumulada;"
            + ";".join(f"mes_{m}_previsto" for m in range(1, 13))
            + ";previsto_anual;executado_acumulado_previsto_anual;investimento_tipo_planejamento")
CAB_CAD = ("concessionaria;id_sigicor;Origem_PER;Nro_Processo_SEI;Item_do_PER;Valor_Contratual;"
           "Data_Base_Valor_Contratual;Descricao;Rodovia;Tipo;Km_Inicial;Km_Final;Extensao;Sentido;"
           "Direcao;Pista;Codigo_SNV_Inicial;Codigo_SNV_Final;Versao_SNV;Latitude_Inicial;"
           "Longitude_Inicial;Latitude_Final;Longitude_Final;Data_inicio_Prevista;Data_Fim_Prevista;"
           "ano_inicio;ano_fim")
CAB_INEX = ("concessionaria;id_sigicor;ano_do_planejamento;versao;data_do_planejamento;origem_PER;"
            "item_do_PER;descricao;rodovia;tipo;km_Inicial;km_Final;extensao;previsao_de_inicio_obra;"
            "previsao_de_Termino_obra;seguro_de_Risco_de_Engenharia;executado_acumulado_anterior;"
            "previsto_anual;executado_acumulado_anterior_previsto_anual;executado_total;inexecucao;fator_D")

CONCESSIONARIAS = {
    # nome: (nível de atraso inicial em logit, variação por ano, rodovia)
    "VIA SUL": (-0.6, 0.65, "BR-386/RS"),
    "VIA COSTEIRA": (0.4, -0.65, "BR-101/SC"),
    "RIOSP": (-1.2, 0.0, "BR-116/RJ"),
    "WAY 262": (-0.7, 0.0, "BR-262/MG"),
}
TIPOS = {"Duplicação": 0.6, "Passarela (local definido)": -0.4, "Interseção em Desnível": 0.3,
         "Base de Serviços Operacionais com Atendimento aos Usuários": -0.6,
         "Adequação de Acostamento": 0.0, "Praça de Pedágio": -0.2}
OBS_ATRASO = ["Aguardando emissão de licença ambiental pelo IBAMA",
              "Processo de desapropriação em andamento (DUP publicada)",
              "Chuvas intensas paralisaram a frente de obra",
              "Interferência com rede de energia, remanejamento pela distribuidora"]
ANOS = list(range(2021, 2027))
ANO_CORRENTE, MES_CORTE = 2026, 6


def _p(v: float) -> str:
    return f"{v:.2f}%".replace(".", ",")


def _km(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def gerar(pasta: Path, obras_por_ano: int = 22, semente: int = 7,
          anos: list[int] | None = None, incluir_opcionais: bool = True) -> Path:
    rng = np.random.default_rng(semente)
    anos = anos or ANOS
    pasta = Path(pasta)
    for sub in ["acompanhamento", "planejamento", "cadastro", "inexecucao"]:
        (pasta / sub).mkdir(parents=True, exist_ok=True)

    acomp, plan, cad, inex = [CAB_ACOMP], [CAB_PLAN], [CAB_CAD], [CAB_INEX]
    id_seq = 1000
    for conc, (base, var, rodovia) in CONCESSIONARIAS.items():
        for ano in anos + [anos[-1] + 1]:  # último ano: só planejamento (futuro)
            futuro = ano > anos[-1]
            for _ in range(obras_por_ano):
                id_seq += 1
                tipo = rng.choice(list(TIPOS))
                km = rng.uniform(1, 500)
                ext = rng.uniform(0.5, 25) if tipo == "Duplicação" else 0.0
                lic = rng.random() < 0.25
                desap = rng.random() < 0.15
                logit = base + var * (ano - anos[0]) + TIPOS[tipo] + 1.8 * lic + 1.3 * desap - 0.3
                atrasou = rng.random() < 1 / (1 + np.exp(-logit))

                prev_anual = rng.uniform(20, 80)
                meses_ativos = sorted(rng.choice(range(1, 13), size=rng.integers(3, 10), replace=False))
                pesos = rng.dirichlet(np.ones(len(meses_ativos)))
                prev_m = {m: 0.0 for m in range(1, 13)}
                for m, w in zip(meses_ativos, pesos):
                    prev_m[m] = prev_anual * w
                frac = rng.uniform(0.05, 0.75) if atrasou else rng.uniform(0.93, 1.05)
                exec_m = {m: max(0.0, prev_m[m] * frac * rng.uniform(0.8, 1.2)) for m in prev_m}
                if ano == ANO_CORRENTE:
                    for m in range(MES_CORTE + 1, 13):
                        exec_m[m] = 0.0
                exec_anual = sum(exec_m.values())
                acum_ant = rng.uniform(0, 100 - prev_anual)
                ini = f"01/{rng.integers(1, 7):02d}/{ano}"
                fim_contrato = f"30/12/{ano}"
                # Término: no início do ano quase sempre igual ao contrato; revisões
                # feitas DEPOIS (quando o atraso já é conhecido) empurram a data.
                aviso_cedo = rng.random() < (0.35 if atrasou else 0.10)
                fim_inicial = f"30/06/{ano + 1}" if aviso_cedo else fim_contrato
                fim_plan = f"30/{rng.integers(6, 13):02d}/{ano + 1}" if atrasou else fim_inicial
                obs = rng.choice(OBS_ATRASO) if (atrasou and rng.random() < 0.5) else ""
                if lic:
                    obs = OBS_ATRASO[0]
                desc = f"{tipo} - km {km:.3f} - obra {id_seq}"

                # Planejamento: 1 a 3 versões; obras que atrasam têm meta cortada
                # Versões: v1 (jan), v2 (mai), v3 (set), v4 (fev do ano seguinte).
                # Obras que atrasam são revisadas mais e têm a meta cortada nas
                # versões tardias — informação que NÃO existe no momento da previsão.
                datas_v = [f"15/01/{ano}", f"15/05/{ano}", f"15/09/{ano}", f"15/02/{ano + 1}"]
                n_versoes = rng.integers(1, 3) if not atrasou else rng.integers(3, 5)
                for v in range(1, n_versoes + 1):
                    tardia = v >= 3
                    corte = 1.3 if (atrasou and not tardia) else 1.0
                    pv = {m: prev_m[m] * corte for m in prev_m}
                    fim_v = fim_plan if tardia else fim_inicial
                    plan.append(";".join([
                        conc, str(id_seq), str(ano), str(v), datas_v[v - 1], "ORIGINAL", "3.2.1",
                        desc, rodovia, tipo, _km(km), "", _km(ext) if ext else "",
                        "Concluído" if rng.random() < 0.8 else "Em análise",
                        "Em análise" if lic else ("Concluído" if rng.random() < 0.5 else ""),
                        "Em andamento" if desap else ("Concluído" if rng.random() < 0.5 else ""),
                        obs, ini, fim_v, str(ano - 2019), str(ano - 2019), _p(acum_ant),
                        *[_p(pv[m]) for m in range(1, 13)], _p(sum(pv.values())),
                        _p(acum_ant + sum(pv.values())),
                        "Ano anterior não executada" if rng.random() < (0.4 if atrasou else 0.1) else "Ano"]))
                valor = rng.uniform(2e6, 80e6)
                valor_txt = f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                cad.append(";".join([
                    conc, str(id_seq), "ORIGINAL", "50501.000000/2020-00", "3.2.1", valor_txt, "07/2020",
                    desc, rodovia, tipo, _km(km), "", _km(ext) if ext else "", "Crescente", "Sul",
                    "Principal", "", "", "", "", "", "", "", ini, fim_contrato, "2", "2"]))
                if futuro:
                    continue
                for v in (1, 2):
                    data_rel = "2026-06-25-03:00" if v == 2 else f"{ano}-03-10-03:00"
                    acomp.append(";".join([
                        conc, str(id_seq), str(ano), str(v), data_rel, "ORIGINAL", "3.2.1", desc, rodovia,
                        tipo, _km(km), "", _km(ext) if ext else "", ini,
                        fim_plan if v == 2 else fim_inicial, "23420278",
                        _p(acum_ant),
                        *[f"{_p(prev_m[m])};{_p(exec_m[m] if v == 2 else 0)}" for m in range(1, 13)],
                        _p(prev_anual), _p(exec_anual if v == 2 else 0), _p(acum_ant + exec_anual)]))
                if conc in ("VIA SUL", "WAY 262") and ano < ANO_CORRENTE:
                    inexec = max(0.0, prev_anual - exec_anual)
                    inex.append(";".join([
                        conc, str(id_seq), str(ano), "1", f"30/03/{ano + 1}", "ORIGINAL", "3.2.1", desc,
                        rodovia, tipo, _km(km), "", "", ini, fim_plan, "123", _p(acum_ant), _p(prev_anual),
                        _p(acum_ant + prev_anual), _p(acum_ant + exec_anual), _p(inexec),
                        _p(inexec * 0.01) if inexec > 0 else ""]))

    (pasta / "acompanhamento" / "acompanhamento.csv").write_text("\n".join(acomp), encoding="utf-8")
    if incluir_opcionais:
        (pasta / "planejamento" / "planejamento.csv").write_text("\n".join(plan), encoding="utf-8")
        (pasta / "cadastro" / "cadastro.csv").write_text("\n".join(cad), encoding="utf-8")
        # Arquivo de inexecução em Windows-1252, como no portal
        (pasta / "inexecucao" / "inexecucao.csv").write_bytes("\n".join(inex).encode("cp1252"))
    return pasta


if __name__ == "__main__":
    import sys
    destino = Path(sys.argv[1] if len(sys.argv) > 1 else "dados_simulados")
    print(gerar(destino))
