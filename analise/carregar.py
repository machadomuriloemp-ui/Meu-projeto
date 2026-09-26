"""Leitura e padronização das bases da ANTT.

Os arquivos variam entre si (maiúsculas nos nomes de colunas, acentos,
codificação UTF-8 ou Windows-1252, vírgula ou ponto decimal, "%" nos números).
Aqui tudo é convertido para um formato único.
"""
from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

MESES = range(1, 13)


# ----------------------------------------------------------------------------
# Utilidades de conversão
# ----------------------------------------------------------------------------
def normalizar_nome(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^0-9a-zA-Z]+", "_", texto.strip().lower())
    return texto.strip("_")


def para_numero(serie: pd.Series) -> pd.Series:
    """Converte '1.234,56', '16.44', ' 337,700', 'R$ 1.000,00', '55,7%' em float."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v).strip().replace("R$", "").replace("%", "").replace(" ", "")
        s = s.replace(" ", "")
        if s in {"", "-", "nan", "None"}:
            return np.nan
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        elif s.count(".") > 1:
            s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return np.nan

    return serie.map(conv).astype(float)


def para_percentual(serie: pd.Series) -> pd.Series:
    """Devolve percentual em pontos (0–100). Aceita '55,7%', '0.557' ou 55.7."""
    bruto = serie.astype(str)
    tinha_sinal = bruto.str.contains("%", regex=False).any()
    valores = para_numero(serie)
    if not tinha_sinal:
        validos = valores.dropna()
        if len(validos) and validos.abs().max() <= 1.0:
            valores = valores * 100
    return valores


def para_data(serie: pd.Series) -> pd.Series:
    s = serie.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    datas = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    datas[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    datas[~iso] = pd.to_datetime(s[~iso], dayfirst=True, errors="coerce")
    return datas


def _consertar_texto(texto):
    """Corrige acentos quebrados de arquivos UTF-8 lidos como Windows-1252."""
    if not isinstance(texto, str) or "Ã" not in texto:
        return texto
    try:
        return texto.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


# ----------------------------------------------------------------------------
# Leitura bruta
# ----------------------------------------------------------------------------
def ler_csv(caminho: Path) -> pd.DataFrame:
    bruto = Path(caminho).read_bytes()
    texto = None
    for cod in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            texto = bruto.decode(cod)
            break
        except UnicodeDecodeError:
            continue
    primeira = texto.splitlines()[0] if texto else ""
    sep = max([";", ",", "\t", "|"], key=primeira.count)
    opcoes = dict(sep=sep, dtype=str, keep_default_na=False, na_values=[""], engine="python")
    try:
        df = pd.read_csv(io.StringIO(texto), **opcoes)
    except pd.errors.ParserError:
        # Linhas com separador extra (ex.: ";" dentro de uma observação) são descartadas
        total = max(len(texto.splitlines()) - 1, 0)
        df = pd.read_csv(io.StringIO(texto), on_bad_lines="skip", **opcoes)
        print(f"  aviso: {Path(caminho).name}: {total - len(df)} linha(s) mal formatada(s) ignorada(s)")
    df.columns = [normalizar_nome(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].map(_consertar_texto)
    df = df.dropna(how="all")
    return df


# ----------------------------------------------------------------------------
# Mapeamento de colunas para nomes padronizados
# ----------------------------------------------------------------------------
ALIASES = {
    "concessionaria": ["concessionaria"],
    "id_sigicor": ["id_sigicor", "idsigicor"],
    "ano": ["ano_do_planejamento", "ano_planejamento", "ano"],
    "versao": ["versao"],
    "data_planejamento": ["data_do_planejamento", "data_planejamento"],
    "origem_per": ["origem_per"],
    "item_per": ["item_do_per", "item_per"],
    "descricao": ["descricao"],
    "rodovia": ["rodovia"],
    "tipo": ["tipo"],
    "km_inicial": ["km_inicial"],
    "km_final": ["km_final"],
    "extensao": ["extensao", "extensao_km"],
    "projeto_executivo": ["projeto_executivo"],
    "licenciamento_ambiental": ["licenciamento_ambiental"],
    "desapropriacao": ["desapropriacao"],
    "observacao": ["observacao", "observacoes"],
    "data_inicio_prevista": ["data_inicio_prevista", "previsao_de_inicio_obra",
                             "previsao_de_inicio_da_obra", "data_prevista_inicio"],
    "data_fim_prevista": ["data_fim_prevista", "previsao_de_termino_obra",
                          "previsao_de_termino_da_obra", "data_prevista_fim"],
    "ano_inicio": ["ano_inicio"],
    "ano_fim": ["ano_fim"],
    "exec_acum_anterior": ["executado_acumulado_anterior", "execucao_acumulada",
                           "executado_acumulado"],
    "previsto_anual": ["previsto_anual"],
    "executado_anual": ["executado_anual"],
    "executado_total": ["executado_total"],
    "inexecucao": ["inexecucao"],
    "fator_d": ["fator_d"],
    "tipo_planejamento": ["investimento_tipo_planejamento"],
    "valor_contratual": ["valor_contratual"],
    "data_base_valor": ["data_base_valor_contratual"],
}
for _m in MESES:
    ALIASES[f"prev_{_m}"] = [f"mes_{_m}_previsto", f"previsto_mes_{_m}"]
    ALIASES[f"exec_{_m}"] = [f"mes_{_m}_executado", f"executado_mes_{_m}"]

COLS_PCT = ["exec_acum_anterior", "previsto_anual", "executado_anual", "executado_total",
            "inexecucao", "fator_d"] + [f"prev_{m}" for m in MESES] + [f"exec_{m}" for m in MESES]
COLS_NUM = ["id_sigicor", "ano", "versao", "km_inicial", "km_final", "extensao",
            "valor_contratual", "ano_inicio", "ano_fim"]
COLS_DATA = ["data_planejamento", "data_inicio_prevista", "data_fim_prevista"]
COLS_TEXTO = ["concessionaria", "origem_per", "item_per", "descricao", "rodovia", "tipo",
              "projeto_executivo", "licenciamento_ambiental", "desapropriacao",
              "observacao", "tipo_planejamento"]


def padronizar(df: pd.DataFrame) -> pd.DataFrame:
    renomear = {}
    for padrao, alternativas in ALIASES.items():
        for alt in alternativas:
            if alt in df.columns and alt not in renomear:
                renomear[alt] = padrao
                break
    df = df.rename(columns=renomear)
    df = df.loc[:, ~df.columns.duplicated()]
    for c in COLS_PCT:
        if c in df:
            df[c] = para_percentual(df[c])
    for c in COLS_NUM:
        if c in df:
            df[c] = para_numero(df[c])
    for c in COLS_DATA:
        if c in df:
            df[c] = para_data(df[c])
    for c in COLS_TEXTO:
        if c in df:
            df[c] = df[c].astype("string").str.strip().replace({"": pd.NA})
    if "concessionaria" in df:
        df["concessionaria"] = df["concessionaria"].str.upper()
    return df


def carregar(arquivos: dict[str, list[Path]]) -> dict[str, pd.DataFrame]:
    bases = {}
    for nome, caminhos in arquivos.items():
        partes = []
        for c in caminhos:
            try:
                partes.append(padronizar(ler_csv(c)))
            except Exception as erro:  # noqa: BLE001
                print(f"  aviso: não foi possível ler {c.name}: {erro}")
        if partes:
            df = pd.concat(partes, ignore_index=True)
            df = df.dropna(subset=[c for c in ["concessionaria"] if c in df])
            bases[nome] = df
            print(f"[{nome}] {len(df):,} linhas, "
                  f"{df['concessionaria'].nunique() if 'concessionaria' in df else 0} concessionária(s)")
        else:
            bases[nome] = pd.DataFrame()
    return bases
