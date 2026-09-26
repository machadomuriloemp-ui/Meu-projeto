"""Download das bases do Portal de Dados Abertos da ANTT.

Estratégia:
1. Pergunta à API do portal (CKAN) quais arquivos CSV existem no conjunto —
   assim, se a ANTT publicar novas concessionárias (ex.: novos arquivos de
   inexecução), eles entram automaticamente.
2. Se a API falhar, usa as URLs fixas de config.py.
3. Se o download falhar, usa o que já estiver salvo em dados/brutos/<fonte>/.
   Você também pode colocar CSVs manualmente nessas pastas.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config

CABECALHOS = {"User-Agent": "analise-concessoes-antt/1.0 (+github)"}


def _descobrir_csvs(slug: str) -> list[str]:
    url = f"{config.PORTAL}/api/3/action/package_show"
    resp = requests.get(url, params={"id": slug}, headers=CABECALHOS, timeout=60)
    resp.raise_for_status()
    recursos = resp.json()["result"]["resources"]
    urls = []
    for r in recursos:
        formato = (r.get("format") or "").lower()
        link = r.get("url") or ""
        if formato == "csv" or link.lower().endswith(".csv"):
            urls.append(link)
    return urls


def _baixar(url: str, destino: Path, tentativas: int = 3) -> bool:
    for i in range(tentativas):
        try:
            resp = requests.get(url, headers=CABECALHOS, timeout=180)
            resp.raise_for_status()
            if len(resp.content) < 50:
                raise ValueError("arquivo vazio")
            destino.write_bytes(resp.content)
            return True
        except Exception as erro:  # noqa: BLE001
            print(f"    tentativa {i + 1} falhou: {erro}")
            time.sleep(3 * (i + 1))
    return False


def baixar_tudo(offline: bool = False) -> dict[str, list[Path]]:
    """Baixa (ou reaproveita) os arquivos de cada fonte e devolve os caminhos."""
    arquivos: dict[str, list[Path]] = {}
    for nome, fonte in config.FONTES.items():
        pasta = config.PASTA_DADOS / nome
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"[{nome}]")
        if not offline:
            try:
                urls = _descobrir_csvs(fonte["slug"]) or fonte["urls"]
                print(f"  {len(urls)} arquivo(s) encontrados no portal")
            except Exception as erro:  # noqa: BLE001
                print(f"  API do portal indisponível ({erro}); usando URLs fixas")
                urls = fonte["urls"]
            for url in urls:
                nome_arquivo = Path(urlparse(url).path).name or f"{nome}.csv"
                ok = _baixar(url, pasta / nome_arquivo)
                print(f"  {'ok ' if ok else 'ERRO'} {nome_arquivo}")
        encontrados = sorted(p for p in pasta.glob("*.csv"))
        if not encontrados and fonte["obrigatoria"]:
            raise SystemExit(
                f"Base obrigatória '{nome}' não encontrada. Coloque o CSV em {pasta}."
            )
        arquivos[nome] = encontrados
        print(f"  usando {len(encontrados)} arquivo(s) local(is)")
    return arquivos
