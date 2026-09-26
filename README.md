# Cumprimento do planejado nas concessões rodoviárias (ANTT)

Analisa, com os **dados abertos da ANTT**, se as concessionárias de rodovias federais
cumprem o que planejam mês a mês. O projeto também:

- **prevê** quais concessionárias e obras têm mais chance de atrasar no ano;
- aponta os **motivos** mais associados aos atrasos;
- estima o **impacto**: km não entregues, valor não executado, meses de postergação e Fator D.

📄 **Resultado:** [`saida/RELATORIO.md`](saida/RELATORIO.md) (e o Excel `saida/analise_concessoes.xlsx`)

---

## Como rodar pelo celular

1. No app ou site do GitHub, abra este repositório e toque na aba **Actions**.
2. Escolha **Rodar análise** na lista da esquerda.
3. Toque em **Run workflow** e depois no botão verde **Run workflow**.
4. Espere cerca de 3 minutos, até aparecer o ✅ verde.
5. Volte à página inicial do repositório e abra **`saida/RELATORIO.md`**.

A análise também roda **sozinha todo dia 16 de cada mês**, depois da atualização mensal da ANTT.

### Se o download da ANTT falhar

Às vezes o portal da ANTT fica fora do ar ou bloqueia acessos. Nesse caso:

1. Baixe o CSV no portal: [Acompanhamento Anual](https://dados.antt.gov.br/dataset/acompanhamento-anual).
2. No GitHub, abra a pasta `dados/brutos/acompanhamento/` e envie o arquivo em **Add file → Upload files**.
   Faça o mesmo para as outras bases, se quiser.
3. Rode a análise com a opção **"Usar só os CSVs já salvos"** marcada.

---

## Bases usadas

Todas vêm do Portal de Dados Abertos da ANTT (sistema SIGICOR) e são preenchidas pelas concessionárias.

| Base | Para que serve | Obrigatória? |
|---|---|---|
| [Acompanhamento Anual de Investimentos](https://dados.antt.gov.br/dataset/acompanhamento-anual) | % previsto × % executado de cada obra, mês a mês | Sim |
| [Planejamento Anual de Investimentos](https://dados.antt.gov.br/dataset/planejamento-anual) | Versões do plano, status de projeto, licenciamento e desapropriação, e observações | Não, mas melhora os motivos |
| [Cadastros de Investimentos](https://dados.antt.gov.br/dataset/cadastros-investimentos) | Valor contratual e cronograma original | Não, mas permite calcular R$ e postergação |
| [Inexecução Anual](https://dados.antt.gov.br/dataset/inexecucao) | Inexecução oficial e Fator D (desconto na tarifa) | Não |

Os arquivos de cada base são descobertos automaticamente pela API do portal. Se a ANTT
publicar arquivos novos, por exemplo de mais concessionárias, eles entram sozinhos.

## O que o relatório mostra

1. **Quem tende a atrasar.** Mostra a chance de cada concessionária executar menos de 90% do plano do ano.
2. **Histórico e tendência.** Mostra o índice executado ÷ previsto mês a mês e se ele está melhorando ou piorando.
3. **Motivos.** Traz os fatores que mais aumentam a chance de atraso (*lift*), os fatores por
   concessionária e os tipos de obra que mais atrasam.
4. **Impacto.** Traz o que deixou de ser entregue (km, R$, meses) e o valor em risco no ano corrente.
5. **Obras com maior risco.** Lista cada obra com a chance de atraso e os motivos prováveis.
6. **Confiabilidade.** Mostra a qualidade do modelo, medida pela AUC.

## Como funciona

```
analise/
  baixar.py       baixa as bases do portal (com fallback para arquivos locais)
  carregar.py     lê e padroniza (acentos, "55,7%", "1.234,56", datas, nomes de colunas)
  cumprimento.py  previsto x executado por mês, por ano e tendência
  motivos.py      fatores de atraso (pendências, observações, revisões do plano)
  obras.py        tabela obra × ano com todas as variáveis
  modelo.py       probabilidade de atraso por obra + simulação por concessionária
  impacto.py      km, R$, postergação, inexecução e Fator D
  relatorio.py    RELATORIO.md, gráficos, Excel e CSVs
```

- **Atraso:** a obra executou menos de 90% do % previsto para o ano. Esse limite fica em `analise/config.py`.
- **Meses sem informação:** meses que a concessionária ainda não informou **não** contam como atraso.
- **Modelo:** o código compara uma regressão logística e um gradient boosting e escolhe o melhor.
  A validação é feita de dois jeitos: cruzada, e treinando com anos antigos para testar no último ano.
- **Sem "olhar o futuro":** nos anos passados, o modelo só usa as versões do plano que já existiam
  no mesmo ponto do ano em que estamos agora. Revisões feitas depois do atraso não entram.
- **Risco da concessionária:** vem de uma simulação de Monte Carlo com as probabilidades de cada
  obra. A simulação considera que obras da mesma concessionária tendem a atrasar juntas.

## Testes

Os testes rodam automaticamente no GitHub a cada mudança (aba **Actions → Testes**). Eles usam
dados **simulados, no mesmo formato dos CSVs reais da ANTT**, com um "gabarito" conhecido, e
conferem se a análise:

- lê corretamente os formatos brasileiros e os cabeçalhos reais das quatro bases;
- identifica a concessionária que está piorando e a que está melhorando;
- aponta licenciamento e desapropriação como motivos de atraso;
- não conta meses futuros como descumprimento;
- não usa informação do futuro (há um teste específico para isso);
- continua funcionando com bases faltando, com pouco histórico ou com o portal fora do ar.

## Rodar no computador (opcional)

```bash
pip install -r requirements.txt
python -m pytest testes         # testes
python -m analise               # baixa os dados e gera saida/RELATORIO.md
python -m analise --offline     # usa só os CSVs em dados/brutos/
```

## Limitações

- Nem todas as concessões estão no SIGICOR, e alguns campos (observações, valor contratual)
  podem vir vazios.
- Os motivos são **associações** estatísticas e registros das próprias bases, não prova de causa.
- O índice mensal dá o mesmo peso a cada obra, porque usa o % de avanço físico.
