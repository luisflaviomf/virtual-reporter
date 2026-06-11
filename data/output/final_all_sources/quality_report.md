# Relatorio de Qualidade - Todas as Fontes IFMT

## Fontes consolidadas

- `final_combined`: portal historico por IP, portal atual e campi atuais.
- `final_antigoportal`: portal antigo em subdominios `*.antigoportal.ifmt.edu.br`.

## Resumo

- Total bruto esperado: 10189
- Total bruto obtido: 10189
- Total canonico final: 8681
- Duplicados nao-canonicos: 1508
- Grupos de duplicidade: 665
- Grupos de duplicidade entre datasets: 44

## Cobertura dos campos principais

- title: 100.0%
- body_text: 99.75%
- publication_date: 89.07%
- author_or_unit: 100.0%
- links: 59.27%
- attachments: 11.05%
- images: 15.39%

## Preservacao dos dados

O arquivo `news_articles_full.jsonl` preserva todos os registros brutos, inclusive duplicatas, `needs_review` e erros. A deduplicacao apenas marca `duplicate_group_id`, `duplicate_of` e `is_canonical`; registros nao sao removidos do arquivo full.

## Limitacoes conhecidas

- O antigoportal tem mais ruido estrutural e mais paginas com corpo fraco; esses registros foram preservados para rastreabilidade.
- Algumas datas do HTML legado sao ambiguas e podem gerar anos improvaveis, exigindo QA posterior.
- A deduplicacao aproximada e conservadora, mas grupos com titulos genericos ainda devem ser revisados antes de uso definitivo.
- Arquivos de midia e links foram consolidados por metadados; download de anexos/midias nao faz parte desta etapa.