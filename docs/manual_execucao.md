# Manual de execução

## Ambiente
1. Instale Python 3.13 ou superior.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Geração dos artefatos
Execute:

```bash
python src/build_artifacts.py
```

O script gera a pasta `artifacts/` com:
- `relatorio_fuzzy.pdf`
- `apresentacao_fuzzy.pdf`

## O que o script faz
- instancia o sistema fuzzy;
- avalia cenários de teste;
- cria gráficos das funções de pertinência;
- gera uma superfície de decisão;
- exporta o relatório e a apresentação em PDF.

## Verificação rápida
Ao final da execução, o terminal mostra os principais resultados numéricos dos testes. Se os arquivos forem criados em `artifacts/`, a execução foi bem-sucedida.
