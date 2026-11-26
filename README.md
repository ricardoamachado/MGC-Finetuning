# MGC Fine-tuning

Projeto em PyTorch para fine-tuning de transformers de áudio pré-treinados na classificação de gêneros musicais (*Music Genre Classification*), nos datasets FMA e GTZAN. Como os gêneros dos datasets não coincidem, cada um gera um modelo separado, escolhido pela flag `--dataset {fma_small,fma_medium,gtzan}` (padrão `fma_small`).

O modelo pré-treinado é escolhido pela flag `--model`:

- `ast` (padrão): Audio Spectrogram Transformer, com os pesos [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).
- `bat`: [Better Audio Transformer](https://github.com/houtan-ghaffari/BAT_ICML2026), com os pesos do encoder [`lrauch/BAT-vit-b16-pretrainedAS2M`](https://huggingface.co/lrauch/BAT-vit-b16-pretrainedAS2M).

Há duas metodologias de ajuste fino, escolhidas pela flag `--finetuning`:

- `head` (padrão): os pesos do backbone ficam congelados e somente a cabeça `Linear + Softmax` é treinada para as classes do dataset.
- `lora`: os pesos originais do backbone continuam congelados, mas adaptadores LoRA de baixo rank são injetados (via [`peft`](https://github.com/huggingface/peft)) nas projeções de atenção e treinados junto com a cabeça `Linear + Softmax`.

Os dois modelos usam os mesmos alvos padrão do LoRA, com o mesmo número de parâmetros nos adaptadores. A cabeça difere em um ponto: no BAT, ela tem uma `LayerNorm` antes da camada linear (veja [Implementação do BAT](#implementação-do-bat)).

## Formato dos dados

```text
datasets/
├── fma_small/
│   └── 000/000002.mp3
├── fma_medium/                   # opcional, para --dataset fma_medium
├── fma_tracks.csv
├── GTZAN/
│   ├── blues/blues.00000.wav
│   ├── ...
│   ├── rock/rock.00099.wav
│   ├── index.txt                 # opcional: artista e título de cada faixa
│   ├── train_stratified.txt
│   ├── valid_stratified.txt
│   └── test_stratified.txt
└── GTZAN_pitch_shifted/          # gerado automaticamente (cache da augmentation)
```

**FMA:** o `fma_tracks.csv` é usado para obter o gênero (`genre_top`) e a divisão oficial do FMA (coluna `split`). O treino usa `training` e a validação usa apenas `validation`; o split `test` fica reservado para o teste e não influencia a escolha da melhor época. No FMA-Small isso resulta em 6400 faixas de treino e 800 de validação, balanceadas entre os 8 gêneros. Os subconjuntos são aninhados (o `fma_medium` contém todas as faixas do `fma_small`), então `--dataset fma_medium` usa as 25.000 faixas da pasta `datasets/fma_medium/` (19.922 de treino e 2505 de validação). Use `--metadata` para informar outro caminho.

**GTZAN:** não há arquivo de metadados; o gênero é o nome da pasta de cada faixa. O treino usa `train_stratified.txt` e a validação usa `valid_stratified.txt` (`test_stratified.txt` fica reservado para o teste). Esses três arquivos de partição vêm do repositório [boblsturm/GTZAN](https://github.com/boblsturm/GTZAN), de Bob L. Sturm, que também traz partições *fault-filtered* (`*_filtered.txt`), não usadas aqui. O README desse repositório lista os trabalhos que usaram essas partições: Kereliuk, Sturm e Larsen, "Deep learning and music adversaries", *IEEE Transactions on Multimedia*, 17, 2015. Também lista a análise do dataset: Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use", [arXiv:1306.1461](https://arxiv.org/abs/1306.1461), 2013. O arquivo `jazz/jazz.00054.wav`, corrompido na distribuição original do GTZAN, é detectado e ignorado automaticamente, deixando 499 faixas no treino.

O `index.txt` do mesmo repositório ([link](https://github.com/boblsturm/GTZAN/blob/main/index.txt)) identifica o artista e o título de cada faixa do GTZAN. Ele é usado apenas pelo `predict` em lote, para incluir os nomes das músicas nas predições. Salve-o em `datasets/GTZAN/index.txt`. Segundo o próprio arquivo, ainda faltam informações de 17 faixas. Na versão atual, 15 faixas não têm artista nem título e 3 têm só um dos dois. Os campos desconhecidos ficam vazios. O `index.txt` pede que se cite Sturm, "The state of the art ten years after a state of the art: Future research in music information retrieval", *Journal of New Music Research*, 43(2), 147–172, 2014.

### Data augmentation do GTZAN

Como o treino do GTZAN tem apenas 500 faixas, cada faixa de treino é incluída também com pitch shift de um semitom (meio tom) para baixo e para cima, triplicando o conjunto (1497 amostras). A augmentation é aplicada apenas ao treino do GTZAN, nunca à validação nem ao FMA.

O pitch shift (`torchaudio.functional.pitch_shift`, phase vocoder com FFT de 2048 pontos) é lento demais para rodar na CPU a cada época, então as versões deslocadas são geradas uma única vez, na GPU, e salvas como WAV de 16 kHz em `datasets/GTZAN_pitch_shifted/semitones_{-1,+1}/` (≈1 GB, cerca de 1 minuto). Execuções seguintes reutilizam o cache. Use `--pitch-shift-semitones` para escolher outros deslocamentos, ou passe a flag sem valores para desativar a augmentation.

## Instalação e treino

No Windows, é preciso instalar o ffmpeg para executar o script.

```powershell
winget install "FFmpeg (Shared)"
```

```powershell
uv sync
uv run python -m mgc_finetuning.train --dataset fma_small --data-dir datasets --metadata datasets/fma_tracks.csv --epochs 30 --batch-size 16
uv run python -m mgc_finetuning.train --model bat --dataset gtzan --data-dir datasets --epochs 30 --batch-size 16
```

O PyTorch e o TorchAudio são obtidos do índice CUDA 13.0 configurado no `pyproject.toml`. O melhor checkpoint fica em `checkpoints/{model}-{dataset}-{finetuning}` (por exemplo `checkpoints/bat-gtzan-lora`), a menos que `--output-dir` seja informado, de modo que os modelos de cada arquitetura, dataset e metodologia não se sobrescrevem. A cada época, as losses e acurácias de treino e validação são impressas e registradas em `history.csv` dentro dessa pasta. O treino imprime também a quantidade de parâmetros treináveis; no modo `head`, imprime a lista deles, que deve conter apenas os parâmetros da cabeça (`classifier.*` e, no BAT, também `head_norm.*`).

O treino roda por até 30 épocas (`--epochs`) e usa early stopping: se a `validation_loss` não diminuir em 6 épocas seguidas (`--early-stopping-patience`), o treino é interrompido. Use `--early-stopping-patience 0` para desativar. O checkpoint salvo continua sendo o da época de maior acurácia de validação.

Para classificar um arquivo com um checkpoint treinado:

```powershell
uv run python -m mgc_finetuning.predict datasets/GTZAN/jazz/jazz.00000.wav --checkpoint checkpoints/ast-gtzan-head
```

A saída é um JSON com o gênero previsto (`label` e `probability`) e, em `top_k`, os 5 gêneros mais prováveis em ordem decrescente de probabilidade (`--top-k` muda a quantidade).

Para prever em lote todas as faixas do split de teste de um dataset, use `--dataset` no lugar do arquivo. O checkpoint precisa ter sido treinado no mesmo dataset:

```powershell
uv run python -m mgc_finetuning.predict --dataset gtzan --checkpoint checkpoints/ast-gtzan-head
uv run python -m mgc_finetuning.predict --dataset fma_small --checkpoint checkpoints/bat-fma_small-lora
```

O resultado é salvo em `predictions/{nome do checkpoint}_{dataset}_{split}.csv`, com uma linha por faixa e as colunas `track_id` (ID do FMA ou nome do arquivo no GTZAN), `artist` e `title` (nome real da música), `path`, `genre` (gênero real) e `top{n}_genre`/`top{n}_probability` para n de 1 a 5. No FMA, artista e título vêm do `fma_tracks.csv` (`artist.name` e `track.title`). No GTZAN, vêm do `index.txt` (veja [Formato dos dados](#formato-dos-dados)). Faixas não identificadas ficam com esses campos vazios. O script também imprime a acurácia top-1, top-2 e top-5 do split, a precisão, o recall e o F1-score macro (`precision_macro`, `recall_macro` e `f1_macro`, calculados sobre o gênero top-1) e a quantidade de faixas sem identificação. Arquivos ilegíveis são ignorados com um aviso.

### Recomendação de faixas por gênero

O script interativo `recommend` lê um CSV de `predictions/` e lista todas as faixas em que um gênero aparece no top-5 com probabilidade maior ou igual a um threshold:

```powershell
uv run python -m mgc_finetuning.recommend
```

O script pede o CSV (se houver mais de um), o gênero (nome ou número da lista) e o threshold (aceita `0.4` ou `0,4`). Em seguida, mostra as faixas ordenadas pela probabilidade, com as colunas `track_id`, `artist`, `title`, `true_genre` (gênero real), `rank` (posição do gênero no top-5 da faixa), `probability` e `path`. Faixas sem identificação aparecem como `(desconhecido)`. Por fim, pergunta se a lista deve ser salva em `predictions/recommendations/`. As buscas se repetem até que o gênero ou o threshold fiquem em branco.

Para uso sem interação, informe o gênero e o threshold:

```powershell
uv run python -m mgc_finetuning.recommend --predictions predictions/ast-gtzan-head_gtzan_test.csv --genre blues --threshold 0.4 --output blues.csv
```

Como o CSV guarda só o top-5, qualquer gênero fora dele tem probabilidade de no máximo 1/6 (≈0,17). Assim, a lista é completa para thresholds acima de 0,17. Com thresholds menores, o script avisa que faixas podem ter ficado de fora.

### Argumentos

`mgc_finetuning.train`:

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--model` | `ast` | Modelo pré-treinado: `ast` (Audio Spectrogram Transformer) ou `bat` (Better Audio Transformer). |
| `--dataset` | `fma_small` | Dataset de treino: `fma_small`, `fma_medium` ou `gtzan`. Cada dataset gera um modelo separado. |
| `--data-dir` | `datasets` | Pasta raiz dos datasets (contém `fma_small/`, `fma_medium/` e `GTZAN/`). |
| `--metadata` | `datasets/fma_tracks.csv` | CSV de metadados do FMA. Ignorado no GTZAN. |
| `--output-dir` | `checkpoints/{model}-{dataset}-{finetuning}` | Pasta onde o melhor checkpoint, o `history.csv` e as métricas são salvos. |
| `--model-name` | repositório oficial de `--model` | Repositório do Hugging Face com os pesos pré-treinados. |
| `--pitch-shift-semitones` | `-1 1` | Deslocamentos de pitch, em semitons, usados como augmentation no treino do GTZAN. Passe a flag sem valores para desativar. Ignorado no FMA. |
| `--epochs` | `30` | Número máximo de épocas. |
| `--early-stopping-patience` | `6` | Épocas seguidas sem redução da `validation_loss` antes de interromper o treino. `0` desativa. |
| `--batch-size` | `16` | Tamanho do batch de treino e validação. |
| `--finetuning` | `head` | Metodologia de ajuste fino: `head` (backbone congelado) ou `lora`. |
| `--learning-rate` | `1e-4` no AST, `1e-3` no BAT | Learning rate do AdamW para os parâmetros treináveis (cabeça e, com LoRA, os adaptadores). |
| `--lora-rank` | `1` | Rank `r` das matrizes LoRA `A` e `B`. Só com `--finetuning lora`. |
| `--lora-alpha` | `2` | Escala `alpha` do LoRA; a atualização é multiplicada por `alpha / r`. Só com `--finetuning lora`. |
| `--lora-dropout` | `0.2` | Dropout aplicado na entrada dos adaptadores LoRA. Só com `--finetuning lora`. |
| `--lora-target-modules` | `q_proj v_proj` | Camadas lineares que recebem adaptadores: `q_proj`, `k_proj`, `v_proj`, `o_proj` (atenção) e `fc1`, `fc2` (MLP) nos dois modelos, e `gate` só no BAT. Só com `--finetuning lora`. |
| `--seed` | `1337` | Semente aleatória do Python e do PyTorch. |

`mgc_finetuning.predict`:

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `audio` (posicional) | — | Arquivo de áudio a ser classificado. Obrigatório se `--dataset` não for informado. |
| `--checkpoint` | obrigatório | Pasta do checkpoint gerado pelo treino, por exemplo `checkpoints/ast-gtzan-head`. |
| `--top-k` | `5` | Quantidade de gêneros mais prováveis reportados. |
| `--dataset` | — | Modo em lote: `fma_small`, `fma_medium` ou `gtzan`. Substitui o arquivo de áudio. |
| `--split` | `test` | Split previsto no modo em lote: `train`, `validation` ou `test`. |
| `--data-dir` | `datasets` | Pasta raiz dos datasets. Só no modo em lote. |
| `--metadata` | `datasets/fma_tracks.csv` | CSV de metadados do FMA (gêneros, splits, artistas e títulos). Só no modo em lote com o FMA. |
| `--gtzan-index` | `{data-dir}/GTZAN/index.txt` | Índice de artistas e títulos do GTZAN. Se não existir, as predições saem sem nomes, com um aviso. |
| `--output` | `predictions/{nome do checkpoint}_{dataset}_{split}.csv` | CSV com as predições do modo em lote. |
| `--batch-size` | `16` | Tamanho do batch do modo em lote. |

`mgc_finetuning.recommend`:

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--predictions` | escolha interativa | CSV de predições gerado pelo `predict --dataset`. |
| `--predictions-dir` | `predictions` | Pasta em que os CSVs de predições são procurados. |
| `--genre` | pergunta interativamente | Gênero buscado (sem diferenciar maiúsculas e minúsculas). |
| `--threshold` | pergunta interativamente | Probabilidade mínima, entre 0 e 1. |
| `--output` | — | CSV de saída quando `--genre` e `--threshold` são informados. |

`mgc_finetuning.plot`:

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--checkpoint` | obrigatório | Pasta do checkpoint com os CSVs gerados pelo treino. |
| `--output-dir` | `{checkpoint}/figures` | Pasta onde os PDFs são salvos. |

### Métricas de desempenho

Todas as métricas são calculadas no conjunto de validação e salvas em `--output-dir`:

| Arquivo | Conteúdo |
| --- | --- |
| `history.csv` | Por época: losses, acurácias e `validation_precision_macro`, `validation_recall_macro` e `validation_f1_macro`. |
| `per_class_metrics.csv` | Precision, recall, F1 e suporte por classe na época de melhor acurácia de validação. |
| `confusion_matrix.csv` | Matriz de confusão normalizada por linha da melhor época: a célula (`true_label`=i, coluna j) é P(predito = j \| verdadeiro = i). |

As métricas macro-averaged são a média simples das métricas por classe, de modo que todos os gêneros têm o mesmo peso independentemente do número de faixas; uma classe nunca prevista entra com valor 0. Os arquivos da melhor época são sobrescritos sempre que um novo checkpoint é salvo.

### Figuras

O script `mgc_finetuning.plot` lê os CSVs de um checkpoint e gera as figuras em PDF, com textos em inglês, prontas para trabalhos acadêmicos:

```powershell
uv run python -m mgc_finetuning.plot --checkpoint checkpoints/ast-gtzan-lora
```

| Arquivo | Conteúdo |
| --- | --- |
| `loss.pdf` | Loss de treino e validação por época. |
| `accuracy.pdf` | Acurácia de treino e validação por época. |
| `macro_metrics.pdf` | Precision, recall e F1 macro-averaged de validação por época. |
| `confusion_matrix.pdf` | Matriz de confusão normalizada da melhor época, com o valor de cada célula. |
| `per_class_metrics.pdf` | Precision, recall e F1 por classe na melhor época. |

As curvas marcam a época do checkpoint salvo (melhor acurácia de validação). As fontes são embutidas como TrueType (sem fontes Type 3) e o texto continua editável. As cores vêm de uma paleta segura para daltonismo, e as séries das curvas também usam marcadores diferentes, para continuarem distinguíveis em impressão em tons de cinza. Checkpoints antigos sem métricas macro ou sem os CSVs da melhor época geram apenas as figuras possíveis.

#### Espectrograma Mel do BAT

O script `mgc_finetuning.spectrogram` gera, para cada arquivo de áudio (MP3, WAV etc.), um PDF com o espectrograma Mel em dB calculado pelo pré-processamento do BAT. A figura mostra os 10,24 s usados na inferência e reproduz os passos do `BATMelSpectrogram` em modo de avaliação até a conversão para dB, com `top_db=80`. Só a normalização min-max para [0, 1] é omitida, para que a magnitude mantenha a unidade:

```powershell
uv run python -m mgc_finetuning.spectrogram datasets/GTZAN/jazz/jazz.00000.wav datasets/fma_small/000/000182.mp3
```

Cada figura é salva em `figures/spectrograms/{nome do arquivo}_bat_mel.pdf`. O eixo vertical mostra a frequência em Mel (escala HTK, de 0 a cerca de 2840 Mel, que corresponde a 0–8 kHz), com cada um dos 128 bins posicionado no valor Mel do seu centro. O eixo horizontal mostra o tempo em segundos. A barra de cores ("Magnitude (dB)") mostra o nível em dB, relativo a uma referência 1 (o fundo de escala digital), e não em dB SPL. Como 10·log₁₀|X|² = 20·log₁₀|X|, o valor é o mesmo para a potência e para a magnitude. Ela cobre os 80 dB abaixo do pico do clipe que o `top_db` preserva. A entrada do modelo é essa mesma matriz mapeada linearmente para [0, 1]. Nas faixas com menos de 10,24 s, os frames que o modelo recebe como padding com zeros ficam em branco, e o início do padding é marcado na figura. O mapa de cores padrão é o `viridis` do matplotlib, perceptualmente uniforme e legível também em impressão em tons de cinza. A imagem é rasterizada dentro do PDF para manter o arquivo pequeno, enquanto eixos e textos continuam vetoriais e com fontes TrueType.

A função `plot_mel_spectrogram(audio_path, output_path, start_seconds=0.0, colormap=None)` também pode ser importada de `mgc_finetuning.spectrogram` para uso em outros scripts.

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `audio` (posicional) | obrigatório | Um ou mais arquivos de áudio. |
| `--output-dir` | `figures/spectrograms` | Pasta dos PDFs. |
| `--start` | `0` | Início, em segundos, do trecho de 10,24 s. Com o padrão, a figura mostra o trecho usado na inferência. |
| `--cmap` | `viridis` | Colormap do matplotlib, por exemplo `magma` ou `inferno`. |

### Ajuste fino com LoRA

```powershell
uv run python -m mgc_finetuning.train --model bat --dataset gtzan --data-dir datasets --finetuning lora
```

Os hiperparâmetros do LoRA (`--lora-rank`, `--lora-alpha`, `--lora-dropout` e `--lora-target-modules`) estão descritos em [Argumentos](#argumentos). Com os valores padrão e as 8 classes do FMA-Small, são treinados 43.016 parâmetros no AST e 44.552 no BAT (≈0,05% do modelo); a diferença é a `LayerNorm` da cabeça do BAT. No modo `lora` o backbone fica em modo de treino para que o dropout dos adaptadores seja aplicado. A metodologia e os hiperparâmetros do LoRA são salvos no `config.json` do checkpoint, de modo que o `predict` reconstrói o modelo correto sem flags adicionais.

## Pré-processamento

Durante o treinamento e a avaliação, os datasets retornam diretamente o espectrograma Mel, nunca a forma de onda. Cada modelo usa o pré-processamento com que seus pesos foram treinados, e o `--model` seleciona o correto:

| | AST | BAT |
| --- | --- | --- |
| Extração | `torchaudio.compliance.kaldi.fbank` | `torchaudio.transforms.MelSpectrogram` |
| Bins Mel e hop | 128 bins, 10 ms | 128 bins, 10 ms |
| Janela | Hanning de 25 ms, FFT de 512 pontos | Hann de 64 ms, FFT de 1024 pontos |
| Outros passos | Pré-ênfase, remoção de offset DC | Conversão para dB com `top_db=80` |
| Normalização | Média e desvio do `ASTFeatureExtractor` | Min-max por amostra, para o intervalo [0, 1] |

Nos dois casos, a entrada tem 1024 frames, cerca de 10 s das faixas de 30 s. O treino usa recorte aleatório como augmentation; validação e inferência usam recorte determinístico. No BAT, a normalização é calculada depois do recorte, porque os pesos foram treinados em clipes de 10 s normalizados sobre toda a entrada.

## Implementação do BAT

O encoder fica em `src/mgc_finetuning/bat.py`, adaptado do [repositório oficial](https://github.com/houtan-ghaffari/BAT_ICML2026) (licença MIT), e os pesos são carregados diretamente do `model.safetensors`, sem `trust_remote_code`. A camada `qkv` original é separada em `q_proj`, `k_proj` e `v_proj`, e `attn.proj` passa a se chamar `o_proj`. Essas mudanças não alteram as saídas e permitem aplicar o LoRA nas mesmas camadas do AST.

As features do token CLS do BAT têm escala muito menor que as do AST (norma L2 média de 0,15 contra 35,6 no GTZAN), e uma camada linear aplicada diretamente sobre elas quase não aprende. Por isso, a cabeça do BAT é `LayerNorm + Dropout + Linear`, como nas configurações oficiais de linear probing e finetuning (`head_norm: true`). Mesmo assim, o BAT aprende bem mais devagar que o AST com learning rate de `1e-4`, então o padrão dele é `1e-3`, o mesmo das configurações oficiais de linear probing. Para comparar os modelos com o mesmo valor, informe `--learning-rate` explicitamente.

Se usar o BAT em um trabalho, cite o artigo [BAT: Better Audio Transformer Guided by Convex Gated Probing](https://arxiv.org/abs/2602.16305) (ICML 2026).
