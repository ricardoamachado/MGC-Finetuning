Diretrizes para Agentes de IA e Assistentes de Código.

Este documento serve como a referência para as restrições de arquitetura, stack tecnológico e padrões de código deste projeto de Music Information Retrieval (MIR). Todos os agentes de IA devem seguir estas regras ao gerar, refatorar ou analisar o código.

1. Gerenciamento de Pacotes e Ambiente

Gerenciador Oficial: Utilize exclusivamente o uv. Não utilize pip padrão, poetry, pipenv ou conda para instalar dependências ou gerenciar o ambiente virtual.

Instalação de Dependências: Ao sugerir comandos de instalação, use a sintaxe do uv, como uv pip install <pacote> ou uv add <pacote>.

PyTorch com CUDA 13.0: O projeto requer PyTorch configurado especificamente para CUDA 13.0. Veja exemplo do pytorch.toml dentro do repo.

GPU Utilizada: RTX 5070 Ti com 16 GB de VRAM.

2. Gerenciamento de dependências.

Manipulação de Dados:

Dar preferência a utilizar a biblioteca polars e pyarrow no lugar da biblioteca pandas.

Manipulação de Arquivos e Diretórios:

USAR: pathlib.Path (para navegação de diretórios, junção de caminhos e verificação de arquivos).

NÃO USAR: os.path.

Processamento de Áudio:

USAR: torchaudio e torchaudio.functional como prioridade. Caso seja necessário utilize a biblioteca librosa.

3. Padrões de Implementação.

Dataset: Estamos trabalhando com os subsets FMA-Small, FMA-Medium do FreeMusicArchive e o dataset GTZAN. Para salvar espaço no computador, estou somente com o fma_small e o GTZAN dentro da pasta de datasets, pois o objetivo inicial é fazer uma prova de conceito.

Train Validation Split FMA: Coluna `split` dentro do csv de metadados do FMA contém separação entre `training`, `validation` e `test`.

Train Validation Split GTZAN: Utilizar os arquivos de texto dentro da pasta GTZAN `train_stratified.txt`, `valid_stratified.txt` e `test_stratified.txt`.

Data Augmentation - APENAS GTZAN: Pitch shift de meio tom para baixo e meio tom para cima, pois o subconjunto de treino original possui apenas 500 amostras.

Caminho: Todos os datasets estarão localizados na pasta /datasets/ dentro da raiz do projeto.

Representação do sinal de áudio: O sinal de áudio deve ser transformado num Mel Espectrograma com a mesma extração usada no treinamento dos pesos de cada modelo. As representações não são intercambiáveis, pois os pesos pré-treinados dependem delas. A implementação está em `src/mgc_finetuning/data.py`.

Comum aos dois modelos:

- Entrada: áudio mono, reamostrado para 16 kHz.
- 128 bins de frequência e hop size de 10 ms (160 amostras).
- Saída com 1024 frames x 128 bins, o que corresponde a 10,24 s de áudio.
- Faixas mais longas: recorte aleatório no treino e primeiros 1024 frames na validação e na inferência.
- Faixas mais curtas: padding com zeros no final.

AST (`ASTMelSpectrogram`):

- Extração: `torchaudio.compliance.kaldi.fbank`.
- Janela: Hanning com comprimento de 25 ms (400 amostras) e FFT de 512 pontos.
- Pré-processamento do sinal: pré-ênfase de 0,97 e remoção de offset DC, sem dither.
- Banco Mel: de 20 Hz a 8 kHz.
- Compressão: logaritmo natural da energia em cada bin.
- Normalização: (log_mel - média) / (2 * desvio padrão), com média -4,2677393 e desvio padrão 4,5689974, lidos do `ASTFeatureExtractor`. É aplicada depois do recorte ou do padding.

BAT (`BATMelSpectrogram`):

- Extração: `torchaudio.transforms.MelSpectrogram`.
- Janela: Hann com comprimento de 64 ms (1024 amostras), FFT de 1024 pontos, com frames centralizados e padding por reflexão.
- Pré-processamento do sinal: sem pré-ênfase e sem remoção de offset DC.
- Banco Mel: escala HTK, de 0 Hz a 8 kHz.
- Compressão: espectro de potência convertido para dB com `AmplitudeToDB`, usando top_db=80.
- Normalização: min-max por amostra para o intervalo [0, 1]. A conversão para dB e a normalização são aplicadas depois do recorte e antes do padding, pois os pesos foram treinados com clipes de 10 s normalizados sobre toda a entrada.

Modelos Pré-treinados a ser utilizado: 

1) https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593. Audio Spectrogram Transformer (AST).
2) https://huggingface.co/lrauch/BAT-vit-b16-pretrainedAS2M. Better Audio Transformer (BAT).

4. Idioma

Os comentários e a documentação do código (docstrings) podem ser escritos em Inglês. O feedback para o usuário deve ser escrito em português.

