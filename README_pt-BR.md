# XTTS-WebUI

## Portátil O projeto agora tem uma versão portátil, portanto, você não precisa se dar ao trabalho de instalar todas as dependências.

[Clique aqui para fazer o download](https://huggingface.co/daswer123/xtts_portable/resolve/main/xtts-webui-v1_0-portable.zip?download=true)

Você não precisa de nada além do Windows e de uma placa de vídeo Nvidia com 6 GB de memória de vídeo para executá-lo.

## Idiomas do Leiame

[English](https://github.com/daswer123/xtts-webui/blob/main/README.md)

[Russian](https://github.com/daswer123/xtts-webui/blob/main/README_ru_RU.md)

[Português](https://github.com/daswer123/xtts-webui/blob/main/README_pt-BR.md)

## Sobre o projeto
XTTS-Webui é uma interface web que permite aproveitar ao máximo o XTTS. Existem outras redes neurais em torno desta interface que irão melhorar seus resultados. Você também pode ajustar o modelo e obter um modelo de voz de alta qualidade.

![image](https://github.com/RafaelGodoyEbert/xtts-webui/assets/78083427/3dd80284-88dd-4555-943b-04f13be88aea)


## Características principais
- Fácil trabalho com XTTSv2
- Processamento em lote para dublagem de um grande número de arquivos
- Capacidade de traduzir qualquer áudio com salvamento de voz
- Capacidade de melhorar resultados usando redes neurais e ferramentas de áudio automaticamente
- Capacidade de ajustar o modelo e usá-lo imediatamente
- Capacidade de usar ferramentas como: **RVC**, **OpenVoice**, **Resemble Enhance**, juntas e separadamente
- Capacidade de personalizar a geração de XTTS, todos os parâmetros, múltiplas amostras de fala

## PENDÊNCIA
- [x] Adicione uma barra de status com informações de progresso e erro
- [x] Integrar o treinamento na interface padrão
- [ ] Adicione a capacidade de transmitir para verificar o resultado
- [ ] Adicionar uma nova maneira de processar texto para narração
- [ ] Adicionar a capacidade de personalizar alto-falantes durante o processamento em lote
- [ ] Adicionar API

## Instalação

Use esta IU da web por meio de [Google Colab](https://colab.research.google.com/drive/1MrzAYgANm6u79rCCQQqBSoelYGiJ1qYL)

**Certifique-se de ter Python 3.10.x ou Python 3.11, CUDA 11.8 ou CUDA 12.1, Microsoft Builder Tools 2019 com pacote c++ e ffmpeg instalados**

### 1 Método, através de scripts

#### Windows
Para começar:
- Execute o arquivo 'install.bat'
- Para iniciar a UI da web, execute 'start_xtts_webui.bat'
- Abra seu navegador preferido e vá para o endereço local exibido no console.

#### Linux
Para começar:
- Execute o arquivo 'install.sh'
- Para iniciar a UI da web, execute 'start_xtts_webui.sh'
- Abra seu navegador preferido e vá para o endereço local exibido no console.

### 2 Método, Manual
Siga estas etapas para instalação:
1. Certifique-se de que `CUDA` esteja instalado
2. Clone o repositório: `git clone https://github.com/daswer123/xtts-webui`
3. Navegue até o diretório: `cd xtts-webui`
4. Crie um ambiente virtual: `python -m venv venv`
5. Ative o ambiente virtual:
    - No Windows use: `venv\scripts\activate`
    - No Linux use: `source venv\bin\activate`

6. Instale PyTorch e torchaudio com o comando pip:

    `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`

7. Instale todas as dependências de requirements.txt:

     `pip install -r requirements.txt`

## Executando o aplicativo

Para iniciar a interface, siga estas etapas:

#### Iniciando o XTTS WebUI:
Ative seu ambiente virtual:
```bash
venv/scripts/activate
```
ou se você estiver no Linux,
```bash
source venv/bin/activate
```
Em seguida, inicie o webui para xtts executando este comando:
```bash
python app.py
```

Aqui estão alguns argumentos de tempo de execução que podem ser usados ao iniciar o aplicativo:

| Argumento | Valor padrão | Descrição |
| --- | --- | --- |
| -hs, --host | 127.0.0.1 | O host ao qual vincular |
| -p, --porta | 8010 | O número da porta para escutar |
| -d, --dispositivo | cuda | Qual dispositivo usar (CPU ou Cuda) |
| -sf,--speaker_folder | alto-falantes/ | Diretório contendo amostras TTS |
|-o,--saída |"saída/" |Diretório de saída|
|-ms,--model-source |"local" |Defina a fonte do modelo: 'api' para a versão mais recente do repositório, inferência de API ou 'local' para usar inferência local e modelo v2.0.2|
|-v,-version |"v2.0.2" |Você pode especificar qual versão do xtts usar. Você pode especificar o nome do modelo customizado para esta finalidade, colocar a pasta em modelos e especificar o nome da pasta neste sinalizador |
|-l,--language 	|"auto"	|Idioma do Webui, você pode ver as traduções disponíveis na pasta i18n/locale.|
|--lowvram ||Ativa o modo low vram que alterna o modelo para RAM quando não está processando ativamente|
|--deepspeed ||Ativa a aceleração deepspeed. Funciona no Windows em python 3.10 e 3.11 |
|--share ||Permite o compartilhamento da interface fora do computador local|
|--rvc ||Habilitar pós-processamento RVC, todos os modelos devem estar localizados na pasta rvc|

### TTS -> RVC

Módulo para RVC, você pode habilitar o módulo RVC para pós-processar o áudio recebido, para isso você precisa adicionar o sinalizador --rvc se estiver executando no console ou gravá-lo no arquivo de inicialização

Para que o modelo funcione nas configurações RVC você precisa selecionar um modelo que você deve primeiro carregar na pasta voice2voice/rvc, o modelo e o arquivo de índice devem estar juntos, o arquivo de índice é opcional, cada modelo deve estar em um arquivo separado pasta.

## Diferenças entre xtts-webui e o [webui oficial](https://github.com/coqui-ai/TTS/pull/3296)

### Processamento de dados

1. Atualizado o sussurro mais rápido para 0.10.0 com a capacidade de selecionar um modelo v3 maior.
2. Pasta de saída alterada para pasta de saída dentro da pasta principal.
3. Se já existe um conjunto de dados na pasta de saída e você deseja adicionar novos dados, pode fazê-lo simplesmente adicionando um novo áudio, o que estava lá não será processado novamente e os novos dados serão adicionados automaticamente
4. Ligue o filtro VAD
5. Após a criação do conjunto de dados, é criado um arquivo que especifica o idioma do conjunto de dados. Este arquivo é lido antes do treino para que o idioma sempre corresponda. É conveniente quando você reinicia a interface

### Codificador XTTS de ajuste fino

1. Adicionada a capacidade de selecionar o modelo básico para XTTS, bem como quando você treinar novamente, não será necessário baixar o modelo novamente.
2. Adicionada capacidade de selecionar modelo personalizado como modelo base durante o treinamento, o que permitirá o ajuste fino do modelo já ajustado.
3. Adicionada possibilidade de obter a versão otimizada do modelo com 1 clique (etapa 2.5, colocar a versão otimizada na pasta de saída).
4. Você pode escolher se deseja excluir as pastas de treinamento depois de otimizar o modelo
5. Ao otimizar o modelo, o áudio de referência do exemplo é movido para a pasta de saída
6. Verificando a exatidão do idioma especificado e do idioma do conjunto de dados

### Inferência

1. Adicionada possibilidade de personalizar as configurações de inferência durante a verificação do modelo.

### Outro

1. Se você reiniciar acidentalmente a interface durante uma das etapas, poderá carregar dados em botões adicionais
2. Removida a exibição de logs, pois causava problemas ao reiniciar
3. O resultado final é copiado para a pasta finalizada, estes são arquivos totalmente finalizados, você pode movê-los para qualquer lugar e usá-los como modelo padrão
4. Adicionado suporte para japonês [aqui](https://github.com/daswer123/xtts-webui/issues/15#issuecomment-1869090189)



