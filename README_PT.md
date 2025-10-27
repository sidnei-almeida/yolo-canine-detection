# 🔬 DogBreed Vision - Sistema Profissional de Reconhecimento de Raças Caninas

Um projeto de visão computacional para portfolio que utiliza YOLOv8 para detectar e classificar raças de cães em imagens com alta precisão.

## 🎯 Sobre o Projeto

O **DogBreed Vision** é um sistema profissional de reconhecimento de raças caninas baseado em deep learning, treinado com o Stanford Dogs Dataset. Utilizando a arquitetura YOLOv8n (nano), o modelo é capaz de identificar **120 raças diferentes** de cães com alta precisão e velocidade.

### 🌟 Características do Portfolio

- Interface web interativa com Streamlit
- Carrossel de imagens para teste do modelo
- Análise em tempo real com feedback visual
- Visualizações de métricas com Plotly
- Design dark premium e profissional

### 📊 Métricas de Performance

- **mAP50-95**: 84.3%
- **Precisão**: 80.6%
- **Recall**: 76.3%
- **mAP50**: 84.5%
- **Épocas Treinadas**: 164 (com early stopping)

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+ (Recomendado: Python 3.11 ou 3.12)
- pip

> **⚠️ Nota:** Se você estiver usando Python 3.13+, todos os pacotes foram atualizados para versões compatíveis.

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/sidnei-almeida/analise_canina_yolo.git
cd analise_canina_yolo
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação Streamlit:
```bash
streamlit run app.py
```

4. Acesse no navegador: `http://localhost:8501`

## 📁 Estrutura do Projeto

```
analise_canina_yolo/
├── app.py                  # Aplicação Streamlit principal
├── config.yaml             # Configurações e thresholds do modelo
├── requirements.txt        # Dependências do projeto
├── README.md              # Documentação
├── .gitignore            # Arquivos ignorados pelo Git
├── run.sh                # Script de execução rápida
├── .streamlit/
│   └── config.toml       # Tema customizado do Streamlit
├── args/
│   └── args.yaml         # Argumentos de treinamento
├── weights/
│   ├── best.pt          # Melhor modelo treinado
│   └── last.pt          # Último checkpoint
├── results/
│   ├── *.png            # Curvas e matrizes de confusão
│   ├── *.jpg            # Batches de treino/validação
│   └── results.csv      # Histórico de treinamento
└── images/              # Imagens para teste (adicione suas imagens aqui)
```

## 🎨 Funcionalidades

### 🏠 Página Inicial
- Visão geral do projeto
- Métricas principais em destaque
- Exemplos de detecção
- Amostras de treinamento

### 📊 Análise de Resultados
- Gráficos interativos de evolução do treinamento
- Visualização de losses (treino e validação)
- Análise de Precisão vs Recall
- Matrizes de confusão
- Curvas PR, P, R e F1

### 🔮 Testar Modelo
- Teste com imagens pré-carregadas
- Upload de novas imagens
- Detecção em tempo real
- Visualização de resultados com bounding boxes
- Confiança de cada predição

### ℹ️ Sobre o Modelo
- Especificações técnicas completas
- Hiperparâmetros de treinamento
- Augmentações de dados utilizadas
- Métricas finais detalhadas
- Casos de uso e aplicações

## 🔬 Tecnologias Utilizadas

- **YOLOv8n**: Modelo de detecção de objetos otimizado
- **PyTorch**: Framework de deep learning
- **Streamlit**: Interface web interativa
- **Plotly**: Visualizações interativas
- **OpenCV**: Processamento de imagens
- **Stanford Dogs Dataset**: Dataset de treinamento

## 📸 Como Testar o Modelo

1. Adicione imagens PNG de cães na pasta `images/`
2. Acesse a seção "🔮 Testar Modelo" no app
3. Selecione uma imagem ou faça upload
4. Visualize as detecções em tempo real

## ⚙️ Configuração de Thresholds

O arquivo `config.yaml` permite ajustar todos os parâmetros do modelo sem modificar o código:

### Parâmetros de Detecção
```yaml
detection:
  confidence_threshold: 0.25    # Confiança mínima (0.0 - 1.0)
  iou_threshold: 0.45           # IoU para NMS
  max_detections: 10            # Máximo de detecções por imagem
  image_size: 640               # Tamanho de entrada (pixels)
```

### Visualização
```yaml
visualization:
  line_thickness: 2             # Espessura das bounding boxes
  show_labels: true             # Mostrar labels
  show_confidence: true         # Mostrar confiança
  confidence_format: "percentage"  # Formato da confiança
```

### Performance
```yaml
performance:
  use_half_precision: false     # Usar FP16 (GPU)
  device: "cpu"                 # Device: cpu, cuda, cuda:0, etc.
```

### Debug
```yaml
debug:
  show_inference_time: true     # Mostrar tempo de inferência
  save_predictions: false       # Salvar predições
  verbose: false                # Modo verbose
```

**Após modificar o `config.yaml`:**
- As mudanças são aplicadas automaticamente na próxima predição
- Use o botão "🔄 Recarregar Config" na sidebar para forçar atualização
- Verifique as configurações ativas na sidebar

## 🎯 Aplicações

- **Veterinária**: Identificação rápida de raças em clínicas
- **Abrigos**: Catalogação automática de animais
- **Apps Mobile**: Integração em aplicativos de pet care
- **Educação**: Ferramenta de aprendizado sobre raças caninas

## 📝 Dataset

O modelo foi treinado com o **Stanford Dogs Dataset**, que contém:
- 120 raças diferentes de cães
- Mais de 20.000 imagens
- Alta diversidade de poses e ambientes
- Anotações de alta qualidade

## 🏗️ Arquitetura do Modelo

- **Modelo Base**: YOLOv8n (Nano)
- **Input Size**: 640x640 pixels
- **Classes**: 120 raças
- **Framework**: Ultralytics YOLOv8

## 📈 Hiperparâmetros

- **Épocas**: 200 (early stop em 164)
- **Paciência**: 15 épocas
- **Learning Rate**: 0.01
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Batch Size**: Auto

## 🚀 Deploy

### Streamlit Cloud (Recomendado)

Este projeto está otimizado para deploy no Streamlit Cloud:

1. **Fork este repositório**
2. **Acesse:** [share.streamlit.io](https://share.streamlit.io)
3. **Configure:**
   - Repository: `seu-usuario/analise_canina_yolo`
   - Branch: `main`
   - Main file: `app.py`
   - Python version: `3.11`
4. **Deploy!**

📖 **Guia completo:** Veja [DEPLOY.md](DEPLOY.md) para instruções detalhadas

### Requisitos para Deploy

- ✅ PyTorch **CPU-only** (já configurado no `requirements.txt`)
- ✅ OpenCV **headless** (sem GUI)
- ✅ Git LFS para modelos grandes (`weights/best.pt`)
- ✅ Configurações otimizadas para CPU

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas funcionalidades
- Melhorar a documentação
- Adicionar novas visualizações

## 📄 Licença

Este projeto está sob a licença MIT.

## 👤 Autor

**Sidnei Almeida**
- GitHub: [@sidnei-almeida](https://github.com/sidnei-almeida)

Desenvolvido com ❤️ para o reconhecimento inteligente de raças caninas

---

**🐕 Canine AI** - Powered by YOLOv8 & Streamlit
