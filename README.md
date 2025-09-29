# DesafioAuto
# 📧 Classificador de E-mails com Resposta Automática

Este projeto implementa uma aplicação web que permite **classificar e-mails** como **Produtivos** ou **Improdutivos** e sugere automaticamente uma **resposta adequada** utilizando técnicas de **Processamento de Linguagem Natural (NLP)** e **Inteligência Artificial (IA)**.

---

## 🚀 Funcionalidades

### 🔹 Interface Web (HTML)
- Upload de arquivos `.txt` ou `.pdf` contendo e-mails.
- Campo para inserir texto de e-mails manualmente.
- Botão para enviar o conteúdo para processamento.
- Exibição do resultado:
  - Categoria do e-mail (**Produtivo** ou **Improdutivo**).
  - Resposta automática sugerida pelo sistema.

### 🔹 Backend em Python
- Leitura e processamento do conteúdo dos e-mails.
- Pré-processamento de texto com NLP:
  - Remoção de *stop words*.
  - Stemming / lematização.
- Classificação automática do e-mail.
- Geração de uma resposta automática com base na categoria.
- Integração direta com a interface web.

---

## 🛠️ Tecnologias Utilizadas

- **Frontend:** HTML5, CSS3, JavaScript.
- **Backend:** Python 3.x Flask ou.
- **NLP:** 
  - `NLTK` ou `spaCy` para pré-processamento de texto.
  - Modelos de classificação via  **OpenAI GPT**.
- **Upload de Arquivos:** `PyPDF2` (para leitura de PDF) e suporte nativo para `.txt`.

---



