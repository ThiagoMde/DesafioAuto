"""
Sistema de Classificação de Emails - Backend
Desenvolvido com FastAPI, OpenAI GPT e técnicas de NLP
"""
import uvicorn
import os
import re
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# FastAPI e dependências web
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

# Processamento de texto e NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import RSLPStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob
import re
from collections import Counter

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# OpenAI Integration
import openai
from openai import AsyncOpenAI

# Processamento de arquivos
import PyPDF2
import docx
from io import BytesIO
#import magic

# Utils
import json
import hashlib
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CONFIGURAÇÕES E CONSTANTES
# ================================

class Settings:
    """Configurações da aplicação"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = "gpt-4-turbo-preview"
    MAX_TOKENS = 1500
    TEMPERATURE = 0.3
    
    # Configurações de arquivo
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}
    
    # Cache e persistência
    CACHE_DIR = Path("cache")
    MODELS_DIR = Path("models")
    
    def __post_init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)

settings = Settings()

# ================================
# MODELOS DE DADOS
# ================================

class EmailCategory(str, Enum):
    """Categorias de email"""
    PRODUCTIVE = "productive"
    UNPRODUCTIVE = "unproductive"

class ResponseTone(str, Enum):
    """Tons de resposta"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"

class EmailInput(BaseModel):
    """Modelo para entrada de email via texto"""
    content: str
    tone: ResponseTone = ResponseTone.PROFESSIONAL
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Conteúdo do email não pode estar vazio")
        if len(v) < 10:
            raise ValueError("Conteúdo muito curto")
        if len(v) > 50000:
            raise ValueError("Conteúdo muito longo")
        return v.strip()

class ClassificationResult(BaseModel):
    """Resultado da classificação"""
    category: EmailCategory
    confidence: float
    reasoning: str
    keywords: List[str]
    sentiment: Dict[str, float]
    entities: List[Dict[str, str]]

class ResponseSuggestion(BaseModel):
    """Sugestão de resposta"""
    text: str
    tone: ResponseTone
    alternatives: List[str]

class EmailProcessingResult(BaseModel):
    """Resultado completo do processamento"""
    classification: ClassificationResult
    suggested_response: ResponseSuggestion
    processing_time: float
    timestamp: datetime
    content_hash: str

# ================================
# PROCESSAMENTO DE TEXTO E NLP
# ================================

class TextPreprocessor:
    """Classe para pré-processamento de texto"""
    
    def __init__(self):
        self._download_nltk_data()
        self._load_spacy_model()
        self.stemmer = RSLPStemmer()
        
        # Stop words em português
        self.stop_words = set(stopwords.words('portuguese'))
        self.stop_words.update(['email', 'mensagem', 'assunto', 'enviar', 'receber'])
        
        # Palavras-chave para classificação
        self.productive_keywords = {
            'reuniao', 'projeto', 'prazo', 'entrega', 'relatorio', 'apresentacao',
            'cliente', 'proposta', 'orcamento', 'contrato', 'cronograma', 'agenda',
            'tarefa', 'trabalho', 'negocio', 'vendas', 'servico', 'produto',
            'analise', 'feedback', 'aprovacao', 'desenvolvimento', 'planejamento'
        }
        
        self.unproductive_keywords = {
            'spam', 'promocao', 'desconto', 'oferta', 'gratis', 'ganhe',
            'loteria', 'premio', 'clique', 'urgente', 'oportunidade unica',
            'marketing', 'publicidade', 'newsletter', 'assinatura', 'cancelar'
        }
    
    def _download_nltk_data(self):
        """Download dos recursos necessários do NLTK"""
        required_nltk_data = [
            'punkt', 'stopwords', 'rslp', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words'
        ]
        
        for data in required_nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                logger.info(f"Baixando {data}...")
                nltk.download(data, quiet=True)
    
    def _load_spacy_model(self):
        """Carrega modelo do spaCy para português"""
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            logger.warning("Modelo do spaCy não encontrado. Usando alternativa básica.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto"""
        # Remove caracteres especiais e normaliza
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """Tokeniza e filtra palavras"""
        tokens = word_tokenize(text, language='portuguese')
        
        # Filtra stop words e palavras muito curtas
        filtered_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return filtered_tokens
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extrai características do texto"""
        clean_text = self.clean_text(text)
        tokens = self.tokenize_and_filter(clean_text)
        
        # Análise de sentimento
        blob = TextBlob(text)
        sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Extração de entidades (se spaCy disponível)
        entities = []
        if self.nlp:
            doc = self.nlp(text)
            entities = [
                {'text': ent.text, 'label': ent.label_, 'start': ent.start, 'end': ent.end}
                for ent in doc.ents
            ]
        
        # Contagem de palavras-chave
        productive_count = sum(1 for token in tokens if token in self.productive_keywords)
        unproductive_count = sum(1 for token in tokens if token in self.unproductive_keywords)
        
        # Estatísticas do texto
        sentences = sent_tokenize(text, language='portuguese')
        
        features = {
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            'productive_keywords': productive_count,
            'unproductive_keywords': unproductive_count,
            'keyword_ratio': productive_count / (unproductive_count + 1),
            'sentiment': sentiment,
            'entities': entities,
            'top_words': dict(Counter(tokens).most_common(10))
        }
        
        return features

# ================================
# PROCESSAMENTO DE ARQUIVOS
# ================================

class FileProcessor:
    """Classe para processamento de diferentes tipos de arquivo"""
    
    @staticmethod
    def validate_file(file: UploadFile) -> bool:
        """Valida se o arquivo é permitido"""
        # Verifica extensão
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            return False
        
        # Verifica tamanho
        if file.size > settings.MAX_FILE_SIZE:
            return False
        
        return True
    
    @staticmethod
    async def extract_text_from_file(file: UploadFile) -> str:
        """Extrai texto do arquivo baseado no tipo"""
        content = await file.read()
        file_ext = Path(file.filename).suffix.lower()
        
        try:
            if file_ext == '.txt':
                return content.decode('utf-8')
            
            elif file_ext == '.pdf':
                return FileProcessor._extract_from_pdf(content)
            
            elif file_ext in ['.docx', '.doc']:
                return FileProcessor._extract_from_docx(content)
            
            else:
                raise ValueError(f"Tipo de arquivo não suportado: {file_ext}")
                
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {file.filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo: {str(e)}")
    
    @staticmethod
    def _extract_from_pdf(content: bytes) -> str:
        """Extrai texto de PDF"""
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text_parts = []
        
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
        
        return '\n'.join(text_parts)
    
    @staticmethod
    def _extract_from_docx(content: bytes) -> str:
        """Extrai texto de DOCX"""
        doc = docx.Document(BytesIO(content))
        text_parts = []
        
        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)
        
        return '\n'.join(text_parts)

# ================================
# CLASSIFICADOR DE EMAILS
# ================================

class EmailClassifier:
    """Classe principal para classificação de emails"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.traditional_classifier = None
        self._load_or_train_traditional_model()
    
    def _load_or_train_traditional_model(self):
        """Carrega ou treina modelo tradicional como fallback"""
        model_path = settings.MODELS_DIR / "traditional_classifier.joblib"
        
        if model_path.exists():
            self.traditional_classifier = joblib.load(model_path)
            logger.info("Modelo tradicional carregado")
        else:
            logger.info("Treinando modelo tradicional...")
            self._train_traditional_model()
    
    def _train_traditional_model(self):
        """Treina modelo tradicional básico"""
        # Dados de exemplo para treinamento
        sample_data = [
            ("Reunião de projeto agendada para segunda", "productive"),
            ("Relatório mensal anexo para análise", "productive"),
            ("Proposta comercial para novo cliente", "productive"),
            ("Cronograma de entregas atualizado", "productive"),
            ("OFERTA IMPERDÍVEL! Clique agora!", "unproductive"),
            ("Ganhe dinheiro fácil em casa", "unproductive"),
            ("Newsletter semanal de produtos", "unproductive"),
            ("Promoção relâmpago por tempo limitado", "unproductive"),
        ]
        
        # Expandir dados com variações
        expanded_data = []
        for text, label in sample_data:
            expanded_data.append((text, label))
            # Adicionar variações
            variations = self._generate_variations(text)
            for variation in variations:
                expanded_data.append((variation, label))
        
        # Preparar dados
        texts = [text for text, _ in expanded_data]
        labels = [label for _, label in expanded_data]
        
        # Vectorizar
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Treinar
        classifier = MultinomialNB()
        classifier.fit(X, labels)
        
        # Salvar
        model_data = {
            'classifier': classifier,
            'vectorizer': vectorizer
        }
        
        joblib.dump(model_data, settings.MODELS_DIR / "traditional_classifier.joblib")
        self.traditional_classifier = model_data
        logger.info("Modelo tradicional treinado e salvo")
    
    def _generate_variations(self, text: str) -> List[str]:
        """Gera variações de texto para aumentar dados de treino"""
        variations = []
        words = text.split()
        
        # Variação com sinônimos básicos
        synonyms = {
            'reunião': ['meeting', 'encontro'],
            'projeto': ['projeto', 'trabalho'],
            'oferta': ['promoção', 'desconto'],
            'clique': ['acesse', 'visite']
        }
        
        for original, syns in synonyms.items():
            for syn in syns:
                if original in text.lower():
                    variations.append(text.lower().replace(original, syn))
        
        return variations[:3]  # Limita variações
    
    async def classify_with_gpt(self, text: str) -> Dict[str, Any]:
        """Classifica email usando GPT"""
        if not settings.OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key não configurada")
        
        prompt = f"""
        Analise o seguinte email e classifique-o como PRODUTIVO ou IMPRODUTIVO.

        Um email PRODUTIVO contém:
        - Informações relevantes para trabalho/negócios
        - Solicitações ou comunicações importantes
        - Conteúdo relacionado a projetos, reuniões, tarefas
        - Comunicações profissionais legítimas

        Um email IMPRODUTIVO contém:
        - Spam ou propaganda
        - Ofertas comerciais não solicitadas
        - Conteúdo promocional
        - Newsletters irrelevantes
        - Comunicações sem valor profissional

        Email para análise:
        "{text[:2000]}"

        Responda APENAS em formato JSON com esta estrutura:
        {{
            "category": "productive" ou "unproductive",
            "confidence": número entre 0 e 1,
            "reasoning": "explicação breve da classificação",
            "keywords": ["palavra1", "palavra2", "palavra3"],
            "main_topics": ["tópico1", "tópico2"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)
            
        except json.JSONDecodeError:
            logger.error("Erro ao parsear resposta do GPT")
            return await self._fallback_classification(text)
        except Exception as e:
            logger.error(f"Erro na classificação GPT: {str(e)}")
            return await self._fallback_classification(text)
    
    async def _fallback_classification(self, text: str) -> Dict[str, Any]:
        """Classificação de fallback usando modelo tradicional"""
        if not self.traditional_classifier:
            # Classificação básica por palavras-chave
            features = self.preprocessor.extract_features(text)
            
            if features['productive_keywords'] > features['unproductive_keywords']:
                category = "productive"
                confidence = min(0.8, 0.5 + features['productive_keywords'] * 0.1)
            else:
                category = "unproductive"  
                confidence = min(0.8, 0.5 + features['unproductive_keywords'] * 0.1)
            
            return {
                "category": category,
                "confidence": confidence,
                "reasoning": f"Classificação baseada em palavras-chave ({features['productive_keywords']} prod vs {features['unproductive_keywords']} improd)",
                "keywords": list(features['top_words'].keys())[:5],
                "main_topics": ["análise textual básica"]
            }
        
        # Usar modelo tradicional treinado
        try:
            clean_text = self.preprocessor.clean_text(text)
            X = self.traditional_classifier['vectorizer'].transform([clean_text])
            prediction = self.traditional_classifier['classifier'].predict(X)[0]
            proba = self.traditional_classifier['classifier'].predict_proba(X)[0]
            
            confidence = max(proba)
            
            return {
                "category": prediction,
                "confidence": float(confidence),
                "reasoning": "Classificação usando modelo Naive Bayes treinado",
                "keywords": self._extract_important_features(text),
                "main_topics": ["classificação automática"]
            }
            
        except Exception as e:
            logger.error(f"Erro no modelo tradicional: {str(e)}")
            return {
                "category": "unproductive",
                "confidence": 0.5,
                "reasoning": "Classificação padrão devido a erro no processamento",
                "keywords": [],
                "main_topics": ["erro de processamento"]
            }
    
    def _extract_important_features(self, text: str) -> List[str]:
        """Extrai palavras importantes do texto"""
        features = self.preprocessor.extract_features(text)
        return list(features['top_words'].keys())[:5]
    
    async def generate_response(self, text: str, category: str, tone: str = "professional") -> Dict[str, Any]:
        """Gera resposta automática usando GPT"""
        
        tone_instructions = {
            "professional": "Mantenha um tom profissional e respeitoso",
            "friendly": "Use um tom amigável e caloroso",
            "formal": "Seja formal e protocolar",
            "casual": "Use linguagem casual e descontraída"
        }
        
        if category == "productive":
            base_prompt = f"""
            Gere uma resposta adequada para este email de trabalho/negócios.
            {tone_instructions.get(tone, tone_instructions['professional'])}.
            
            Email original:
            "{text[:1000]}"
            
            Gere uma resposta que:
            - Reconheça o recebimento da mensagem
            - Seja apropriada ao contexto
            - Indique próximos passos se necessário
            - {tone_instructions.get(tone, "")}
            """
        else:
            base_prompt = f"""
            Gere uma resposta educada mas firme para declinar esta oferta/spam.
            {tone_instructions.get(tone, tone_instructions['professional'])}.
            
            Email recebido:
            "{text[:1000]}"
            
            Gere uma resposta que:
            - Seja educada mas clara sobre não ter interesse
            - Seja breve e direta
            - {tone_instructions.get(tone, "")}
            """
        
        prompt = f"""
        {base_prompt}
        
        Responda em formato JSON:
        {{
            "response": "texto da resposta",
            "alternatives": ["alternativa 1", "alternativa 2"]
        }}
        """
        
        try:
            if settings.OPENAI_API_KEY:
                response = await self.openai_client.chat.completions.create(
                    model=settings.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings.MAX_TOKENS,
                    temperature=0.4
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
            else:
                return self._generate_fallback_response(category, tone)
                
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            return self._generate_fallback_response(category, tone)
    
    def _generate_fallback_response(self, category: str, tone: str) -> Dict[str, Any]:
        """Gera resposta de fallback sem GPT"""
        
        if category == "productive":
            responses = {
                "professional": [
                    "Obrigado pela mensagem. Recebi as informações e vou analisar o conteúdo. Retornarei em breve com minha resposta.",
                    "Agradecemos pelo contato. Sua solicitação foi recebida e será processada. Aguarde nosso retorno.",
                    "Mensagem recebida. Vou revisar os detalhes e responder até o final do dia útil."
                ],
                "friendly": [
                    "Oi! Obrigado pela mensagem. Vou dar uma olhada e te respondo logo!",
                    "Olá! Recebi seu email e vou analisar. Falo com você em breve!",
                    "Oi! Sua mensagem chegou certinho. Vou verificar e te dou um retorno rapidinho."
                ],
                "formal": [
                    "Prezado Senhor(a), acusamos o recebimento de sua mensagem e informamos que será devidamente analisada.",
                    "Cordiais saudações. Confirmamos o recebimento de sua correspondência eletrônica.",
                    "Respeitosamente, informamos que sua mensagem foi recebida e será processada conforme nossos procedimentos."
                ],
                "casual": [
                    "Oi! Recebi sua mensagem. Vou ver isso e te falo!",
                    "Valeu pela mensagem! Vou checar e te respondo.",
                    "Opa! Sua mensagem chegou. Vou dar uma olhada e te retorno!"
                ]
            }
        else:
            responses = {
                "professional": [
                    "Obrigado por sua mensagem. No momento, não temos interesse na oferta apresentada.",
                    "Agradecemos o contato, mas no momento não é do nosso interesse.",
                    "Recebemos sua mensagem. Não temos necessidade deste tipo de serviço/produto atualmente."
                ],
                "friendly": [
                    "Oi! Obrigado pela mensagem, mas não tenho interesse no momento.",
                    "Olá! Agradecemos a oferta, mas não é para nós desta vez.",
                    "Oi! Valeu pelo contato, mas vou passar dessa oportunidade."
                ],
                "formal": [
                    "Prezado Senhor(a), agradecemos sua proposta, porém não atende às nossas necessidades atuais.",
                    "Cordiais saudações. Não possuímos interesse na proposta apresentada.",
                    "Respeitosamente, informamos que a oferta não se adequa ao nosso perfil empresarial."
                ],
                "casual": [
                    "Oi! Valeu pela mensagem, mas não tô interessado.",
                    "Opa! Obrigado, mas não é pra mim.",
                    "Olá! Não preciso disso agora, mas obrigado!"
                ]
            }
        
        tone_responses = responses.get(tone, responses["professional"])
        main_response = tone_responses[0]
        alternatives = tone_responses[1:3]
        
        return {
            "response": main_response,
            "alternatives": alternatives
        }

# ================================
# CACHE E PERFORMANCE
# ================================

class CacheManager:
    """Gerenciador de cache para otimizar performance"""
    
    def __init__(self):
        self.cache_dir = settings.CACHE_DIR
    
    def generate_cache_key(self, content: str) -> str:
        """Gera chave única para o conteúdo"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Recupera resultado do cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Verifica se cache não expirou (24 horas)
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if (datetime.now() - cached_time).seconds < 86400:
                    return cached_data['result']
                    
            except Exception as e:
                logger.warning(f"Erro ao ler cache: {str(e)}")
        
        return None
    
    def save_to_cache(self, cache_key: str, result: Dict):
        """Salva resultado no cache"""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {str(e)}")

# ================================
# APLICAÇÃO FASTAPI
# ================================

app = FastAPI(
    title="Sistema de Classificação de Emails",
    description="API para classificação inteligente de emails e geração de respostas automáticas",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
classifier = EmailClassifier()
file_processor = FileProcessor()
cache_manager = CacheManager()

# ================================
# ENDPOINTS DA API
# ================================

@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Sistema de Classificação de Emails",
        "version": "1.0.0",
        "status": "ativo",
        "endpoints": {
            "classify_text": "/classify/text",
            "classify_file": "/classify/file",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica saúde da aplicação"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "cache_dir_exists": settings.CACHE_DIR.exists(),
        "models_dir_exists": settings.MODELS_DIR.exists()
    }

@app.post("/classify/text", response_model=EmailProcessingResult)
async def classify_text(email_data: EmailInput):
    """Classifica email a partir de texto"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Gerar chave de cache
        cache_key = cache_manager.generate_cache_key(
            f"{email_data.content}_{email_data.tone}"
        )
        
        # Verificar cache
        cached_result = cache_manager.get_cached_result(cache_key)
        if cached_result:
            logger.info("Resultado obtido do cache")
            return cached_result
        
        # Processar email
        content = email_data.content
        
        # Extrair características do texto
        features = classifier.preprocessor.extract_features(content)
        
        # Classificar com GPT
        gpt_result = await classifier.classify_with_gpt(content)
        
        # Gerar resposta
        response_data = await classifier.generate_response(
            content, 
            gpt_result['category'], 
            email_data.tone
        )
        
        # Montar resultado
        processing_time = asyncio.get_event_loop().time() - start_time
        content_hash = cache_manager.generate_cache_key(content)
        
        result = EmailProcessingResult(
            classification=ClassificationResult(
                category=EmailCategory(gpt_result['category']),
                confidence=gpt_result['confidence'],
                reasoning=gpt_result['reasoning'],
                keywords=gpt_result.get('keywords', []),
                sentiment=features['sentiment'],
                entities=features['entities']
            ),
            suggested_response=ResponseSuggestion(
                text=response_data['response'],
                tone=email_data.tone,
                alternatives=response_data.get('alternatives', [])
            ),
            processing_time=processing_time,
            timestamp=datetime.now(),
            content_hash=content_hash
        )
        
        # Salvar no cache
        cache_manager.save_to_cache(cache_key, result.dict())
        
        logger.info(f"Email classificado como {gpt_result['category']} com {gpt_result['confidence']:.2f} de confiança")
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao processar email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/classify/file", response_model=EmailProcessingResult)
async def classify_file(
    file: UploadFile = File(...),
    tone: ResponseTone = Form(ResponseTone.PROFESSIONAL)
):
    """Classifica email a partir de arquivo"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Validar arquivo
        if not file_processor.validate_file(file):
            raise HTTPException(
                status_code=400, 
                detail="Arquivo inválido. Verifique o tipo e tamanho."
            )
        
        # Extrair texto do arquivo
        content = await file_processor.extract_text_from_file(file)
        
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="Não foi possível extrair texto do arquivo"
            )
        
        # Processar como texto
        email_input = EmailInput(content=content, tone=tone)
        return await classify_text(email_input)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar arquivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

@app.post("/generate_response")
async def generate_custom_response(
    content: str = Form(...),
    category: EmailCategory = Form(...),
    tone: ResponseTone = Form(ResponseTone.PROFESSIONAL)
):
    """Gera resposta customizada para um email específico"""
    
    try:
        response_data = await classifier.generate_response(content, category.value, tone.value)
        
        return {
            "suggested_response": ResponseSuggestion(
                text=response_data['response'],
                tone=tone,
                alternatives=response_data.get('alternatives', [])
            ),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Retorna estatísticas do sistema"""
    
    try:
        # Contar arquivos de cache
        cache_files = len(list(settings.CACHE_DIR.glob("*.json")))
        
        # Informações sobre modelos
        model_files = len(list(settings.MODELS_DIR.glob("*")))
        
        return {
            "cache_entries": cache_files,
            "model_files": model_files,
            "openai_enabled": bool(settings.OPENAI_API_KEY),
            "supported_formats": list(settings.ALLOWED_EXTENSIONS),
            "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
            "uptime": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@app.delete("/cache/clear")
async def clear_cache():
    """Limpa o cache do sistema"""
    
    try:
        cache_files = list(settings.CACHE_DIR.glob("*.json"))
        removed_count = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Erro ao remover {cache_file}: {str(e)}")
        
        return {
            "message": f"Cache limpo. {removed_count} arquivos removidos.",
            "removed_files": removed_count
        }
        
    except Exception as e:
        logger.error(f"Erro ao limpar cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao limpar cache")

# ================================
# BATCH PROCESSING
# ================================

@app.post("/classify/batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    tone: ResponseTone = Form(ResponseTone.PROFESSIONAL)
):
    """Processa múltiplos arquivos em lote"""
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Máximo de 10 arquivos por lote"
        )
    
    results = []
    errors = []
    
    for file in files:
        try:
            # Validar arquivo
            if not file_processor.validate_file(file):
                errors.append({
                    "filename": file.filename,
                    "error": "Arquivo inválido"
                })
                continue
            
            # Processar arquivo
            content = await file_processor.extract_text_from_file(file)
            email_input = EmailInput(content=content, tone=tone)
            
            result = await classify_text(email_input)
            results.append({
                "filename": file.filename,
                "result": result
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "processed_files": len(results),
        "errors": len(errors),
        "results": results,
        "errors_detail": errors,
        "timestamp": datetime.now()
    }

# ================================
# WEBSOCKETS PARA TEMPO REAL
# ================================

from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    """Gerencia conexões WebSocket"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove conexões mortas
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para processamento em tempo real"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Recebe dados do cliente
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Processa email
            email_input = EmailInput(
                content=request_data['content'],
                tone=request_data.get('tone', 'professional')
            )
            
            # Envia status de processamento
            await manager.send_personal_message(
                json.dumps({
                    "status": "processing",
                    "message": "Classificando email..."
                }), 
                websocket
            )
            
            # Classifica email
            result = await classify_text(email_input)
            
            # Envia resultado
            await manager.send_personal_message(
                json.dumps({
                    "status": "complete",
                    "result": result.dict()
                }), 
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_personal_message(
            json.dumps({
                "status": "error",
                "message": str(e)
            }), 
            websocket
        )

# ================================
# MIDDLEWARE E LOGGING
# ================================

import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware para logging e tempo de resposta"""
    start_time = time.time()
    
    # Log da requisição
    logger.info(f"Requisição: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Calcular tempo de processamento
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log da resposta
    logger.info(f"Resposta: {response.status_code} - {process_time:.3f}s")
    
    return response

# ================================
# SCRIPTS DE INICIALIZAÇÃO
# ================================

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação"""
    logger.info("🚀 Iniciando Sistema de Classificação de Emails")
    
    # Verificar configurações
    if not settings.OPENAI_API_KEY:
        logger.warning("⚠️  OpenAI API Key não configurada - usando fallback")
    
    # Criar diretórios necessários
    settings.CACHE_DIR.mkdir(exist_ok=True)
    settings.MODELS_DIR.mkdir(exist_ok=True)
    
    # Baixar recursos do NLTK se necessário
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("📦 Baixando recursos do NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('rslp', quiet=True)
    
    logger.info("✅ Sistema inicializado com sucesso!")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpeza na finalização da aplicação"""
    logger.info("🛑 Finalizando aplicação...")
    
    # Fechar conexões do OpenAI
    if hasattr(classifier, 'openai_client'):
        try:
            await classifier.openai_client.close()
        except:
            pass
    
    logger.info("✅ Aplicação finalizada")

# ================================
# CONFIGURAÇÃO PARA PRODUÇÃO
# ================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuração de desenvolvimento
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

