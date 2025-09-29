
class EmailClassifier {
    constructor() {
        this.processedCount = 0;
        this.history = [];
        this.currentFiles = [];
        this.isProcessing = false;
        
        this.init();
    }

    /**
     * Inicializa o sistema e configura event listeners
     */
    init() {
        this.setupEventListeners();
        this.loadStoredData();
        this.updateProcessedCount();
    }

    /**
     * Configura todos os event listeners da aplicação
     */
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // File upload
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        dropZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        dropZone.addEventListener('drop', (e) => this.handleFileDrop(e));
        
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Text input
        const emailText = document.getElementById('emailText');
        emailText.addEventListener('input', (e) => this.handleTextInput(e));

        // Process button
        document.getElementById('processBtn').addEventListener('click', () => this.processEmail());

        // Response actions
        document.getElementById('copyResponseBtn').addEventListener('click', () => this.copyResponse());
        document.getElementById('editResponseBtn').addEventListener('click', () => this.openEditModal());

        // Tone selector
        document.getElementById('responseTone').addEventListener('change', (e) => this.updateResponseTone(e.target.value));

        // Modal controls
        document.getElementById('closeModal').addEventListener('click', () => this.closeEditModal());
        document.getElementById('cancelEdit').addEventListener('click', () => this.closeEditModal());
        document.getElementById('saveEdit').addEventListener('click', () => this.saveEditedResponse());

        // History
        document.getElementById('clearHistoryBtn').addEventListener('click', () => this.clearHistory());

        // Click outside modal to close
        document.getElementById('editModal').addEventListener('click', (e) => {
            if (e.target.id === 'editModal') {
                this.closeEditModal();
            }
        });
    }

    /**
     * Alterna entre as abas de input (arquivo/texto)
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.validateInput();
    }

    /**
     * Manipula eventos de drag over na zona de upload
     */
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('dropZone').classList.add('dragover');
    }

    /**
     * Manipula eventos de drag leave na zona de upload
     */
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('dropZone').classList.remove('dragover');
    }

    /**
     * Manipula o drop de arquivos na zona de upload
     */
    handleFileDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('dropZone').classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }

    /**
     * Manipula a seleção de arquivos através do input
     */
    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.processFiles(files);
    }

    /**
     * Processa arquivos selecionados ou arrastados
     */
    processFiles(files) {
        const validFiles = files.filter(file => this.validateFile(file));
        
        if (validFiles.length === 0) {
            this.showToast('Nenhum arquivo válido selecionado', 'warning');
            return;
        }

        validFiles.forEach(file => this.addFile(file));
        this.displayUploadedFiles();
        this.validateInput();
    }

    /**
     * Valida se o arquivo é permitido
     */
    validateFile(file) {
        const allowedTypes = ['text/plain', 'application/pdf'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!allowedTypes.includes(file.type)) {
            this.showToast(`Tipo de arquivo não suportado: ${file.name}`, 'error');
            return false;
        }

        if (file.size > maxSize) {
            this.showToast(`Arquivo muito grande: ${file.name}`, 'error');
            return false;
        }

        return true;
    }

    /**
     * Adiciona arquivo à lista de arquivos atuais
     */
    addFile(file) {
        const fileData = {
            id: Date.now() + Math.random(),
            file: file,
            name: file.name,
            size: this.formatFileSize(file.size),
            type: file.type
        };

        this.currentFiles.push(fileData);
    }

    /**
     * Exibe arquivos uploaded na interface
     */
    displayUploadedFiles() {
        const container = document.getElementById('uploadedFiles');
        
        if (this.currentFiles.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        container.innerHTML = this.currentFiles.map(fileData => `
            <div class="file-item">
                <div class="file-icon">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" fill="currentColor"/>
                    </svg>
                </div>
                <div class="file-info">
                    <div class="file-name">${fileData.name}</div>
                    <div class="file-size">${fileData.size}</div>
                </div>
                <button class="file-remove" onclick="emailClassifier.removeFile('${fileData.id}')">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" stroke-width="2"/>
                        <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2"/>
                    </svg>
                </button>
            </div>
        `).join('');
    }

    /**
     * Remove arquivo da lista
     */
    removeFile(fileId) {
        this.currentFiles = this.currentFiles.filter(file => file.id !== fileId);
        this.displayUploadedFiles();
        this.validateInput();
    }

    /**
     * Manipula input de texto e atualiza estatísticas
     */
    handleTextInput(e) {
        const text = e.target.value;
        const charCount = text.length;
        const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

        document.getElementById('charCount').textContent = `${charCount} caracteres`;
        document.getElementById('wordCount').textContent = `${wordCount} palavras`;

        this.validateInput();
    }

    /**
     * Valida se há input suficiente para processar
     */
    validateInput() {
        const activeTab = document.querySelector('.tab-button.active').dataset.tab;
        const processBtn = document.getElementById('processBtn');
        
        let hasInput = false;

        if (activeTab === 'file') {
            hasInput = this.currentFiles.length > 0;
        } else if (activeTab === 'text') {
            const text = document.getElementById('emailText').value.trim();
            hasInput = text.length > 10; // Mínimo de 10 caracteres
        }

        processBtn.disabled = !hasInput || this.isProcessing;
    }

    /**
     * Processa o email (simulação de IA)
     */
    async processEmail() {
        if (this.isProcessing) return;

        this.isProcessing = true;
        this.updateProcessButton(true);

        try {
            // Simula processamento com delay
            await this.delay(2000 + Math.random() * 2000);

            const result = await this.simulateAIProcessing();
            this.displayResults(result);
            this.addToHistory(result);
            this.incrementProcessedCount();
            
            this.showToast('Email classificado com sucesso!', 'success');
            
        } catch (error) {
            this.showToast('Erro ao processar email', 'error');
            console.error('Processing error:', error);
        } finally {
            this.isProcessing = false;
            this.updateProcessButton(false);
        }
    }

    /**
     * Simula processamento de IA
     */
    async simulateAIProcessing() {
        const activeTab = document.querySelector('.tab-button.active').dataset.tab;
        let content = '';

        if (activeTab === 'file') {
            // Simula leitura de arquivo
            content = `Conteúdo do arquivo: ${this.currentFiles[0].name}`;
        } else {
            content = document.getElementById('emailText').value;
        }

        // Simula análise de IA baseada em palavras-chave
        const productiveKeywords = [
            'reunião', 'projeto', 'prazo', 'entrega', 'relatório', 'apresentação',
            'cliente', 'proposta', 'orçamento', 'contrato', 'cronograma', 'agenda'
        ];

        const unproductiveKeywords = [
            'spam', 'promoção', 'desconto', 'oferta', 'grátis', 'ganhe',
            'loteria', 'prêmio', 'clique', 'urgente', 'oportunidade única'
        ];

        const contentLower = content.toLowerCase();
        const productiveScore = productiveKeywords.filter(word => contentLower.includes(word)).length;
        const unproductiveScore = unproductiveKeywords.filter(word => contentLower.includes(word)).length;

        const isProductive = productiveScore > unproductiveScore;
        const confidence = Math.min(95, 75 + Math.random() * 20);

        return {
            classification: isProductive ? 'productive' : 'unproductive',
            confidence: confidence,
            content: content,
            tags: this.generateAnalysisTags(isProductive, contentLower),
            suggestedResponse: this.generateResponse(isProductive, content),
            timestamp: new Date(),
            source: activeTab === 'file' ? this.currentFiles[0].name : 'Texto direto'
        };
    }

    /**
     * Gera tags de análise baseadas no conteúdo
     */
    generateAnalysisTags(isProductive, content) {
        if (isProductive) {
            const tags = ['Conteúdo relevante', 'Tom profissional'];
            
            if (content.includes('prazo') || content.includes('urgente')) {
                tags.push('Ação requerida');
            }
            if (content.includes('reunião') || content.includes('agenda')) {
                tags.push('Agendamento necessário');
            }
            
            return tags.map(tag => ({ text: tag, type: 'positive' }));
        } else {
            return [
                { text: 'Conteúdo promocional', type: 'negative' },
                { text: 'Possível spam', type: 'negative' },
                { text: 'Baixa relevância', type: 'neutral' }
            ];
        }
    }

    /**
     * Gera resposta sugerida baseada na classificação
     */
    generateResponse(isProductive, content) {
        if (isProductive) {
            const responses = [
                "Obrigado pela mensagem. Recebi as informações e vou analisar o conteúdo. Retornarei em breve com minha resposta.",
                "Agradecemos pelo contato. Sua solicitação foi recebida e será processada. Aguarde nosso retorno.",
                "Mensagem recebida. Vou revisar os detalhes e responder até o final do dia útil.",
                "Obrigado por compartilhar essas informações. Vou avaliar e dar continuidade aos próximos passos."
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        } else {
            const responses = [
                "Obrigado por sua mensagem. No momento, não temos interesse na oferta apresentada.",
                "Agradecemos o contato, mas no momento não é do nosso interesse.",
                "Recebemos sua mensagem. Não temos necessidade deste tipo de serviço/produto atualmente.",
                "Obrigado pela informação, mas não se adequa às nossas necessidades no momento."
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        }
    }

    /**
     * Exibe os resultados na interface
     */
    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const classificationIcon = document.getElementById('classificationIcon');
        const classificationType = document.getElementById('classificationType');
        const classificationDescription = document.getElementById('classificationDescription');
        const confidenceBadge = document.getElementById('confidenceBadge');
        const analysisTags = document.getElementById('analysisTags');
        const suggestedResponse = document.getElementById('suggestedResponse');

        // Atualiza classificação
        const isProductive = result.classification === 'productive';
        classificationIcon.className = `classification-icon ${result.classification}`;
        classificationType.textContent = isProductive ? 'Email Produtivo' : 'Email Improdutivo';
        classificationDescription.textContent = isProductive ? 
            'Este email contém informações relevantes para o trabalho' : 
            'Este email não contém informações relevantes para o trabalho';

        confidenceBadge.textContent = `${Math.round(result.confidence)}% confiança`;

        // Atualiza ícone
        if (isProductive) {
            classificationIcon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 12l2 2 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                </svg>
            `;
        } else {
            classificationIcon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                </svg>
            `;
        }

        // Atualiza tags de análise
        analysisTags.innerHTML = result.tags.map(tag => 
            `<span class="tag ${tag.type}">${tag.text}</span>`
        ).join('');

        // Atualiza resposta sugerida
        suggestedResponse.textContent = result.suggestedResponse;

        // Mostra seção de resultados
        resultsSection.classList.add('visible');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    /**
     * Atualiza o estado visual do botão de processar
     */
    updateProcessButton(isProcessing) {
        const processBtn = document.getElementById('processBtn');
        
        if (isProcessing) {
            processBtn.classList.add('processing');
            processBtn.disabled = true;
        } else {
            processBtn.classList.remove('processing');
            this.validateInput(); // Revalida para habilitar se necessário
        }
    }

    /**
     * Copia a resposta sugerida para o clipboard
     */
    async copyResponse() {
        const responseText = document.getElementById('suggestedResponse').textContent;
        
        try {
            await navigator.clipboard.writeText(responseText);
            this.showToast('Resposta copiada para a área de transferência!', 'success');
        } catch (error) {
            // Fallback para navegadores que não suportam clipboard API
            this.fallbackCopyToClipboard(responseText);
        }
    }

    /**
     * Fallback para copiar texto em navegadores antigos
     */
    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            this.showToast('Resposta copiada para a área de transferência!', 'success');
        } catch (error) {
            this.showToast('Não foi possível copiar o texto', 'error');
        } finally {
            document.body.removeChild(textArea);
        }
    }

    /**
     * Abre o modal de edição da resposta
     */
    openEditModal() {
        const currentResponse = document.getElementById('suggestedResponse').textContent;
        const editTextarea = document.getElementById('editResponseText');
        const modal = document.getElementById('editModal');

        editTextarea.value = currentResponse;
        modal.classList.add('active');

        // Focus no textarea após a animação
        setTimeout(() => {
            editTextarea.focus();
            editTextarea.setSelectionRange(editTextarea.value.length, editTextarea.value.length);
        }, 300);
    }

    /**
     * Fecha o modal de edição
     */
    closeEditModal() {
        document.getElementById('editModal').classList.remove('active');
    }

    /**
     * Salva a resposta editada
     */
    saveEditedResponse() {
        const editedText = document.getElementById('editResponseText').value.trim();
        
        if (!editedText) {
            this.showToast('A resposta não pode estar vazia', 'warning');
            return;
        }

        document.getElementById('suggestedResponse').textContent = editedText;
        this.closeEditModal();
        this.showToast('Resposta atualizada com sucesso!', 'success');
    }

    /**
     * Atualiza o tom da resposta
     */
    updateResponseTone(tone) {
        const currentResponse = document.getElementById('suggestedResponse').textContent;
        const newResponse = this.adjustResponseTone(currentResponse, tone);
        document.getElementById('suggestedResponse').textContent = newResponse;
        
        this.showToast(`Tom alterado para: ${this.getToneLabel(tone)}`, 'info');
    }

    /**
     * Ajusta o tom da resposta baseado na seleção
     */
    adjustResponseTone(response, tone) {
        const toneTemplates = {
            professional: {
                prefix: "Prezado(a),\n\n",
                suffix: "\n\nAtenciosamente,\n[Seu nome]"
            },
            friendly: {
                prefix: "Olá!\n\n",
                suffix: "\n\nUm abraço!"
            },
            formal: {
                prefix: "Prezado Senhor(a),\n\n",
                suffix: "\n\nRespeitosamente,\n[Seu nome]"
            },
            casual: {
                prefix: "Oi!\n\n",
                suffix: "\n\nValeu!"
            }
        };

        const template = toneTemplates[tone] || toneTemplates.professional;
        return template.prefix + response + template.suffix;
    }

    /**
     * Retorna o label do tom selecionado
     */
    getToneLabel(tone) {
        const labels = {
            professional: 'Profissional',
            friendly: 'Amigável',
            formal: 'Formal',
            casual: 'Casual'
        };
        return labels[tone] || 'Profissional';
    }

    /**
     * Adiciona resultado ao histórico
     */
    addToHistory(result) {
        const historyItem = {
            id: Date.now(),
            ...result,
            preview: result.content.substring(0, 100) + (result.content.length > 100 ? '...' : '')
        };

        this.history.unshift(historyItem); // Adiciona no início
        
        // Limita o histórico a 50 itens
        if (this.history.length > 50) {
            this.history = this.history.slice(0, 50);
        }

        this.updateHistoryDisplay();
        this.saveToStorage();
    }

    /**
     * Atualiza a exibição do histórico
     */
    updateHistoryDisplay() {
        const historyList = document.getElementById('historyList');
        
        if (this.history.length === 0) {
            historyList.innerHTML = `
                <div class="empty-history">
                    <p>Nenhum email processado ainda.</p>
                </div>
            `;
            return;
        }

        historyList.innerHTML = this.history.map(item => `
            <div class="history-item">
                <div class="history-status ${item.classification}">
                    ${item.classification === 'productive' ? 'P' : 'I'}
                </div>
                <div class="history-info">
                    <div class="history-filename">${item.source}</div>
                    <div class="history-time">${this.formatDate(item.timestamp)}</div>
                    <div class="history-preview">${item.preview}</div>
                </div>
            </div>
        `).join('');
    }

    /**
     * Limpa o histórico
     */
    clearHistory() {
        if (this.history.length === 0) {
            this.showToast('Histórico já está vazio', 'info');
            return;
        }

        if (confirm('Tem certeza que deseja limpar todo o histórico?')) {
            this.history = [];
            this.updateHistoryDisplay();
            this.saveToStorage();
            this.showToast('Histórico limpo com sucesso!', 'success');
        }
    }

    /**
     * Incrementa contador de emails processados
     */
    incrementProcessedCount() {
        this.processedCount++;
        this.updateProcessedCount();
        this.saveToStorage();
    }

    /**
     * Atualiza a exibição do contador
     */
    updateProcessedCount() {
        document.getElementById('processedCount').textContent = this.processedCount;
    }

    /**
     * Exibe notificação toast
     */
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = this.getToastIcon(type);
        
        toast.innerHTML = `
            <div class="toast-icon">${icon}</div>
            <div class="toast-message">${message}</div>
        `;

        document.getElementById('toastContainer').appendChild(toast);

        // Remove o toast após 3 segundos
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }

    /**
     * Retorna ícone para o tipo de toast
     */
    getToastIcon(type) {
        const icons = {
            success: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 12l2 2 4-4" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="12" r="10" stroke="#10b981" stroke-width="2"/>
            </svg>`,
            error: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <line x1="18" y1="6" x2="6" y2="18" stroke="#ef4444" stroke-width="2"/>
                <line x1="6" y1="6" x2="18" y2="18" stroke="#ef4444" stroke-width="2"/>
                <circle cx="12" cy="12" r="10" stroke="#ef4444" stroke-width="2"/>
            </svg>`,
            warning: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="#f59e0b" stroke-width="2"/>
                <line x1="12" y1="9" x2="12" y2="13" stroke="#f59e0b" stroke-width="2"/>
                <line x1="12" y1="17" x2="12.01" y2="17" stroke="#f59e0b" stroke-width="2"/>
            </svg>`,
            info: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" stroke="#3b82f6" stroke-width="2"/>
                <line x1="12" y1="16" x2="12" y2="12" stroke="#3b82f6" stroke-width="2"/>
                <line x1="12" y1="8" x2="12.01" y2="8" stroke="#3b82f6" stroke-width="2"/>
            </svg>`
        };
        return icons[type] || icons.info;
    }

    /**
     * Formata tamanho de arquivo
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Formata data para exibição
     */
    formatDate(date) {
        const now = new Date();
        const diff = now - date;
        const diffMinutes = Math.floor(diff / (1000 * 60));
        const diffHours = Math.floor(diff / (1000 * 60 * 60));
        const diffDays = Math.floor(diff / (1000 * 60 * 60 * 24));

        if (diffMinutes < 1) return 'Agora mesmo';
        if (diffMinutes < 60) return `${diffMinutes} min atrás`;
        if (diffHours < 24) return `${diffHours}h atrás`;
        if (diffDays < 7) return `${diffDays}d atrás`;
        
        return date.toLocaleDateString('pt-BR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    /**
     * Salva dados no localStorage
     */
    saveToStorage() {
        try {
            const data = {
                processedCount: this.processedCount,
                history: this.history.slice(0, 20) // Salva apenas os 20 mais recentes
            };
            localStorage.setItem('emailClassifierData', JSON.stringify(data));
        } catch (error) {
            console.warn('Não foi possível salvar no localStorage:', error);
        }
    }

    /**
     * Carrega dados do localStorage
     */
    loadStoredData() {
        try {
            const stored = localStorage.getItem('emailClassifierData');
            if (stored) {
                const data = JSON.parse(stored);
                this.processedCount = data.processedCount || 0;
                this.history = data.history || [];
                
                // Reconstroi objetos Date
                this.history.forEach(item => {
                    item.timestamp = new Date(item.timestamp);
                });
                
                this.updateHistoryDisplay();
            }
        } catch (error) {
            console.warn('Não foi possível carregar do localStorage:', error);
            this.processedCount = 0;
            this.history = [];
        }
    }

    /**
     * Função utilitária para delay
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Limpa formulário após processamento
     */
    resetForm() {
        // Limpa arquivos
        this.currentFiles = [];
        this.displayUploadedFiles();
        
        // Limpa texto
        document.getElementById('emailText').value = '';
        document.getElementById('charCount').textContent = '0 caracteres';
        document.getElementById('wordCount').textContent = '0 palavras';
        
        // Limpa input de arquivo
        document.getElementById('fileInput').value = '';
        
        // Revalida
        this.validateInput();
    }

    /**
     * Simula diferentes cenários de processamento para demonstração
     */
    simulateRandomScenario() {
        const scenarios = [
            {
                content: "Reunião de projeto agendada para segunda-feira às 14h. Favor confirmar presença e revisar documentos anexos.",
                classification: 'productive',
                confidence: 92
            },
            {
                content: "OFERTA IMPERDÍVEL! Clique agora e ganhe 50% de desconto em todos os produtos. Oferta por tempo limitado!",
                classification: 'unproductive',
                confidence: 89
            },
            {
                content: "Relatório mensal de vendas anexo. Solicitamos análise e feedback até sexta-feira.",
                classification: 'productive',
                confidence: 95
            }
        ];

        return scenarios[Math.floor(Math.random() * scenarios.length)];
    }
}

// Inicializa o sistema quando a página carregar
let emailClassifier;

document.addEventListener('DOMContentLoaded', () => {
    emailClassifier = new EmailClassifier();
});

// Adiciona animações de entrada
window.addEventListener('load', () => {
    // Anima elementos na tela
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
            }
        });
    }, observerOptions);

    // Observa elementos para animação
    document.querySelectorAll('.upload-card, .result-card, .history-section').forEach(el => {
        observer.observe(el);
    });
});

// Service Worker para cache (opcional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl + Enter para processar
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!document.getElementById('processBtn').disabled) {
            emailClassifier.processEmail();
        }
    }
    
    // Escape para fechar modal
    if (e.key === 'Escape') {
        emailClassifier.closeEditModal();
    }
    
    // Ctrl + C quando resposta está focada
    if ((e.ctrlKey || e.metaKey) && e.key === 'c' && 
        document.activeElement === document.getElementById('suggestedResponse')) {
        emailClassifier.copyResponse();
        e.preventDefault();
    }
});

// PWA Install prompt
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    
    // Mostra botão de instalação customizado (opcional)
    const installBtn = document.getElementById('installBtn');
    if (installBtn) {
        installBtn.style.display = 'block';
        installBtn.addEventListener('click', () => {
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then((choiceResult) => {
                if (choiceResult.outcome === 'accepted') {
                    console.log('PWA instalado');
                }
                deferredPrompt = null;
            });
        });
    }
});

// Analytics de uso (simulado)
function trackUsage(event, data = {}) {
    // Aqui você adicionaria integração com Google Analytics, Mixpanel, etc.
    console.log(`Analytics: ${event}`, data);
}

document.getElementById("processBtn").addEventListener("click", async () => {
    const emailText = document.getElementById("emailText").value;
    const tone = document.getElementById("responseTone").value;

    const response = await fetch("http://127.0.0.1:8000/classify/text", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            content: emailText,
            tone: tone
        })
    });

    const result = await response.json();
    console.log(result);

    // Exibe no HTML
    document.getElementById("classificationType").innerText =
        result.classification.category === "productive" ? "Email Produtivo" : "Email Improdutivo";

    document.getElementById("confidenceBadge").innerText =
        `${Math.round(result.classification.confidence * 100)}% confiança`;

    document.getElementById("suggestedResponse").innerText = result.suggested_response.text;
});
