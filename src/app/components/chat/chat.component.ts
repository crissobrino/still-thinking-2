import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { RagResponse, Article } from '../../models/rag-response.model';
import { ThemeService } from '../../services/theme.service';
import { LanguageService } from '../../../services/language.service';
import { Router } from '@angular/router';
import { Network } from 'vis-network';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Component({
  selector: 'app-chat',
  standalone: true, 
  imports: [CommonModule, FormsModule], 
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.css'] 
})
export class ChatComponent {
  userInput = '';
  lastQuery = ''; 
  messages: { text: SafeHtml | string, isUser: boolean }[] = [];
  currentResponse?: RagResponse;
  isLoading = false;
  isEvalMode = false; 
  useReranker = false; 
  showGraph = false;
  usePowerfulModel: boolean = false; 
  showingFullGraph: boolean = false; 

  constructor(
    private chatService: ChatService,
    public lang: LanguageService,
    private cdr: ChangeDetectorRef,
    private router: Router,
    public theme: ThemeService,
    private sanitizer: DomSanitizer 
  ) {}

  // --- LÓGICA DEL GRAFO ---

  openGraphModal() {
    this.showGraph = true;
    // Pequeño delay para asegurar que el div 'mynetwork' existe en el DOM
    setTimeout(() => this.renderGraph(), 50);
  }

  toggleFullGraph() {
    this.showingFullGraph = !this.showingFullGraph;
    this.renderGraph(); 
  }

  renderGraph() {
    const container = document.getElementById('mynetwork');
    if (!container || !this.currentResponse) return;

    // 1. Selección de datos: ¿Pintamos los 5 del chat o los 30 del mapa?
    const articlesToRender = this.showingFullGraph 
      ? (this.currentResponse.extended_articles || this.currentResponse.articles)
      : this.currentResponse.articles;

    // 2. Nodo central
    const nodesArray: any[] = [{ 
      id: 0, 
      label: 'Tu Búsqueda', 
      color: '#4f46e5', 
      font: { color: 'white', size: 16, bold: true }, 
      shape: 'circle',
      shadow: true
    }];

    // 3. Crear nodos de artículos
    articlesToRender.forEach((art, index) => {
      const isExtra = this.showingFullGraph && index >= 5; // Diferenciamos visualmente los extras
      nodesArray.push({
        id: index + 1,
        label: art.title.substring(0, 25) + '...',
        title: `<b>${art.title}</b><br>${art.authors} (${art.year})`, 
        value: art.relevanceScore || 10, 
        color: isExtra ? '#e2e8f0' : '#c7d2fe', // Gris para extras, azul para principales
        shape: 'dot',
        shadow: true
      });
    });

    // 4. Crear conexiones
    const edgesArray = nodesArray.filter(n => n.id !== 0).map(n => ({
      from: 0,
      to: n.id,
      value: (n.value / 10),
      color: { color: '#818cf8', opacity: 0.4 }
    }));

    const options = {
      nodes: { scaling: { min: 10, max: 35 } },
      physics: {
        stabilization: true,
        barnesHut: { 
          springLength: this.showingFullGraph ? 250 : 150, // Más espacio si hay muchos nodos
          gravitationalConstant: -2000
        }
      },
      interaction: { hover: true }
    };

    new Network(container, { nodes: nodesArray, edges: edgesArray }, options);
  }

  // --- LÓGICA DE MENSAJES ---

  onSendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    const savedQuery = this.userInput;
    this.lastQuery = savedQuery;
    this.messages.push({ text: savedQuery, isUser: true });
    
    this.userInput = ''; 
    this.isLoading = true; 
    
    this.chatService.getQueryResponse(savedQuery, 5, this.isEvalMode, this.useReranker, this.usePowerfulModel)
      .subscribe({
        next: (response: RagResponse) => {
          try {
            let botText = response.answer || "";
            // Limpiamos el rastro del "pensamiento" de modelos tipo DeepSeek/Qwen
            botText = botText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

            // Procesamos los scores de todas las listas disponibles
            this.processArticleScores(response.articles);
            if (response.extended_articles) this.processArticleScores(response.extended_articles);
            if (response.enn_articles) this.processArticleScores(response.enn_articles);

            // Lógica de "Key Difference" extraída del texto del bot
            response.articles.forEach(article => {
              const safeTitle = article.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
              const regex = new RegExp(`${safeTitle}[^—\\-]*[—\\-]\\s*(.*?)(?:\\n|$)`, 'i');
              const match = botText.match(regex);
              if (match) article.keyDifference = match[1].trim();
            });

            this.currentResponse = response;
            this.messages.push({ 
              text: this.sanitizer.bypassSecurityTrustHtml(botText), 
              isUser: false 
            });

          } finally {
            this.isLoading = false; 
            this.cdr.detectChanges(); 
          }
        },
        error: (err) => {
          console.error("Error en búsqueda:", err);
          this.messages.push({ text: "Lo siento, hubo un error en la conexión.", isUser: false });
          this.isLoading = false;
          this.cdr.detectChanges();
        }
      });
  }

  private processArticleScores(articles: Article[]) {
    if (!articles) return;
    articles.forEach(art => {
      let rawScore = art.score || art.relevanceScore || 0;
      // Normalizamos a base 100 para el tamaño de las burbujas del grafo
      art.relevanceScore = rawScore > 1 ? rawScore : rawScore * 100;
    });
  }

  // --- UTILIDADES ---

  viewSummary(article: Article) {
    if (!article.abstract || article.summary) return;
    article.loadingSummary = true;
    this.chatService.getSummary(article.abstract, this.lastQuery, this.currentResponse?.language || 'en')
      .subscribe({
        next: (res) => {
          article.summary = res.summary.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
          article.loadingSummary = false;
          this.cdr.detectChanges();
        },
        error: () => article.loadingSummary = false
      });
  }

  toggleAbstract(article: Article) {
    article.showAbstract = !article.showAbstract;
  }

  goBack() {
    this.router.navigate(['/']);
  }
}