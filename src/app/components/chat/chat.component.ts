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
  
  // 👇 NUEVA VARIABLE
  useReranker = false; 
  showGraph = false;

  // Nuevas variables
  usePowerfulModel: boolean = false; // Controla el modelo (Rápido vs Avanzado)
  showingFullGraph: boolean = false; // Controla si vemos el Top-5 o la base de datos ampliada

  openGraphModal() {
    this.showGraph = true;
    // Esperamos 50ms para que Angular renderice el modal en el DOM
    setTimeout(() => {
      this.renderGraph();
    }, 50);
    
  }
  toggleFullGraph() {
      this.showingFullGraph = !this.showingFullGraph;
      
      // Aquí tienes que decidir cómo consigues los nodos extra. 
      // Opción A: Haces una llamada rápida al backend para pedir el Top-50.
      // Opción B: Si el backend ya te devolvió más artículos de los que muestras, usas esos.
      
      // Una vez tengas los datos nuevos, vuelves a pintar el grafo:
      this.renderGraph(); 
    }

  // 3. La función que dibuja la magia
  renderGraph() {
    const container = document.getElementById('mynetwork');
    if (!container) return;

    // Nodo central (La pregunta)
    const nodesArray: any[] = [
      { 
        id: 0, 
        label: 'Tu Búsqueda\n(Query)', 
        color: '#4f46e5', 
        font: { color: 'white', size: 16, multi: true }, 
        shape: 'circle',
        shadow: true
      }
    ];

    // Nodos de los papers
    if (this.currentResponse?.articles) {
      this.currentResponse.articles.forEach((art, index) => {
        nodesArray.push({
          id: index + 1,
          label: art.title.substring(0, 25) + '...', // Acortamos el título para que quepa
          title: art.title, // Esto hace que al pasar el ratón se vea el título completo
          value: art.relevanceScore, // El tamaño del nodo dependerá de su % de match
          color: '#c7d2fe',
          shape: 'dot',
          shadow: true
        });
      });
    }

    // Conexiones (Aristas)
    const edgesArray = nodesArray.filter(n => n.id !== 0).map(n => ({
      from: 0,
      to: n.id,
      value: n.value, // Grosor de la línea según la relevancia
      color: { color: '#818cf8', opacity: 0.6 }
    }));

    const data = { nodes: nodesArray, edges: edgesArray };
    const options = {
      nodes: {
        scaling: { min: 10, max: 35 } // Tamaños mínimo y máximo de las burbujas
      },
      physics: {
        stabilization: true,
        barnesHut: { springLength: 150 } // Distancia de los papers al centro
      }
    };

    // Renderizamos el grafo
    new Network(container, data, options);
  }

  constructor(
    private chatService: ChatService,
    public lang: LanguageService,
    private cdr: ChangeDetectorRef,
    private router: Router,
    public theme: ThemeService,
    private sanitizer: DomSanitizer 
  ) {}

  goBack() {
    this.router.navigate(['/']);
  }


  onSendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    const savedQuery = this.userInput;
    this.lastQuery = savedQuery;
    this.messages.push({ 
      text: savedQuery, 
      isUser: true 
    });
    this.userInput = ''; 
    this.isLoading = true; 
    

    // 👇 AÑADIMOS this.useReranker COMO 4º PARÁMETRO
    this.chatService.getQueryResponse(savedQuery, 5, this.isEvalMode, this.useReranker, this.usePowerfulModel).subscribe({
    
      next: (response: RagResponse) => {
        try {
          // AQUÍ NACE botText
          let botText = response.answer || "";
          botText = botText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

          this.processArticleScores(response.articles);
          if (response.enn_articles) {
            this.processArticleScores(response.enn_articles);
          }

          response.articles.forEach(article => {
            const safeTitle = article.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`${safeTitle}[^—\\-]*[—\\-]\\s*(.*?)(?:\\n|$)`, 'i');
            const match = botText.match(regex);
            if (match) article.keyDifference = match[1].trim();
          });

          this.currentResponse = response;
          
          // 👇 AQUÍ LA USAMOS Y LA GUARDAMOS EN LOS MENSAJES (Justo antes de cerrar el try)
          this.messages.push({ 
            text: this.sanitizer.bypassSecurityTrustHtml(botText), 
            isUser: false 
          });

        } finally {
          this.isLoading = false; 
          this.cdr.detectChanges(); 
        }
      },
      // ... (bloque error)
      error: (err) => {
        this.messages.push({ text: "Error de conexión.", isUser: false });
        this.isLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  private processArticleScores(articles: Article[]) {
    articles.forEach(art => {
      let rawScore = art.score || art.relevanceScore || 0;
      art.relevanceScore = rawScore > 1 ? rawScore : rawScore * 100;
    });
  }

  viewSummary(article: Article) {
    if (!article.abstract || article.summary) return;
    article.loadingSummary = true;
    this.chatService.getSummary(article.abstract, this.lastQuery, this.currentResponse?.language || 'en')
      .subscribe({
        next: (res) => {
          article.summary = res.summary.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
          article.loadingSummary = false;
          this.cdr.detectChanges();
        }
      });
  }

  toggleAbstract(article: Article) {
    article.showAbstract = !article.showAbstract;
  }
}