import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { RagResponse, Article } from '../../models/rag-response.model';
import { ThemeService } from '../../services/theme.service';
import { LanguageService } from '../../../services/language.service';
import { Router } from '@angular/router';


@Component({
  selector: 'app-chat',
  standalone: true, // <--- ESTO ES LO QUE ARREGLA EL ERROR NG2012
  imports: [CommonModule, FormsModule], // <--- Y ESTO TAMBIÉN ES OBLIGATORIO
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.css'] // (o .scss, según lo que uses)
})
export class ChatComponent {
  userInput = '';
  lastQuery = ''; 
  messages: { text: string, isUser: boolean }[] = [];
  currentResponse?: RagResponse;
  isLoading = false;
  isEvalMode = false; 

  constructor(
    private chatService: ChatService,
    public lang: LanguageService,
    private cdr: ChangeDetectorRef,
    private router: Router,
    public theme: ThemeService 
  ) {}

  goBack() {
    this.router.navigate(['/']);
  }

  onSendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    const savedQuery = this.userInput;
    this.lastQuery = savedQuery;
    this.messages.push({ text: savedQuery, isUser: true });
    this.userInput = ''; 
    this.isLoading = true; 

    // Asegúrate de que tu ChatService acepte (query, k, evalMode)
    this.chatService.getQueryResponse(savedQuery, 5, this.isEvalMode).subscribe({
      next: (response: RagResponse) => {
        try {
          let botText = response.answer || "";
          botText = botText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

          // 1. Procesar scores de ambas listas
          this.processArticleScores(response.articles);
          if (response.enn_articles) {
            this.processArticleScores(response.enn_articles);
          }

          // 2. Extraer diferencias clave del texto del bot (solo para ANN)
          response.articles.forEach(article => {
            const safeTitle = article.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`${safeTitle}[^—\\-]*[—\\-]\\s*(.*?)(?:\\n|$)`, 'i');
            const match = botText.match(regex);
            if (match) article.keyDifference = match[1].trim();
          });

          this.currentResponse = response;
          this.messages.push({ text: botText, isUser: false });
        } finally {
          this.isLoading = false; 
          this.cdr.detectChanges(); 
        }
      },
      error: (err) => {
        this.messages.push({ text: "Error de conexión.", isUser: false });
        this.isLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  // Método privado corregido con el tipo Article[]
  private processArticleScores(articles: Article[]) {
    articles.forEach(art => {
      let rawScore = art.score || art.relevanceScore || 0;
      art.relevanceScore = rawScore > 1 ? rawScore : rawScore * 100;
    });
  }

  // Métodos auxiliares se mantienen igual...
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