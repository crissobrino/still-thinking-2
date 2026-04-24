import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { RagResponse } from '../../models/rag-response.model';

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

  constructor(
    private chatService: ChatService,
    private cdr: ChangeDetectorRef 
  ) {}

  onSendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    this.lastQuery = this.userInput; // <--- GUARDAMOS LA QUERY
    const savedQuery = this.userInput;
    this.messages.push({ text: savedQuery, isUser: true });
    this.userInput = ''; 
    this.isLoading = true; 

    this.chatService.getQueryResponse(savedQuery).subscribe({
      next: (response) => {
        try {
          let botText = response.answer || "";
          botText = botText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

          // Procesamiento de artículos (Score y Diferencias)
          if (response.articles) {
            response.articles.forEach(article => {
              let rawScore = article.score || article.relevanceScore || 0;
              article.relevanceScore = rawScore > 1 ? rawScore : rawScore * 100;

              const safeTitle = article.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
              const regex = new RegExp(`${safeTitle}[^—\\-]*[—\\-]\\s*(.*?)(?:\\n|$)`, 'i');
              const match = botText.match(regex);
              if (match) article.keyDifference = match[1].trim();
            });
          }

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

  viewSummary(article: any) {
    console.log("Botón clicado para el artículo:", article.title);
    
    if (!article.abstract) {
      console.error("ERROR: El artículo no tiene 'abstract'. Revisa el backend.");
      return;
    }

    if (article.summary) return;

    article.loadingSummary = true;
    console.log("Enviando a resumir:", { q: this.lastQuery, abs: article.abstract.substring(0, 50) + "..." });
    
    this.chatService.getSummary(article.abstract, this.lastQuery, this.currentResponse?.language || 'en')
      .subscribe({
        next: (res) => {
          console.log("Resumen recibido:", res.summary);
          article.summary = res.summary;
          article.loadingSummary = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error("Error en la petición de resumen:", err);
          article.loadingSummary = false;
          this.cdr.detectChanges();
        }
      });
  }
}