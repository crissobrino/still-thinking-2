import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { RagResponse } from '../../models/rag-response.model';

@Component({
  selector: 'app-chat',
  standalone: true, 
  imports: [
    CommonModule,
    FormsModule
  ],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.css']
})
export class ChatComponent {

  userInput = '';
  messages: { text: string, isUser: boolean }[] = [];
  currentResponse?: RagResponse;
  isLoading = false;

  // 1. Inyectamos ChangeDetectorRef aquí
  constructor(
    private chatService: ChatService,
    private cdr: ChangeDetectorRef 
  ) {}

  onSendMessage() {
    // 1. Validaciones iniciales
    if (!this.userInput.trim() || this.isLoading) return;

    // 2. Preparamos la UI para el envío
    const savedQuery = this.userInput;
    this.messages.push({ text: savedQuery, isUser: true });
    this.userInput = ''; 
    this.isLoading = true; 

    // 3. Llamada ÚNICA al servicio de backend
    this.chatService.getQueryResponse(savedQuery).subscribe({
      next: (response) => {
        try {
          // Limpiamos el <think>
          let botText = response.answer || "El servidor no devolvió una respuesta clara.";
          botText = botText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

          // Magia para procesar los artículos (Problema 1 y 4)
          if (response.articles && response.articles.length > 0) {
            response.articles.forEach(article => {
              // --- Problema 1: Ajuste del porcentaje ---
              let rawScore = article.score || article.relevanceScore || 0;
              // Si ya viene como 17.5, lo dejamos. Si viene como 0.175, lo multiplicamos
              article.relevanceScore = rawScore > 1 ? rawScore : rawScore * 100;

              // --- Problema 4: Extraer la descripción del LLM para la caja amarilla ---
              // Escapamos el título por si tiene caracteres raros
              const safeTitle = article.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
              // Buscamos el título, seguido de símbolos raros/guiones, y capturamos el texto
              const regex = new RegExp(`${safeTitle}[^—\\-]*[—\\-]\\s*(.*?)(?:\\n|$)`, 'i');
              const match = botText.match(regex);
              
              if (match && match[1]) {
                article.keyDifference = match[1].trim(); // Guardamos la descripción encontrada
              } else if (!article.keyDifference || article.keyDifference.includes("Analizado en la respuesta")) {
                article.keyDifference = "Ver detalles en la respuesta del asistente.";
              }
            });
          }

          this.currentResponse = response;
          this.messages.push({ text: botText, isUser: false });

        } catch (e) {
          console.error("Error procesando la respuesta:", e);
        } finally {
          this.isLoading = false; 
          this.cdr.detectChanges(); 
        }
      },
      error: (err) => {
        console.error('Error detectado en la petición:', err);
        this.messages.push({ 
          text: "Error de conexión con el servidor. Revisa si el backend está activo.", 
          isUser: false 
        });
        this.isLoading = false; 
        this.cdr.detectChanges(); // Obligamos a repintar también en caso de error
      }
    });
  }
}