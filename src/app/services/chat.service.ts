import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { RagResponse } from '../models/rag-response.model';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  sendQuery(query: string): Observable<RagResponse> {
    // Simulamos una respuesta del backend (Mock data)
    const mockResponse: RagResponse = {
      answer: "He encontrado que tu investigación sobre redes neuronales se diferencia de los artículos de ACL en que ellos no aplican federated learning. [1].",
      articles: [
        {
          title: "Federated Topic Modeling for Health Data",
          authors: "Smith et al.",
          year: 2023,
          url: "http://example.com",
          relevanceScore: 0.95,
          keyDifference: "Usa modelos Bayesianos en lugar de redes neuronales puras."
        }
      ],
      language: "es"
    };

    // 'of' crea un observable y 'delay' simula el tiempo que tarda la IA (Response Time)
    return of(mockResponse).pipe(delay(10));
  }
}