import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RagResponse } from '../models/rag-response.model';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  // Definimos la base para que sea más fácil añadir endpoints
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

 

  // Añadimos useReranker con un valor por defecto false
  getQueryResponse(query: string, k: number = 5, evalMode: boolean = false, useReranker: boolean = false, usePowerfulModel: boolean = false) {
    
    // Lo incluimos en el cuerpo de la petición que va al backend de Python
    const body = {
      query: query,
      k: k,
      eval_mode: evalMode,
      use_reranker: useReranker, // <-- AQUÍ SE ENVÍA AL MAIN.PY
      usePowerfulModel: usePowerfulModel
    };

    // Tu llamada HTTP normal (la URL dependerá de cómo la tengas puesta)
    return this.http.post<RagResponse>(`${this.baseUrl}/search`, body);
  }


  getSummary(abstract: string, query: string, lang: string): Observable<{summary: string}> {
    // Apunta a http://localhost:8000/summarize
    return this.http.post<{summary: string}>(`${this.baseUrl}/summarize`, {
      abstract,
      query,
      language: lang === 'es' ? 'Spanish' : 'English'
    });
  }
}