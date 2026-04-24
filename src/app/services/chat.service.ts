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

  getQueryResponse(query: string): Observable<RagResponse> {
    // Apunta a http://localhost:8000/search
    return this.http.post<RagResponse>(`${this.baseUrl}/search`, { query: query });
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