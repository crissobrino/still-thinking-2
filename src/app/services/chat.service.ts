import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RagResponse } from '../models/rag-response.model';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  getQueryResponse(query: string, k: number = 5, evalMode: boolean = false, useReranker: boolean = false, usePowerfulModel: boolean = false) {
    const body = {
      query: query,
      k: k,
      eval_mode: evalMode,
      use_reranker: useReranker,
      use_powerful_model: usePowerfulModel
    };

    return this.http.post<RagResponse>(`${this.baseUrl}/search`, body);
  }

  getSummary(abstract: string, query: string, lang: string): Observable<{summary: string}> {
    return this.http.post<{summary: string}>(`${this.baseUrl}/summarize`, {
      abstract,
      query,
      language: lang === 'es' ? 'Spanish' : 'English'
    });
  }
}