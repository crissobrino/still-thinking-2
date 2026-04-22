import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RagResponse } from '../models/rag-response.model';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  // FastAPI default URL is 8000. Use /search as defined in your main.py
  private API_URL = 'http://localhost:8000/search';

  constructor(private http: HttpClient) {}

  // This replaces your mock logic
  getQueryResponse(query: string): Observable<RagResponse> {
    // We send { query: query } to match your SearchRequest BaseModel in Python
    return this.http.post<RagResponse>(this.API_URL, { query: query });
  }
}