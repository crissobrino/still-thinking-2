import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { RagResponse } from '../models/rag-response.model';
import mockData from '../../assets/data/mock-response.json';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  constructor() {}

  sendQuery(userQuery: string): Observable<RagResponse> {

    console.log("Consulta enviada:", userQuery);

    return of(mockData as RagResponse).pipe(
      delay(3000)
    );

  }

}