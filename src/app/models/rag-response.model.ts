export interface Article {
  title: string;
  authors: string;
  year?: number;
  url?: string;
  relevanceScore?: number; // El que usas en el HTML
  score?: number;          // El que parece enviar el backend
  keyDifference?: string;
}

export interface RagResponse {
  answer: string;
  articles: Article[];
  language: string;
}