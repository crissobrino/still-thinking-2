export interface Article {
  title: string;
  authors: string;
  year: number;
  similarity_score: number;
  snippet: string;
}

export interface RagResponse {
  answer: string;
  articles: Article[];
  language: string;
}