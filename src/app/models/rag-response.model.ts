export interface Article {
  id?: string;
  title: string;
  authors: string;
  year?: number;
  abstract?: string;
  url?: string;
  relevanceScore?: number; // El que usas en el HTML
  score?: number;          // El que parece enviar el backend
  keyDifference?: string;
  showAbstract?: boolean;

  summary?: string;
  loadingSummary?: boolean;
}

export interface Metrics {
  ann_time: number;
  enn_time?: number;
  recall?: number;
}

export interface RagResponse {
  answer: string;
  articles: Article[];
  enn_articles?: Article[];
  metrics: Metrics;
  language: string;
}