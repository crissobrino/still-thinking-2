export interface Article {
  title: string;
  authors: string;
  year: number;
  url: string;
  relevanceScore: number; // Para la "Feature Adicional" de métricas
  keyDifference: string;  // Requisito funcional: diferencias con tu idea
}

export interface RagResponse {
  answer: string;
  articles: Article[];
  language: string;       // Requisito funcional: mismo idioma
}