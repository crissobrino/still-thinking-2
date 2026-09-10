import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class LanguageService {
  currentLang = signal<'es' | 'en' | 'fr'>('es');

  private translations: any = {
    es: {
      title: 'Investigación AI',
      subtitle: 'Tu puente hacia la literatura científica',
      startBtn: 'Empezar Investigación',
      backBtn: 'Volver',
      langBtn: 'Idioma',
      placeholder: 'Pregunta sobre artículos...',
      send: 'Enviar',
      evalMode: 'Modo Evaluación',
      noArticles: 'No hay artículos cargados.',
      match: 'coincidencia',
      analyzing: '🤖 Analizando artículos y generando comparación...',
      iaSummary: 'Resumen IA',
      summaryReady: 'Resumen listo',
      viewAbstract: 'Ver abstract',
      hideInfo: 'Ocultar info',
      recallMetric: 'Métrica Recall',
      reRanker: 'Re-ranker',
      advancedModel: 'Modelo Avanzado',
      fastModel: 'Modelo Rápido',
      globalPerformance: 'Rendimiento Global',
      llmInference: 'Inferencia LLM',
      totalTime: 'Tiempo Total',
      visualMapBtn: 'VER MAPA VISUAL DE RELACIONES',
      relevanceMap: 'Mapa de Relevancia',
      seeTop5: 'VER SOLO TOP-5',
      seeExtended: 'VER RED EXTENDIDA',
      generatingGraph: 'Generando conexiones visuales...',
      connectionError: 'Lo siento, hubo un error en la conexión.',
      yourSearchNode: 'Tu Búsqueda'
    },
    en: {
      title: 'Research AI',
      subtitle: 'Your bridge to scientific literature',
      startBtn: 'Start Research',
      backBtn: 'Back',
      langBtn: 'Language',
      placeholder: 'Ask about research papers...',
      send: 'Send',
      evalMode: 'Evaluation Mode',
      noArticles: 'No articles loaded.',
      match: 'match',
      analyzing: '🤖 Analyzing papers and generating comparison...',
      iaSummary: 'AI Summary',
      summaryReady: 'Summary ready',
      viewAbstract: 'View abstract',
      hideInfo: 'Hide info',
      recallMetric: 'Recall Metric',
      reRanker: 'Re-ranker',
      advancedModel: 'Advanced Model',
      fastModel: 'Fast Model',
      globalPerformance: 'Global Performance',
      llmInference: 'LLM Inference',
      totalTime: 'Total Time',
      visualMapBtn: 'VIEW VISUAL RELATIONSHIP MAP',
      relevanceMap: 'Relevance Map',
      seeTop5: 'VIEW ONLY TOP-5',
      seeExtended: 'VIEW EXTENDED NETWORK',
      generatingGraph: 'Generating visual connections...',
      connectionError: 'Sorry, there was a connection error.',
      yourSearchNode: 'Your Search'
    },
    fr: {
      title: 'Recherche IA',
      subtitle: 'Votre pont vers la littérature scientifique',
      startBtn: 'Commencer la recherche',
      backBtn: 'Retour',
      langBtn: 'Langue',
      placeholder: 'Poser des questions sur...',
      send: 'Envoyer',
      evalMode: 'Mode Évaluation',
      noArticles: 'Aucun article chargé.',
      match: 'correspondance',
      analyzing: '🤖 Analyse des articles et comparaison...',
      iaSummary: 'Résumé IA',
      summaryReady: 'Résumé prêt',
      viewAbstract: 'Voir l\'abstract',
      hideInfo: 'Masquer info',
      recallMetric: 'Métrique Recall',
      reRanker: 'Re-ranker',
      advancedModel: 'Modèle Avancé',
      fastModel: 'Modèle Rapide',
      globalPerformance: 'Performance Globale',
      llmInference: 'Inférence LLM',
      totalTime: 'Temps Total',
      visualMapBtn: 'VOIR LA CARTE VISUELLE DES RELATIONS',
      relevanceMap: 'Carte de Pertinence',
      seeTop5: 'VOIR UNIQUEMENT LE TOP-5',
      seeExtended: 'VOIR LE RÉSEAU ÉTENDU',
      generatingGraph: 'Génération des connexions visuelles...',
      connectionError: 'Désolé, une erreur de connexion est survenue.',
      yourSearchNode: 'Votre Recherche'
    }
  };

  translate(key: string) {
    return this.translations[this.currentLang()][key] || key;
  }

  setLanguage(lang: 'es' | 'en' | 'fr') {
    this.currentLang.set(lang);
  }
}