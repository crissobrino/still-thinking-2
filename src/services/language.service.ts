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
      recallMetric: 'Métrica Recall'
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
      recallMetric: 'Recall Metric'
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
      recallMetric: 'Métrique Recall'
    }
  };

  translate(key: string) {
    return this.translations[this.currentLang()][key] || key;
  }

  setLanguage(lang: 'es' | 'en' | 'fr') {
    this.currentLang.set(lang);
  }
}