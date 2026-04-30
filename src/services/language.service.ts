import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class LanguageService {
  currentLang = signal<'es' | 'en' | 'fr'>('es');

  private translations: any = {
    es: {
      title: 'Investigación AI',
      subtitle: 'Tu puente hacia la literatura científica',
      startBtn: 'Empezar Investigación',
      langBtn: 'Idioma',
      placeholder: 'Pregunta sobre artículos...',
      send: 'Enviar',
      evalMode: 'Modo Evaluación (ENN)',
      noArticles: 'No hay artículos cargados.',
      match: 'coincidencia'
    },
    en: {
      title: 'Research AI',
      subtitle: 'Your bridge to scientific literature',
      startBtn: 'Start Research',
      langBtn: 'Language',
      placeholder: 'Ask about research papers...',
      send: 'Send',
      evalMode: 'Evaluation Mode (ENN)',
      noArticles: 'No articles loaded.',
      match: 'match'
    },
    fr: {
      title: 'Recherche IA',
      subtitle: 'Votre pont vers la littérature scientifique',
      startBtn: 'Commencer la recherche',
      langBtn: 'Langue',
      placeholder: 'Poser des questions sur...',
      send: 'Envoyer',
      evalMode: 'Mode Évaluation (ENN)',
      noArticles: 'Aucun artículo chargé.',
      match: 'correspondance'
    }
  };

  translate(key: string) {
    return this.translations[this.currentLang()][key] || key;
  }

  setLanguage(lang: 'es' | 'en' | 'fr') {
    this.currentLang.set(lang);
  }
}