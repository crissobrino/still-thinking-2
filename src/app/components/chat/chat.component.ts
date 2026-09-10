import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { RagResponse, Article } from '../../models/rag-response.model';
import { ThemeService } from '../../services/theme.service';
import { LanguageService } from '../../services/language.service';
import { Router } from '@angular/router';
import { Network } from 'vis-network';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Component({
  selector: 'app-chat',
  standalone: true, 
  imports: [CommonModule, FormsModule], 
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.css'] 
})
export class ChatComponent {
  userInput = '';
  lastQuery = ''; 
  messages: { text: SafeHtml | string, isUser: boolean }[] = [];
  currentResponse?: RagResponse;
  isLoading = false;
  isEvalMode = false; 
  useReranker = false; 
  showGraph = false;
  usePowerfulModel: boolean = false; 
  showingFullGraph: boolean = false; 

  constructor(
    private chatService: ChatService,
    public lang: LanguageService,
    private cdr: ChangeDetectorRef,
    private router: Router,
    public theme: ThemeService,
    private sanitizer: DomSanitizer 
  ) {}

  // GRAPH LOGIC

  openGraphModal() {
    this.showGraph = true;
    // Small delay to make sure the 'mynetwork' div exists in the DOM
    setTimeout(() => this.renderGraph(), 50);
  }

  toggleFullGraph() {
    this.showingFullGraph = !this.showingFullGraph;
    this.renderGraph(); 
  }

  renderGraph() {
    const container = document.getElementById('mynetwork');
    if (!container || !this.currentResponse) return;

    // Show either the top 5 from the chat or the extended set of 30 on the full map
    const articlesToRender = this.showingFullGraph
      ? (this.currentResponse.extended_articles || this.currentResponse.articles)
      : this.currentResponse.articles;

    const nodesArray: any[] = [{
      id: 0,
      label: this.lang.translate('yourSearchNode'),
      color: '#4f46e5',
      font: { color: 'white', size: 16, bold: true },
      shape: 'circle',
      shadow: true
    }];

    articlesToRender.forEach((art, index) => {
      const isExtra = this.showingFullGraph && index >= 5; // visually distinguish the extended nodes
      nodesArray.push({
        id: index + 1,
        label: art.title.substring(0, 25) + '...',
        title: `<b>${art.title}</b><br>${art.authors} (${art.year})`,
        value: art.relevanceScore || 10,
        color: isExtra ? '#e2e8f0' : '#c7d2fe', // gray for extras, blue for main results
        shape: 'dot',
        shadow: true
      });
    });

    const edgesArray = nodesArray.filter(n => n.id !== 0).map(n => ({
      from: 0,
      to: n.id,
      value: (n.value / 10),
      color: { color: '#818cf8', opacity: 0.4 }
    }));

    const options = {
      nodes: { scaling: { min: 10, max: 35 } },
      physics: {
        stabilization: true,
        barnesHut: { 
          springLength: this.showingFullGraph ? 250 : 150, // more room when there are many nodes
          gravitationalConstant: -2000
        }
      },
      interaction: { hover: true }
    };

    new Network(container, { nodes: nodesArray, edges: edgesArray }, options);
  }

  // MESSAGES LOGIC

  onSendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    const savedQuery = this.userInput;
    this.lastQuery = savedQuery;
    this.messages.push({ text: savedQuery, isUser: true });
    
    this.userInput = ''; 
    this.isLoading = true; 
    
    this.chatService.getQueryResponse(savedQuery, 5, this.isEvalMode, this.useReranker, this.usePowerfulModel)
      .subscribe({
        next: (response: RagResponse) => {
          try {
            let botText = response.answer || "";
            // Strip the "thinking" trace some models (DeepSeek/Qwen-style) emit
            botText = botText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

            this.processArticleScores(response.articles);
            if (response.extended_articles) this.processArticleScores(response.extended_articles);
            if (response.enn_articles) this.processArticleScores(response.enn_articles);

            // Extract each article's "key difference" from the bot's free-text response
            response.articles.forEach(article => {
              const safeTitle = article.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
              const regex = new RegExp(`${safeTitle}[^—\\-]*[—\\-]\\s*(.*?)(?:\\n|$)`, 'i');
              const match = botText.match(regex);
              if (match) article.keyDifference = match[1].trim();
            });

            this.currentResponse = response;
            this.messages.push({ 
              text: this.sanitizer.bypassSecurityTrustHtml(botText), 
              isUser: false 
            });

          } finally {
            this.isLoading = false; 
            this.cdr.detectChanges(); 
          }
        },
        error: (err) => {
          console.error("Search error:", err);
          this.messages.push({ text: this.lang.translate('connectionError'), isUser: false });
          this.isLoading = false;
          this.cdr.detectChanges();
        }
      });
  }

  private processArticleScores(articles: Article[]) {
    if (!articles) return;
    articles.forEach(art => {
      let rawScore = art.score || art.relevanceScore || 0;
      // Normalize to a 0-100 scale for the graph bubble sizes
      art.relevanceScore = rawScore > 1 ? rawScore : rawScore * 100;
    });
  }

  // UTILITIES

  viewSummary(article: Article) {
    if (!article.abstract || article.summary) return;
    article.loadingSummary = true;
    this.chatService.getSummary(article.abstract, this.lastQuery, this.currentResponse?.language || 'en')
      .subscribe({
        next: (res) => {
          article.summary = res.summary.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
          article.loadingSummary = false;
          this.cdr.detectChanges();
        },
        error: () => article.loadingSummary = false
      });
  }

  toggleAbstract(article: Article) {
    article.showAbstract = !article.showAbstract;
  }

  goBack() {
    this.router.navigate(['/']);
  }
}