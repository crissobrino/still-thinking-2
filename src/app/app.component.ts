import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatComponent } from './components/chat/chat.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, ChatComponent],
  template: `
    @if (showSplash) {
      <div class="splash-container">
        <div class="content">
          <div class="logo">📚🤖</div>
          <h1>Research Assistant AI</h1>
          <p>Tu puente hacia la literatura científica</p>
          <button (click)="goToApp()">Empezar Investigación</button>
        </div>
      </div>
    } @else {
      <app-chat></app-chat>
    }
  `,
  styleUrls: ['./app.css'] // Asegúrate de crear este archivo con el CSS que te di antes
})
export class AppComponent {
  showSplash = true;

  goToApp() {
    this.showSplash = false;
  }
}
