import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ThemeService } from '../../services/theme.service';
import { LanguageService } from '../../services/language.service';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.css']
})
export class LandingComponent {
  languages: ('es' | 'en' | 'fr')[] = ['es', 'en', 'fr'];

  constructor(
    public theme: ThemeService, 
    public lang: LanguageService, 
    private router: Router
  ) {}

  start() {
    this.router.navigate(['/chat']);
  }
}