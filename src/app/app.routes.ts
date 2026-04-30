import { Routes } from '@angular/router';
import { ChatComponent } from './components/chat/chat.component';
import { LandingComponent } from './components/landing/landing.component';

export const routes: Routes = [
  { path: '', component: LandingComponent },    // Esto cargará el Landing al abrir la web
  { path: 'chat', component: ChatComponent },    // Esto cargará el Chat en /chat
  { path: '**', redirectTo: '' }                 // Por si acaso escriben cualquier cosa
];