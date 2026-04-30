import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router'; // Importante para las rutas

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet], // Solo necesitamos el RouterOutlet
  template: `
    <router-outlet></router-outlet> 
  `
})
export class AppComponent {}