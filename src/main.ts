import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { provideHttpClient } from '@angular/common/http'; // Necesario para tu ChatService

bootstrapApplication(AppComponent, {
  providers: [
    provideHttpClient() // Proveemos HttpClient aquí para que ChatService pueda hacer peticiones
  ]
}).catch(err => console.error(err));