import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';

import { AppComponent } from './app.component';
import { ChatComponent } from './components/chat/chat.component';

@NgModule({
  declarations: [
    // DEJA ESTO VACÍO. Los componentes standalone no se declaran aquí.
  ],
  imports: [
    BrowserModule,
    FormsModule,
    AppComponent, // MUEVE AppComponent AQUÍ
    ChatComponent  // MUEVE ChatComponent AQUÍ
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }