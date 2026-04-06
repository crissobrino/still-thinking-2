import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { RagResponse } from '../../models/rag-response.model';

@Component({
  selector: 'app-chat',
  standalone: true, 
  imports: [
    CommonModule,
    FormsModule
],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.css']
})
export class ChatComponent {

  userInput = '';
  messages: { text: string, isUser: boolean }[] = [];
  currentResponse?: RagResponse;
  isLoading = false;

  constructor(private chatService: ChatService) {}

  onSendMessage() {

    if (!this.userInput.trim()) return;

    this.messages.push({
      text: this.userInput,
      isUser: true
    });

    this.isLoading = true;

    this.chatService.sendQuery(this.userInput).subscribe(response => {

      this.currentResponse = response;

      this.messages.push({
        text: response.answer,
        isUser: false
      });

      this.isLoading = false;
      this.userInput = '';

    });

  }

}