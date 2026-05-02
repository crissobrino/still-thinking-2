import tiktoken

def calculate_token_efficiency(context_text, answer_text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    
    tokens_context = len(encoding.encode(context_text))
    tokens_answer = len(encoding.encode(answer_text))
    
    # Eficiencia: Qué parte del contexto se tradujo en información útil
    # Una métrica común es la ratio de tokens de salida vs entrada
    efficiency = (tokens_answer / tokens_context) * 100
    
    return {
        "context_tokens": tokens_context,
        "answer_tokens": tokens_answer,
        "efficiency_pct": round(efficiency, 2)
    }

# Ejemplo de uso con tu respuesta de la API
# context = " ".join([a['abstract'] for a in response['articles']])
# stats = calculate_token_efficiency(context, response['answer'])