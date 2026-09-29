"""Pacote do agente usado no curso 'Testes de Agentes baseados em LLM'."""
import logging

# O SDK do Gemini emite um aviso (inofensivo) quando usamos chamada automática
# de ferramentas em generate_content. Silenciamos para não poluir a saída dos testes.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
