"""Testando o AGENTE sem LLM, usando um dublê (fake) do cliente Gemini.

Isso só é possível porque o Agente recebe o cliente por injeção de dependência.
Aqui verificamos a LÓGICA ao redor do modelo: montagem do prompt, leitura das
ferramentas chamadas e a rota de decisão — tudo determinístico.
"""
from types import SimpleNamespace as NS

from agente.agente import Agente


class ClienteFalso:
    def __init__(self, texto, ferramentas=()):
        historico = [NS(parts=[NS(function_call=NS(name=f))]) for f in ferramentas]
        self._resposta = NS(text=texto, automatic_function_calling_history=historico)
        self.prompts = []
        self.models = self  # imita cliente.models.generate_content

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        return self._resposta


def test_rota_ferramenta_quando_modelo_chama_tool():
    cliente = ClienteFalso("Sua média é 8,0: aprovado.", ferramentas=["calcular_media"])
    resposta = Agente(cliente=cliente, modelo="falso").responder("Tirei 7, 8 e 9. Passei?")
    assert resposta.ferramentas_chamadas == ["calcular_media"]
    assert resposta.rota == "ferramenta"


def test_rota_conhecimento_usa_contexto_no_prompt():
    cliente = ClienteFalso("A frequência mínima é 75%.")
    resposta = Agente(cliente=cliente, modelo="falso").responder("Qual a frequência mínima?")
    assert resposta.rota == "conhecimento"
    assert "75%" in cliente.prompts[0]  # o contexto certo foi enviado ao modelo


def test_rota_sem_informacao():
    cliente = ClienteFalso("Não tenho essa informação.")
    resposta = Agente(cliente=cliente, modelo="falso").responder("Quem ganhou a Copa?")
    assert resposta.rota == "sem_informacao"
