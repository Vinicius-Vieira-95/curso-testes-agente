"""O agente testado no curso: um assistente acadêmico.

Fluxo de uma pergunta:
  1. busca trechos relevantes na base de conhecimento (contexto);
  2. envia pergunta + contexto ao Gemini, que PODE chamar ferramentas
     (calcular_media, consultar_calendario);
  3. devolve uma RespostaAgente com tudo o que precisamos para testar:
     texto, contexto usado, ferramentas chamadas e a rota de decisão.
"""
from dataclasses import dataclass, field

from agente.base_conhecimento import buscar_contexto
from agente.config import carregar_configuracao
from agente.ferramentas import FERRAMENTAS
from agente.resiliencia import LimitadorDeTaxa, com_retry

INSTRUCOES = (
    "Você é o assistente acadêmico do curso de Ciência da Computação. "
    "Responda em português, de forma curta e objetiva. "
    "Use SOMENTE as informações do CONTEXTO ou o resultado das ferramentas. "
    "Para cálculos de média use a ferramenta calcular_media; para datas use "
    "consultar_calendario. Se a informação não estiver disponível, diga "
    "'Não tenho essa informação.' e não invente."
)


@dataclass
class RespostaAgente:
    pergunta: str
    texto: str
    contexto: list[str] = field(default_factory=list)
    ferramentas_chamadas: list[str] = field(default_factory=list)

    @property
    def rota(self) -> str:
        """Rota de decisão tomada pelo agente — útil para testes de roteamento."""
        if self.ferramentas_chamadas:
            return "ferramenta"
        if self.contexto:
            return "conhecimento"
        return "sem_informacao"


class Agente:
    """`cliente` é injetável: em produção é o genai.Client; nos testes, um dublê."""

    def __init__(self, cliente=None, modelo: str | None = None):
        cfg = carregar_configuracao() if cliente is None or modelo is None else None
        if cliente is None:
            from google import genai
            cliente = genai.Client(api_key=cfg.google_api_key)
        self.cliente = cliente
        self.modelo = modelo or cfg.modelo_agente
        rpm = cfg.requisicoes_por_minuto if cfg else 60
        tentativas = cfg.max_tentativas if cfg else 5
        self._limitador = LimitadorDeTaxa(rpm)
        self._gerar = com_retry(max_tentativas=tentativas)(self._gerar_sem_retry)

    def _gerar_sem_retry(self, conteudo: str):
        from google.genai import types
        self._limitador.aguardar_vez()
        return self.cliente.models.generate_content(
            model=self.modelo,
            contents=conteudo,
            config=types.GenerateContentConfig(
                system_instruction=INSTRUCOES,
                tools=FERRAMENTAS,
                temperature=0,
            ),
        )

    def responder(self, pergunta: str) -> RespostaAgente:
        contexto = buscar_contexto(pergunta)
        bloco = "\n".join(f"- {c}" for c in contexto) or "(vazio)"
        resposta = self._gerar(f"CONTEXTO:\n{bloco}\n\nPERGUNTA: {pergunta}")

        chamadas = []
        for conteudo in resposta.automatic_function_calling_history or []:
            for parte in conteudo.parts or []:
                if parte.function_call:
                    chamadas.append(parte.function_call.name)

        return RespostaAgente(
            pergunta=pergunta,
            texto=(resposta.text or "").strip(),
            contexto=contexto,
            ferramentas_chamadas=chamadas,
        )
