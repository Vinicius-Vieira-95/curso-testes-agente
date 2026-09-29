"""Métricas customizadas com G-Eval: o critério é escrito em linguagem natural.

G-Eval funciona em 2 etapas:
  1. a partir do `criteria`, o juiz gera passos de avaliação (ou usamos os nossos
     `evaluation_steps`, mais estáveis);
  2. o juiz aplica esses passos ao caso de teste e devolve uma nota de 0 a 1.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams as P


def metrica_correcao(juiz):
    """Compara a resposta com o gabarito (expected_output)."""
    return GEval(
        name="Correção",
        criteria="Verifique se a resposta real está factualmente de acordo com a resposta esperada.",
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        threshold=0.7,
        model=juiz,
        async_mode=False,
    )


def metrica_nao_inventa(juiz):
    """Honestidade: sem informação no contexto, o agente deve admitir."""
    return GEval(
        name="Não Inventa",
        evaluation_steps=[
            "Leia o contexto recuperado e a pergunta.",
            "Se o contexto NÃO contém a resposta, a resposta real deve dizer que não tem a informação.",
            "Penalize fortemente qualquer dado (número, data, nome) que não esteja no contexto.",
            "Se o contexto contém a resposta, verifique se a resposta real a usa corretamente.",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT],
        threshold=0.7,
        model=juiz,
        async_mode=False,
    )


def metrica_clareza_para_calouro(juiz):
    """Critério de 'estilo': algo que um assert clássico nunca conseguiria medir."""
    return GEval(
        name="Clareza para Calouro",
        evaluation_steps=[
            "A resposta está em português e responde diretamente à pergunta na primeira frase?",
            "Evita jargões administrativos sem explicação?",
            "Tem no máximo 3 frases?",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
        threshold=0.6,
        model=juiz,
        async_mode=False,
    )


@pytest.mark.llm
def test_correcao_com_gabarito(agente, juiz):
    pergunta = "Com qual média eu sou aprovado direto?"
    resposta = agente.responder(pergunta)
    caso = LLMTestCase(input=pergunta, actual_output=resposta.texto,
                       expected_output="A média para aprovação direta é 7,0.")
    assert_test(caso, [metrica_correcao(juiz), metrica_clareza_para_calouro(juiz)])


@pytest.mark.llm
def test_nao_inventa_quando_nao_sabe(agente, juiz):
    pergunta = "Qual o valor da mensalidade do curso?"
    resposta = agente.responder(pergunta)
    caso = LLMTestCase(input=pergunta, actual_output=resposta.texto,
                       retrieval_context=resposta.contexto or ["(nenhum documento encontrado)"])
    assert_test(caso, [metrica_nao_inventa(juiz)])


@pytest.mark.llm
def test_metrica_detecta_resposta_inventada(juiz):
    """Teste DA MÉTRICA: uma resposta alucinada precisa tirar nota baixa.

    Se este teste passar a falhar, o problema está no critério, não no agente.
    """
    caso = LLMTestCase(
        input="Qual o valor da mensalidade do curso?",
        actual_output="A mensalidade é de R$ 850,00, com desconto de 10% à vista.",
        retrieval_context=["A frequência mínima exigida em cada disciplina é de 75% das aulas."],
    )
    metrica = metrica_nao_inventa(juiz)
    metrica.measure(caso)
    print(f"nota={metrica.score:.2f} motivo={metrica.reason}")
    assert metrica.score < metrica.threshold
