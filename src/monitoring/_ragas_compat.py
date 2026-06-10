"""Shim de compatibilité entre ragas et langchain-community >= 0.4.

ragas 0.4.x importe ``ChatVertexAI`` depuis
``langchain_community.chat_models.vertexai``, un module supprimé de
langchain-community 0.4 — l'import de ragas plante donc avec la pile
LangChain 1.x du projet. ragas n'utilise cette classe que dans des tests
``isinstance`` : une classe factice jamais instanciée suffit.

Ce module doit être importé AVANT tout import de ragas.
"""

import sys
import types

_MODULE = "langchain_community.chat_models.vertexai"

if _MODULE not in sys.modules:
    try:
        __import__(_MODULE)
    except ImportError:
        _shim = types.ModuleType(_MODULE)

        class _ChatVertexAIStub:
            """Jamais instanciée : sert uniquement aux isinstance de ragas."""

        _shim.ChatVertexAI = _ChatVertexAIStub
        sys.modules[_MODULE] = _shim
