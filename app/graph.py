from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.llm import LLMClient
from app.prompts import (
    ANSWER_SYSTEM_PROMPT,
    GROUNDING_GRADER_SYSTEM_PROMPT,
    STRICT_RETRY_SUFFIX,
    build_context_block,
)
from app.scoring import combined_confidence, retrieval_confidence
from app.vector_store import PineconeVectorStore


FALLBACK_ANSWER = (
    "I could not find sufficient information in the provided Agentic AI eBook "
    "to answer this question."
)


class RAGState(TypedDict, total=False):
    question: str
    retrieved_contexts: list[dict]
    relevant_contexts: list[dict]
    retrieval_confidence: float
    answer: str
    grounding_score: float
    grounded: bool
    confidence: float
    grading_reason: str
    generation_attempts: int


class RAGGraph:
    def __init__(
        self,
        settings: Settings,
        vector_store: PineconeVectorStore,
        llm: LLMClient,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node("normalize_question", self.normalize_question)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("assess_context", self.assess_context)
        builder.add_node("generate_answer", self.generate_answer)
        builder.add_node("grade_grounding", self.grade_grounding)
        builder.add_node("finalize", self.finalize)
        builder.add_node("refuse", self.refuse)

        builder.add_edge(START, "normalize_question")
        builder.add_edge("normalize_question", "retrieve")
        builder.add_edge("retrieve", "assess_context")

        builder.add_conditional_edges(
            "assess_context",
            self.route_after_context,
            {
                "generate": "generate_answer",
                "refuse": "refuse",
            },
        )

        builder.add_edge("generate_answer", "grade_grounding")

        builder.add_conditional_edges(
            "grade_grounding",
            self.route_after_grounding,
            {
                "finalize": "finalize",
                "retry": "generate_answer",
                "refuse": "refuse",
            },
        )

        builder.add_edge("finalize", END)
        builder.add_edge("refuse", END)

        return builder.compile()

    def normalize_question(self, state: RAGState) -> dict:
        question = " ".join((state.get("question") or "").split()).strip()
        if len(question) < 2:
            raise ValueError("Question is too short.")
        return {
            "question": question,
            "generation_attempts": 0,
        }

    def retrieve(self, state: RAGState) -> dict:
        hits = self.vector_store.search(
            query=state["question"],
            top_k=self.settings.top_k,
        )

        contexts = [
            {
                "chunk_id": hit.chunk_id,
                "page": hit.page,
                "score": round(hit.score, 4),
                "content": hit.text,
                "source": hit.source,
            }
            for hit in hits
        ]

        return {"retrieved_contexts": contexts}

    def assess_context(self, state: RAGState) -> dict:
        all_contexts = state.get("retrieved_contexts", [])
        relevant = [
            item
            for item in all_contexts
            if float(item.get("score", 0.0)) >= self.settings.min_similarity
            and item.get("content")
        ]

        retrieval_score = retrieval_confidence(
            [float(item["score"]) for item in relevant],
            self.settings.min_similarity,
        )

        return {
            "relevant_contexts": relevant,
            "retrieval_confidence": retrieval_score,
        }

    def route_after_context(self, state: RAGState) -> Literal["generate", "refuse"]:
        return "generate" if state.get("relevant_contexts") else "refuse"

    def generate_answer(self, state: RAGState) -> dict:
        attempts = int(state.get("generation_attempts", 0)) + 1
        context_block = build_context_block(state["relevant_contexts"])

        system_prompt = ANSWER_SYSTEM_PROMPT
        if attempts > 1:
            system_prompt += STRICT_RETRY_SUFFIX

        user_prompt = f"""QUESTION:
{state["question"]}

CONTEXT:
{context_block}

Write the grounded answer now.
"""

        answer = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        return {
            "answer": answer or FALLBACK_ANSWER,
            "generation_attempts": attempts,
        }

    def grade_grounding(self, state: RAGState) -> dict:
        context_block = build_context_block(state["relevant_contexts"])

        grader_prompt = f"""QUESTION:
{state["question"]}

SOURCE CONTEXT:
{context_block}

CANDIDATE ANSWER:
{state["answer"]}

Evaluate grounding.
"""

        try:
            result = self.llm.json_chat(
                system_prompt=GROUNDING_GRADER_SYSTEM_PROMPT,
                user_prompt=grader_prompt,
            )
            grounded = bool(result.get("grounded", False))
            score = float(result.get("score", 0.0))
            reason = str(result.get("reason", ""))
        except Exception as exc:
            # Strict failure mode: if grading fails, do not claim the answer is grounded.
            grounded = False
            score = 0.0
            reason = f"Grounding grader failed: {type(exc).__name__}"

        score = max(0.0, min(1.0, score))
        grounded = grounded and score >= self.settings.min_grounding_score

        return {
            "grounded": grounded,
            "grounding_score": round(score, 4),
            "grading_reason": reason,
        }

    def route_after_grounding(
        self,
        state: RAGState,
    ) -> Literal["finalize", "retry", "refuse"]:
        if state.get("grounded", False):
            return "finalize"

        attempts = int(state.get("generation_attempts", 0))
        if attempts < self.settings.max_generation_attempts:
            return "retry"

        return "refuse"

    def finalize(self, state: RAGState) -> dict:
        confidence = combined_confidence(
            retrieval_score=float(state.get("retrieval_confidence", 0.0)),
            grounding_score=float(state.get("grounding_score", 0.0)),
            grounded=bool(state.get("grounded", False)),
        )
        return {"confidence": confidence}

    def refuse(self, state: RAGState) -> dict:
        retrieval_score = float(state.get("retrieval_confidence", 0.0))
        confidence = combined_confidence(
            retrieval_score=retrieval_score,
            grounding_score=0.0,
            grounded=False,
        )
        return {
            "answer": FALLBACK_ANSWER,
            "grounded": False,
            "grounding_score": 0.0,
            "confidence": confidence,
        }

    def invoke(self, question: str) -> RAGState:
        return self.graph.invoke({"question": question})
