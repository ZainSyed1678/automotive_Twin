"""
Knowledge Base Ingestion, Hybrid Search Index, and Similar-Case Retrieval Engine.
Operates 100% locally and offline using standard library + scikit-learn TF-IDF.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.schema import Observations
from .schema import DocType, KnowledgeDocument, SearchResult, SimilarCaseResult


class KnowledgeBase:
    """
    Local technical knowledge base with hybrid search over OBD-II definitions,
    repair manual excerpts, and technical service bulletins.
    """

    def __init__(self, doc_path: Optional[str] = None):
        if doc_path is None:
            doc_path = str(Path(__file__).parent / "documents.yaml")

        self.doc_path = Path(doc_path)
        self.documents: List[KnowledgeDocument] = []
        self.doc_map: Dict[str, KnowledgeDocument] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        if not self.doc_path.exists():
            raise FileNotFoundError(f"Knowledge documents file not found at: {self.doc_path}")

        with open(self.doc_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_docs = data.get("documents", [])
        corpus: List[str] = []

        for item in raw_docs:
            doc = KnowledgeDocument(**item)
            self.documents.append(doc)
            self.doc_map[doc.doc_id] = doc

            # Build searchable text representation: title + summary + content + DTC tags
            doc_text = (
                f"{doc.title} {doc.summary} {doc.content} "
                f"{' '.join(doc.related_dtcs)} {doc.component_id or ''}"
            )
            corpus.append(doc_text)

        # Fit TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def get_code_info(self, code: str) -> Optional[SearchResult]:
        """
        Direct exact-match lookup for any generic OBD-II fault code.
        """
        normalized_code = code.strip().upper()
        # Check direct doc_id match first (e.g. DOC-OBD-P0171)
        direct_id = f"DOC-OBD-{normalized_code}"
        if direct_id in self.doc_map:
            doc = self.doc_map[direct_id]
            return SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                doc_type=doc.doc_type.value,
                component_id=doc.component_id,
                related_dtcs=doc.related_dtcs,
                score=1.0,
                snippet=doc.summary,
                citation=f"[{doc.doc_id}] {doc.title}",
            )

        # Check related_dtcs across all OBD code documents
        for doc in self.documents:
            if doc.doc_type == DocType.OBD_CODE and normalized_code in doc.related_dtcs:
                return SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    doc_type=doc.doc_type.value,
                    component_id=doc.component_id,
                    related_dtcs=doc.related_dtcs,
                    score=0.95,
                    snippet=doc.summary,
                    citation=f"[{doc.doc_id}] {doc.title}",
                )

        return None

    def search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        component_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Hybrid search combining TF-IDF cosine similarity with structural metadata filtering.
        """
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        results: List[SearchResult] = []
        normalized_query_upper = query.upper()

        for idx, doc in enumerate(self.documents):
            score = float(similarities[idx])

            # Apply structural filters
            if doc_type and doc.doc_type.value != doc_type:
                continue
            if component_id and doc.component_id != component_id:
                continue

            # Metadata keyword boosts (e.g. exact DTC mention)
            for dtc in doc.related_dtcs:
                if dtc in normalized_query_upper:
                    score += 0.25

            if score > 0.05:
                # Extract clean snippet from content (first 250 characters)
                snippet_text = doc.summary if doc.summary else doc.content[:250].replace("\n", " ")
                results.append(
                    SearchResult(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        doc_type=doc.doc_type.value,
                        component_id=doc.component_id,
                        related_dtcs=doc.related_dtcs,
                        score=round(score, 4),
                        snippet=snippet_text,
                        citation=f"[{doc.doc_id}] {doc.title}",
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class SimilarCaseIndex:
    """
    Search index over historical development service cases.
    Guarantees strict isolation: NEVER loads or indexes held-out evaluation cases.
    """

    def __init__(self, dev_cases_path: Optional[str] = None):
        if dev_cases_path is None:
            dev_cases_path = str(Path(__file__).parent.parent / "data" / "cases_dev.json")

        self.dev_cases_path = Path(dev_cases_path)
        # Security Guard: Assert that evaluation cases are NEVER loaded here
        if "cases_eval" in self.dev_cases_path.name:
            raise PermissionError("Security violation! SimilarCaseIndex cannot load held-out evaluation cases.")

        self.cases: List[Dict[str, Any]] = []
        self._load_cases()

    def _load_cases(self) -> None:
        if not self.dev_cases_path.exists():
            raise FileNotFoundError(f"Development cases not found at: {self.dev_cases_path}")

        with open(self.dev_cases_path, "r", encoding="utf-8") as f:
            self.cases = json.load(f)

    def find_similar(
        self,
        observations: Observations,
        top_k: int = 3,
    ) -> List[SimilarCaseResult]:
        """
        Finds historical service cases matching current observations using Jaccard & feature overlap.
        Returns verified historical repair outcomes while stripping generator private metadata.
        """
        q_dtcs = set(observations.dtc_codes)
        q_symptoms = observations.symptoms

        scored: List[SimilarCaseResult] = []

        for case in self.cases:
            c_obs = case.get("observations", {})
            c_dtcs = set(c_obs.get("dtc_codes", []))
            c_symptoms = c_obs.get("symptoms", {})

            # 1. DTC similarity (Jaccard)
            dtc_intersection = q_dtcs.intersection(c_dtcs)
            dtc_union = q_dtcs.union(c_dtcs)
            dtc_sim = (len(dtc_intersection) / len(dtc_union)) if dtc_union else 0.5

            # 2. Symptom similarity (Match ratio over shared symptom keys)
            shared_keys = set(q_symptoms.keys()).intersection(set(c_symptoms.keys()))
            matching_symptoms: Dict[str, str] = {}
            if shared_keys:
                for k in shared_keys:
                    if q_symptoms[k] == c_symptoms[k]:
                        matching_symptoms[k] = q_symptoms[k]
                symptom_sim = len(matching_symptoms) / len(shared_keys)
            else:
                symptom_sim = 0.0 if (q_symptoms or c_symptoms) else 0.5

            # Combined similarity weight
            combined_score = round(0.55 * dtc_sim + 0.45 * symptom_sim, 4)

            # Build readable resolution summary
            fault_id = case.get("true_fault_id", "unknown_fault")
            repair_summary = (
                f"Historical workshop case confirmed {fault_id}. "
                f"Verified with {len(matching_symptoms)} matching symptom(s) and {len(dtc_intersection)} DTC(s)."
            )

            scored.append(
                SimilarCaseResult(
                    case_id=case.get("case_id", "CASE-UNKNOWN"),
                    similarity_score=combined_score,
                    matching_dtcs=sorted(list(dtc_intersection)),
                    matching_symptoms=matching_symptoms,
                    confirmed_fault=fault_id,
                    confirmed_component=fault_id.split("_")[0],
                    difficulty=case.get("difficulty", "standard"),
                    repair_summary=repair_summary,
                )
            )

        scored.sort(key=lambda item: item.similarity_score, reverse=True)
        return scored[:top_k]
