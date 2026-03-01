"""
Dimension Matcher - Busca candidatos en una dimensión específica.
Código extraído de api/main.py (líneas 126-206).
"""

from typing import List, Dict, Any, Optional, Tuple

from tfm_match.core.embeddings_manager import EmbeddingsManager


class DimensionMatcher:
    """Maneja búsquedas en colecciones ChromaDB por dimensión."""
    
    def __init__(self, chroma_collections: Dict[str, Any], embeddings_manager: EmbeddingsManager):
        """
        Args:
            chroma_collections: Dict con colecciones ChromaDB por dimensión
            embeddings_manager: Manager para generar embeddings
        """
        self.collections = chroma_collections
        self.embeddings_mgr = embeddings_manager
    
    @staticmethod
    def dist_to_sim(dist: float) -> float:
        """
        Convierte distancia coseno a similitud.
        LÓGICA EXACTA de dist_to_sim original (líneas 126-133).
        
        Args:
            dist: Distancia coseno
            
        Returns:
            Similitud (0 a 1)
        """
        sim = 1.0 - float(dist)
        if sim < 0:
            sim = 0.0
        if sim > 1:
            sim = 1.0
        return sim
    
    def query_dimension(
        self,
        dim: str,
        query_text: str,
        n_results: int,
        dedup_by_candidate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Busca en una dimensión específica.
        LÓGICA EXACTA de query_dim original (líneas 143-206).
        
        Args:
            dim: Dimensión ('skills', 'experience', etc.)
            query_text: Texto de búsqueda
            n_results: Número de resultados
            dedup_by_candidate: Si deduplicar por candidato
            
        Returns:
            Lista de hits normalizados con id_candidate, sim, dist, meta, doc
        """
        col = self.collections.get(dim)
        if not col or not query_text or not query_text.strip():
            return []

        # -----------------------------
        # Rule-based scoring helpers (language)
        # -----------------------------
        LANG_UI_TO_CANON = {
            "ingles": "english",
            "inglés": "english",
            "english": "english",
            "frances": "french",
            "francés": "french",
            "french": "french",
            "portugues": "portuguese",
            "portugués": "portuguese",
            "portuguese": "portuguese",
            "espanol": "spanish",
            "español": "spanish",
            "spanish": "spanish",
            "alemán": "german",
            "aleman": "german",
            "german": "german",
            "italiano": "italian",
            "italian": "italian",
        }
        CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
        KW_RANK = {"basic": 2, "intermediate": 4, "advanced": 5, "fluent": 6, "native": 7}

        def _parse_lang_requirements(q: str) -> List[Tuple[str, int]]:
            """
            Parsea requisitos desde texto libre.
            Soporta entradas como:
              - "Inglés B2"
              - "english: B2; french: A2"
              - "portugués intermedio"
            Retorna lista de (canon_lang, req_rank). req_rank=0 si no hay nivel.
            """
            if not q:
                return []
            txt = q.strip()
            # Separar por delimitadores comunes
            parts = [p.strip() for p in re.split(r"[;\n,|]+", txt) if p.strip()]
            out: List[Tuple[str, int]] = []
            for p in parts:
                p_low = p.lower().strip()
                # detecta idioma (primer token)
                # acepta "english: B2" / "ingles B2" / "francés avanzado"
                m = re.match(r"^\s*([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\s*[: ]?\s*([a-zA-Z0-9]+)?\s*$", p)
                lang_raw = m.group(1).strip().lower() if m else ""
                lvl_raw = (m.group(2) or "").strip() if m else ""
                if not lang_raw:
                    continue
                canon = LANG_UI_TO_CANON.get(lang_raw, lang_raw)

                req_rank = 0
                if lvl_raw:
                    lv_up = lvl_raw.upper()
                    if lv_up in CEFR_RANK:
                        req_rank = CEFR_RANK[lv_up]
                    else:
                        lv_low = lvl_raw.lower()
                        req_rank = KW_RANK.get(lv_low, 0)
                out.append((canon, req_rank))

            # Dedup por idioma: nos quedamos con el mayor requerimiento
            best: Dict[str, int] = {}
            for canon, rr in out:
                best[canon] = max(best.get(canon, 0), rr)
            return list(best.items())

        def _language_rule_score(meta: Dict[str, Any], reqs: List[Tuple[str, int]]) -> Optional[float]:
            """
            Implementa la regla del documento:
              - cumple (>=N) -> 1
              - a 1 nivel por debajo -> 0.5
              - 2+ niveles por debajo -> 0
            Si hay varios idiomas, usa media simple (no hay "importancia" explícita en el payload).
            """
            if not reqs:
                return None
            scores: List[float] = []
            for canon, req_rank in reqs:
                has_flag = meta.get(f"has_{canon}", 0)
                try:
                    has_flag = int(has_flag)
                except Exception:
                    has_flag = 0
                if has_flag != 1:
                    scores.append(0.0)
                    continue

                cand_rank = meta.get(f"lvl_{canon}_rank", 0)
                try:
                    cand_rank = int(cand_rank)
                except Exception:
                    cand_rank = 0

                # si no se especificó nivel, basta con tener el idioma
                if req_rank <= 0:
                    scores.append(1.0)
                    continue

                if cand_rank >= req_rank:
                    scores.append(1.0)
                elif cand_rank == req_rank - 1:
                    scores.append(0.5)
                else:
                    scores.append(0.0)

            if not scores:
                return None
            return sum(scores) / len(scores)
        
        emb = self.embeddings_mgr.embed_text(query_text)

        # Importante: si hay múltiples documentos por candidato (p.ej., job_title::1..N),
        # pedir solo n_results puede devolver muchos hits del mismo candidato.
        # Hacemos oversampling y luego deduplicamos y recortamos.
        n_fetch = n_results
        if dedup_by_candidate:
            # Factor conservador para mantener latencia controlada.
            # (En job_title típicamente hay ~6 docs por candidato con el nuevo indexador.)
            oversample_factor = 6 if dim == "job_title" else 4

            # Importante: NUNCA pedir menos de n_results (si n_results es grande).
            # Solo aplicamos "cap" al oversampling extra, no al mínimo requerido.
            max_fetch = max(n_results, 200)
            n_fetch = min(n_results * oversample_factor, max_fetch)
        
        res = col.query(
            query_embeddings=[emb],
            n_results=n_fetch,
            include=["metadatas", "documents", "distances"]
        )
        
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]
        
        hits = []
        # Si es language, calcula requisitos desde el query_text para rule-based sim
        lang_reqs: List[Tuple[str, int]] = []
        if dim == "language":
            import re  # local para evitar dependencia global si no se usa
            lang_reqs = _parse_lang_requirements(query_text)

        for i in range(len(ids)):
            # Protección contra None en metadatos, documentos y distancias
            meta = metas[i] if metas and i < len(metas) and metas[i] is not None else {}
            doc = docs[i] if docs and i < len(docs) and docs[i] is not None else ""
            dist = float(dists[i]) if dists and i < len(dists) and dists[i] is not None else 1.0
            
            # Para skills a veces el id devuelto ya es el candidate_id.
            cand = meta.get("id_candidate")
            if cand is None or str(cand).strip() == "":
                cand = ids[i]
            
            cand = str(cand)
            h = {
                "id_candidate": cand,
                "dist": dist,
                "sim": self.dist_to_sim(dist),
                "meta": meta or {},
                "doc": doc or "",
            }

            # Override opcional de sim para language usando regla discreta (si hay requisitos parseables)
            if dim == "language" and lang_reqs:
                rule_sim = _language_rule_score(h["meta"], lang_reqs)
                if rule_sim is not None:
                    h["sim"] = float(rule_sim)
                    h["meta"]["sim_language_rule"] = float(rule_sim)
                    h["meta"]["language_requirements"] = [{"language": l, "req_rank": r} for (l, r) in lang_reqs]

            hits.append(h)
        
        if not dedup_by_candidate:
            return hits
        
        # Dedup por candidato (nos quedamos con el mejor sim)
        best: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            c = h["id_candidate"]
            if c not in best or h["sim"] > best[c]["sim"]:
                best[c] = h

        # Ordenar por similitud desc y recortar a n_results (candidatos únicos)
        out = sorted(best.values(), key=lambda x: x["sim"], reverse=True)
        return out[:n_results]
