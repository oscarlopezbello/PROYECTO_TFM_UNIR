"""
Hard Filters - Aplica filtros obligatorios (educación, idioma).
Código extraído de api/main.py (líneas 220-336).
"""

from typing import List, Dict, Any


# Mapeos exactos del código original
EDU_RANK_UI = {
    "tecnico": 2,
    "técnico": 2,
    "tecnologo": 3,
    "tecnólogo": 3,
    "profesional": 4,
    "posgrado": 5,
}

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
}

CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


class HardFilters:
    """Aplica filtros obligatorios sobre candidatos."""
    
    def __init__(self, chroma_collections: Dict[str, Any]):
        """
        Args:
            chroma_collections: Dict con colecciones ChromaDB
        """
        self.collections = chroma_collections
    
    def fetch_metas_by_doc_ids(self, dim: str, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene metadatas por doc_id.
        LÓGICA EXACTA de fetch_metas_by_doc_ids original (líneas 251-272).
        
        Args:
            dim: Dimensión ('education', 'language', 'sector', 'job_title')
            doc_ids: Lista de IDs de documentos
            
        Returns:
            Dict de doc_id -> metadata
        """
        col = self.collections.get(dim)
        if not col or not doc_ids:
            return {}
        try:
            got = col.get(ids=doc_ids, include=["metadatas"])
            ids_out = got.get("ids", [])
            metas_out = got.get("metadatas", [])
            out = {}
            for i in range(len(ids_out)):
                out[str(ids_out[i])] = metas_out[i] or {}
            return out
        except Exception:
            return {}
    
    def apply(self, candidate_ids: List[str], filters_config: Any) -> List[str]:
        """
        Aplica hard filters.
        LÓGICA EXACTA de apply_hard_filters original (líneas 274-336).
        
        Args:
            candidate_ids: Lista de IDs de candidatos
            filters_config: Objeto HardFilters con education_min, language_required, etc.
            
        Returns:
            Lista filtrada de candidate_ids
        """
        if not candidate_ids:
            return candidate_ids
        
        keep = set(candidate_ids)
        
        # ---------- education_min ----------
        education_min = getattr(filters_config, 'education_min', None)
        if education_min:
            req_rank = EDU_RANK_UI.get(education_min.strip().lower())
            if req_rank:
                edu_doc_ids = [f"{c}::edu" for c in candidate_ids]
                metas = self.fetch_metas_by_doc_ids("education", edu_doc_ids)
                
                ok = set()
                for c in candidate_ids:
                    meta = metas.get(f"{c}::edu", {})
                    rank = meta.get("edu_rank")
                    try:
                        rank = int(rank)
                    except Exception:
                        rank = 0
                    if rank >= req_rank:
                        ok.add(c)
                keep = keep.intersection(ok)
        
        # ---------- language_required (+ min level) ----------
        language_required = getattr(filters_config, 'language_required', None)
        if language_required:
            canon = LANG_UI_TO_CANON.get(language_required.strip().lower())
            if canon:
                req_lvl_rank = None
                language_min_level = getattr(filters_config, 'language_min_level', None)
                if language_min_level:
                    req_lvl_rank = CEFR_RANK.get(language_min_level.strip().upper())
                
                lang_doc_ids = [f"{c}::lang" for c in list(keep)]
                metas = self.fetch_metas_by_doc_ids("language", lang_doc_ids)
                
                ok = set()
                for c in list(keep):
                    meta = metas.get(f"{c}::lang", {})
                    has_flag = meta.get(f"has_{canon}", 0)
                    try:
                        has_flag = int(has_flag)
                    except Exception:
                        has_flag = 0
                    
                    if has_flag != 1:
                        continue
                    
                    if req_lvl_rank is not None:
                        lvl_rank = meta.get(f"lvl_{canon}_rank", 0)
                        try:
                            lvl_rank = int(lvl_rank)
                        except Exception:
                            lvl_rank = 0
                        if lvl_rank < req_lvl_rank:
                            continue
                    
                    ok.add(c)
                
                keep = keep.intersection(ok)
        
        return [c for c in candidate_ids if c in keep]
