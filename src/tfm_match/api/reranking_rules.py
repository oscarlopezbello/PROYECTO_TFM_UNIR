"""
Rule-based reranking - Ajustes post-scoring por reglas discretas.

Contiene la misma lógica de reranking que _run_match() en main.py,
extraída para reutilización por el flujo orquestado con tools por dimensión.
"""

import re
import math
import unicodedata
from typing import List, Dict, Any


def apply_reranking_rules(
    scored: List[Dict[str, Any]],
    q_skills: str,
    q_experience: str,
    q_education: str,
    q_language: str,
    q_sector: str,
    q_city: str,
    weight_map: Dict[str, int],
    top_k: int,
    chroma_collections: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Aplica reglas de reranking rule-based sobre los candidatos ya ranqueados.
    Misma lógica exacta que _run_match() lines 460-1244 en main.py.

    Args:
        scored: Lista de candidatos con affinity, breakdown, skills, education, etc.
        q_skills: Texto de búsqueda de skills
        q_experience: Texto de búsqueda de experiencia (ej: "0-1", "1-3")
        q_education: Texto de búsqueda de educación
        q_language: Texto de búsqueda de idioma
        q_sector: Texto de búsqueda de sector
        q_city: Texto de búsqueda de ciudad
        weight_map: Pesos por dimensión
        top_k: Número de candidatos a retornar
        chroma_collections: Colecciones ChromaDB (necesaria para metadata de idioma)

    Returns:
        Lista de candidatos reranqueados
    """

    # ================================================================
    # 7.05) Similitud de idioma (rule-based)
    # ================================================================
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
    KW_RANK = {"basic": 2, "intermediate": 4, "advanced": 5, "fluent": 6, "native": 7, "none": 0}

    def _parse_lang_reqs(q: str) -> List[Dict[str, Any]]:
        if not q:
            return []
        txt = str(q).strip()
        parts = [p.strip() for p in re.split(r"[;\n,|]+", txt) if p.strip()]
        reqs: Dict[str, int] = {}
        for p in parts:
            m = re.match(r"^\s*([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\s*[: ]?\s*([A-Za-z0-9]+)?\s*$", p)
            lang_raw = m.group(1).strip().lower() if m else ""
            lvl_raw = (m.group(2) or "").strip() if m else ""
            if not lang_raw:
                continue
            canon = LANG_UI_TO_CANON.get(lang_raw, lang_raw)
            rr = 0
            if lvl_raw:
                lv_up = lvl_raw.upper()
                if lv_up in CEFR_RANK:
                    rr = CEFR_RANK[lv_up]
                else:
                    rr = KW_RANK.get(lvl_raw.lower(), 0)
            reqs[canon] = max(reqs.get(canon, 0), rr)
        return [{"language": k, "req_rank": v} for k, v in reqs.items()]

    if q_language and chroma_collections.get("language"):
        col_lang = chroma_collections["language"]
        w_lang = int(weight_map.get("language", 0) or 0)
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)

        reqs = _parse_lang_reqs(q_language)
        if reqs:
            doc_ids = [f"{r.get('candidate_id')}::lang" for r in scored if r.get("candidate_id") is not None]
            metas_by_id: Dict[str, Dict[str, Any]] = {}
            try:
                res_get = col_lang.get(ids=doc_ids, include=["metadatas"])
                ids_out = res_get.get("ids", [])
                metas_out = res_get.get("metadatas", [])
                for i in range(len(ids_out)):
                    metas_by_id[str(ids_out[i])] = metas_out[i] or {}
            except Exception:
                metas_by_id = {}

            for r in scored:
                cid = r.get("candidate_id")
                if cid is None:
                    continue
                doc_id = f"{cid}::lang"
                meta = metas_by_id.get(doc_id, {}) or {}

                scores: List[float] = []
                for req in reqs:
                    canon = req["language"]
                    req_rank = int(req["req_rank"] or 0)
                    has_flag = meta.get(f"has_{canon}", 0)
                    try:
                        has_flag = int(has_flag)
                    except Exception:
                        has_flag = 0
                    if has_flag != 1:
                        scores.append(0.0)
                        continue

                    if req_rank <= 0:
                        scores.append(1.0)
                        continue

                    cand_rank = meta.get(f"lvl_{canon}_rank", 0)
                    try:
                        cand_rank = int(cand_rank)
                    except Exception:
                        cand_rank = 0

                    if cand_rank >= req_rank:
                        scores.append(1.0)
                    elif cand_rank == req_rank - 1:
                        scores.append(0.5)
                    else:
                        scores.append(0.0)

                lang_sim = float(sum(scores) / len(scores)) if scores else 0.0

                bd = r.get("breakdown") or {}
                cosine_original = float((bd.get("language") or {}).get("score_0_1", 0.0) or 0.0)
                bd["language"] = {
                    "score_0_1": round(float(lang_sim), 4),
                    "score_pct": round(float(lang_sim) * 100, 2),
                    "weight": w_lang,
                    "contribution": round(float(lang_sim) * float(w_lang), 4),
                    "rule": "rule_based_language (>=req=1; 1-below=0.5; else=0)",
                    "requirements": reqs,
                    "cosine_sim_original": round(float(cosine_original), 4),
                    "doc_id": doc_id,
                }
                r["breakdown"] = bd

                if w_lang > 0 and total_weight > 0:
                    weighted_sum = 0.0
                    for dim, ww in weight_map.items():
                        if ww <= 0:
                            continue
                        if dim == "language":
                            dim_score = float(lang_sim)
                        else:
                            dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                        weighted_sum += dim_score * float(ww)
                    r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

            scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
            scored = scored[:top_k]

    # ================================================================
    # 7.1) Similitud de skills (fallback léxico)
    # ================================================================
    STOPWORDS_SKILLS = {
        "y", "de", "la", "el", "los", "las", "en", "con", "sin", "para", "por", "a", "al", "del", "o", "u",
        "un", "una", "unos", "unas",
    }

    def _skills_tokens(text_val: str) -> List[str]:
        if not text_val:
            return []
        t = str(text_val)
        t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)
        t = unicodedata.normalize("NFKD", t)
        t = "".join([c for c in t if not unicodedata.combining(c)])
        t = t.lower()
        t = re.sub(r"[,\|\;/•·\-]+", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        toks = [x for x in t.split(" ") if x and x not in STOPWORDS_SKILLS]
        return toks

    def _office_suite_present(cand_tokens: List[str]) -> bool:
        office_markers = {
            "office", "ofimatica", "ofimatica", "ofimaticas", "microsoftoffice",
            "excel", "microsoftexcel",
            "word", "microsoftword",
            "powerpoint", "microsoftpowerpoint",
            "outlook", "microsoftoutlook",
        }
        for ct in cand_tokens:
            if ct in office_markers:
                return True
            if ct.startswith("microsoft") and any(x in ct for x in ("excel", "word", "powerpoint", "outlook", "office")):
                return True
        return False

    def _skills_lex_sim(query_text: str, cand_text: str) -> Dict[str, Any]:
        q_toks = _skills_tokens(query_text)
        c_toks = _skills_tokens(cand_text)
        if not q_toks:
            return {"sim": 0.0, "matched": [], "missing": [], "query_tokens": [], "cand_tokens": []}

        seen = set()
        q_uniq: List[str] = []
        for t in q_toks:
            if t not in seen:
                q_uniq.append(t)
                seen.add(t)

        c_set = set(c_toks)
        matched: List[str] = []
        missing: List[str] = []

        for qt in q_uniq:
            ok = False
            if qt in ("office", "ofimatica", "ofimatica", "microsoftoffice"):
                ok = _office_suite_present(c_toks)
            else:
                if qt in c_set:
                    ok = True
                else:
                    special_sub = qt in ("excel", "word", "powerpoint", "outlook", "atencion", "cliente")
                    if len(qt) >= 5 or special_sub:
                        for ct in c_toks:
                            if qt in ct:
                                ok = True
                                break

            if ok:
                matched.append(qt)
            else:
                missing.append(qt)

        sim = float(len(matched)) / float(len(q_uniq)) if q_uniq else 0.0
        return {"sim": sim, "matched": matched, "missing": missing, "query_tokens": q_uniq, "cand_tokens": c_toks[:40]}

    if q_skills:
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_sk = int(weight_map.get("skills", 0) or 0)
        for r in scored:
            cand_sk_txt = r.get("skills", "") or ""
            bd = r.get("breakdown") or {}
            cos_sim = float((bd.get("skills") or {}).get("score_0_1", 0.0) or 0.0)
            lex = _skills_lex_sim(q_skills, cand_sk_txt)
            sk_sim = float(lex["sim"])

            bd["skills"] = {
                "score_0_1": round(sk_sim, 4),
                "score_pct": round(sk_sim * 100, 2),
                "weight": w_sk,
                "contribution": round(sk_sim * float(w_sk), 4),
                "rule": "lexical_token_match (supports microsoftexcel/excel and office suite)",
                "matched": lex["matched"],
                "missing": lex["missing"],
                "cosine_sim_original": round(cos_sim, 4),
            }
            r["breakdown"] = bd

            if w_sk > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "skills":
                        dim_score = sk_sim
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[:top_k]

    # ================================================================
    # 7.15) Similitud de sector (fallback léxico)
    # ================================================================
    STOPWORDS_SECTOR = {
        "y", "de", "la", "el", "los", "las", "en", "con", "sin", "para", "por", "a", "al", "del", "o", "u",
        "un", "una", "unos", "unas",
        "tecnologia", "tecnologia", "tecnica", "tecnico", "profesional", "industrial",
    }

    def _sector_tokens(text_val: str) -> List[str]:
        if not text_val:
            return []
        t = str(text_val)
        t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)
        t = unicodedata.normalize("NFKD", t)
        t = "".join([c for c in t if not unicodedata.combining(c)])
        t = t.lower()
        t = re.sub(r"[,\|\;/•·\-]+", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        toks = [x for x in t.split(" ") if x and x not in STOPWORDS_SECTOR]
        return toks

    def _sector_lex_sim(query_text: str, cand_text: str) -> Dict[str, Any]:
        q_toks = _sector_tokens(query_text)
        c_toks = _sector_tokens(cand_text)
        if not q_toks:
            return {"sim": 0.0, "matched": [], "missing": [], "query_tokens": [], "cand_tokens": []}

        seen = set()
        q_uniq: List[str] = []
        for t in q_toks:
            if t not in seen:
                q_uniq.append(t)
                seen.add(t)

        matched: List[str] = []
        missing: List[str] = []
        for qt in q_uniq:
            ok = False
            if qt in c_toks:
                ok = True
            elif len(qt) >= 4:
                for ct in c_toks:
                    if qt in ct or ct in qt:
                        ok = True
                        break
            if ok:
                matched.append(qt)
            else:
                missing.append(qt)

        sim = float(len(matched)) / float(len(q_uniq)) if q_uniq else 0.0
        return {"sim": sim, "matched": matched, "missing": missing, "query_tokens": q_uniq, "cand_tokens": c_toks[:40]}

    if q_sector:
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_sec = int(weight_map.get("sector", 0) or 0)
        for r in scored:
            cand_sec_txt = r.get("sector", "") or ""
            bd = r.get("breakdown") or {}
            cos_sim = float((bd.get("sector") or {}).get("score_0_1", 0.0) or 0.0)
            lex = _sector_lex_sim(q_sector, cand_sec_txt)
            sec_sim = float(lex["sim"])

            bd["sector"] = {
                "score_0_1": round(sec_sim, 4),
                "score_pct": round(sec_sim * 100, 2),
                "weight": w_sec,
                "contribution": round(sec_sim * float(w_sec), 4),
                "rule": "lexical_token_match (CamelCase + accents normalization)",
                "matched": lex["matched"],
                "missing": lex["missing"],
                "cosine_sim_original": round(cos_sim, 4),
            }
            r["breakdown"] = bd

            if w_sec > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "sector":
                        dim_score = sec_sim
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[:top_k]

    # ================================================================
    # 7.2) Similitud de experiencia (rule-based por rango)
    # ================================================================
    RANGE_MAP_MONTHS = {
        "0-1": (0, 12),
        "1-3": (13, 36),
        "3-5": (37, 60),
        "5+": (61, 10_000),
    }

    def _parse_duration_piece_to_months(piece: str) -> int:
        if not piece:
            return 0
        t = str(piece).strip().lower()
        t = (
            t.replace("años", "anos")
            .replace("año", "ano")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        m = re.search(r"\b(\d+)\s*anos?\s*y\s*(\d+)\s*mes(?:es)?\b", t)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        m = re.search(r"\b(\d+)\s*anos?\b", t)
        if m:
            return int(m.group(1)) * 12
        m = re.search(r"\b(\d+)\s*mes(?:es)?\b", t)
        if m:
            return int(m.group(1))
        return 0

    def _total_months_from_experience_text(exp_text: str) -> int:
        if not exp_text:
            return 0
        parts = [p.strip() for p in re.split(r"[;,\n|/]+", str(exp_text)) if p.strip()]
        return sum(_parse_duration_piece_to_months(p) for p in parts)

    EXP_RANGE_CFG_YEARS = {
        "0-1": {"m": 0.0, "upper": 1.0, "penalize_after": 2.0},
        "1-3": {"m": 1.0, "upper": 3.0, "penalize_after": 5.0},
        "3-5": {"m": 3.0, "upper": 5.0, "penalize_after": 8.0},
        "5+": {"m": 5.0, "upper": 5.0, "penalize_after": 12.0},
    }

    def _f_years_bucket(x_years: float, m: float, upper: float, penalize_after: float, k: float) -> float:
        if x_years < 0:
            return 0.0
        if x_years < m:
            if m <= 0:
                return 1.0
            return max(0.0, min(1.0, x_years / m))
        if x_years <= penalize_after:
            return 1.0
        return float(math.exp(-float(k) * (x_years - penalize_after)))

    def _f_years_5plus(x_years: float, penalize_after: float, k: float) -> float:
        if x_years < 0:
            return 0.0
        if x_years < 5.0:
            return max(0.0, min(1.0, x_years / 5.0))
        if x_years <= penalize_after:
            return 1.0
        return float(math.exp(-float(k) * (x_years - penalize_after)))

    if q_experience and q_experience.strip() in RANGE_MAP_MONTHS:
        rng = q_experience.strip()
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_exp = int(weight_map.get("experience", 0) or 0)
        k_exp = 0.3
        cfg = EXP_RANGE_CFG_YEARS.get(rng, {"m": 0.0, "upper": 1.0, "penalize_after": 2.0})
        m_req = float(cfg["m"])
        upper_req = float(cfg["upper"])
        p_req = float(cfg["penalize_after"])

        for r in scored:
            cand_exp_txt = r.get("experience", "") or ""
            total_m = _total_months_from_experience_text(cand_exp_txt)
            x_years = float(total_m) / 12.0 if total_m and total_m > 0 else 0.0
            if rng == "5+":
                exp_score = _f_years_5plus(x_years=x_years, penalize_after=p_req, k=k_exp)
            else:
                exp_score = _f_years_bucket(x_years=x_years, m=m_req, upper=upper_req, penalize_after=p_req, k=k_exp)

            bd = r.get("breakdown") or {}
            cos_sim = float((bd.get("experience") or {}).get("score_0_1", 0.0) or 0.0)
            bd["experience"] = {
                "score_0_1": round(float(exp_score), 4),
                "score_pct": round(float(exp_score) * 100, 2),
                "weight": w_exp,
                "contribution": round(float(exp_score) * float(w_exp), 4),
                "rule": "bucket w/ penalize_after: x<m => x/m; x in [m,upper] => 1; x in (upper,P] => 1; x>P => exp(-k(x-P))",
                "query_experience": rng,
                "candidate_experience": cand_exp_txt,
                "total_months": int(total_m),
                "x_years": round(float(x_years), 4),
                "m": m_req,
                "upper": upper_req,
                "penalize_after": p_req,
                "k": k_exp,
                "cosine_sim_tiebreak": round(float(cos_sim), 4),
            }
            r["breakdown"] = bd

            if w_exp > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "experience":
                        dim_score = float(exp_score)
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        def _exp_sort_key(item: dict):
            bd = item.get("breakdown") or {}
            exp_bd = bd.get("experience") or {}
            score = float(exp_bd.get("score_0_1", 0.0) or 0.0)
            tie = float(exp_bd.get("cosine_sim_tiebreak", 0.0) or 0.0)
            return (score, tie)

        scored.sort(key=_exp_sort_key, reverse=True)
        scored = scored[:top_k]

    # ================================================================
    # 7.25) Similitud de educación (rule-based)
    # ================================================================
    EDU_RANK_UI = {
        "none": 0,
        "bachiller": 1,
        "tecnico": 2,
        "técnico": 2,
        "tecnologo": 3,
        "tecnólogo": 3,
        "profesional": 4,
        "posgrado": 5,
    }

    def _edu_rank_from_text(text_val: str) -> int:
        if not text_val:
            return 0
        t = str(text_val)
        t = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", t)
        t = t.lower()
        t = (
            t.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )
        t = re.sub(r"\s+", " ", t).strip()

        if re.search(r"\b(doctorado|phd|d\.?phil|maestria|master|magister|m\.?sc|mba|especializacion|posgrado)\b", t):
            return 5
        if re.search(r"\b(profesional|pregrado|grado|universitario|ingenier[ia]|ingeniero|licenciad[oa]|administrador|abogad[oa]|contador|economista|psicolog[oa])\b", t):
            return 4
        if re.search(r"\b(tecnolog[oa]|tecnologia)\b", t):
            return 3
        if re.search(r"\b(tecnic[oa]|tecnico\s+profesional|tecnico\s+laboral)\b", t):
            return 2
        if re.search(r"\b(bachiller|secundaria|media|bachillerato)\b", t):
            return 1
        return 0

    if q_education:
        q_edu_rank = _edu_rank_from_text(q_education)
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        w_edu = int(weight_map.get("education", 0) or 0)

        for r in scored:
            cand_edu_txt = r.get("education", "") or ""
            c_rank = _edu_rank_from_text(cand_edu_txt)
            edu_sim = 0.0
            if q_edu_rank > 0 and c_rank > 0:
                if c_rank >= q_edu_rank:
                    edu_sim = 1.0
                elif c_rank == q_edu_rank - 1:
                    edu_sim = 0.5
                else:
                    edu_sim = 0.0

            bd = r.get("breakdown") or {}
            bd["education"] = {
                "score_0_1": round(float(edu_sim), 4),
                "score_pct": round(float(edu_sim) * 100, 2),
                "weight": w_edu,
                "contribution": round(float(edu_sim) * float(w_edu), 4),
                "rule": ">=req:1.0; 1-below:0.5; else:0.0",
                "query_education": q_education,
                "candidate_education": cand_edu_txt,
                "req_rank": q_edu_rank,
                "cand_rank": c_rank,
            }
            r["breakdown"] = bd

            if w_edu > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    if dim == "education":
                        dim_score = float(edu_sim)
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[:top_k]

    # ================================================================
    # 7.5) Similitud de ubicación (rule-based)
    # ================================================================
    def _split_location(loc: str):
        if not loc:
            return "", ""
        t = str(loc).strip()
        parts = [p.strip() for p in re.split(r"[,;\-\|/]+", t) if p.strip()]
        if not parts:
            return "", ""

        BASE_CO_DEPARTMENTS = [
            "Amazonas", "Antioquia", "Arauca", "Atlántico", "Bolívar", "Boyacá",
            "Caldas", "Caquetá", "Casanare", "Cauca", "Cesar", "Chocó", "Córdoba",
            "Cundinamarca", "Guainía", "Guaviare", "Huila", "La Guajira", "Magdalena",
            "Meta", "Nariño", "Norte de Santander", "Putumayo", "Quindío", "Risaralda",
            "San Andrés y Providencia", "Santander", "Sucre", "Tolima", "Valle del Cauca",
            "Vaupés", "Vichada",
            "Bogotá", "Bogotá D.C.", "Bogota DC", "Distrito Capital",
        ]

        def _strip_accents(s: str) -> str:
            s2 = unicodedata.normalize("NFKD", s)
            return "".join([c for c in s2 if not unicodedata.combining(c)])

        def _canon_key(s: str) -> str:
            s = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", str(s))
            s = _strip_accents(s).lower()
            s = re.sub(r"[^a-z\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _nospace_key(s: str) -> str:
            return _canon_key(s).replace(" ", "")

        CO_DEPT_KEYS = set()
        for dname in BASE_CO_DEPARTMENTS:
            CO_DEPT_KEYS.add(_canon_key(dname))
            CO_DEPT_KEYS.add(_nospace_key(dname))

        if len(parts) == 1:
            one = parts[0].strip()
            if one.lower() in ("no aplica", "na", "n/a", "0"):
                return "", ""
            one_c = _canon_key(one)
            if one_c in CO_DEPT_KEYS or one_c.replace(" ", "") in CO_DEPT_KEYS:
                return "", one_c
            return one_c, ""

        a = parts[0].strip().lower()
        b = parts[-1].strip().lower()

        a_n = _canon_key(a)
        b_n = _canon_key(b)
        a_ns = a_n.replace(" ", "")
        b_ns = b_n.replace(" ", "")

        if (a_n in CO_DEPT_KEYS or a_ns in CO_DEPT_KEYS) and not (b_n in CO_DEPT_KEYS or b_ns in CO_DEPT_KEYS):
            depto = a_n
            municipio = b_n
            return municipio, depto

        if (b_n in CO_DEPT_KEYS or b_ns in CO_DEPT_KEYS) and not (a_n in CO_DEPT_KEYS or a_ns in CO_DEPT_KEYS):
            depto = b_n
            municipio = a_n
            return municipio, depto

        municipio = a_n
        depto = b_n
        return municipio, depto

    if q_city:
        q_mun, q_depto = _split_location(q_city)
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)

        for r in scored:
            cand_city = r.get("city", "") or ""
            c_mun, c_depto = _split_location(cand_city)

            city_sim = 0.0
            if q_mun and c_mun and q_mun == c_mun:
                city_sim = 1.0
            elif q_depto and c_depto and q_depto == c_depto:
                city_sim = 0.85

            bd = r.get("breakdown") or {}
            w_city = int(weight_map.get("city", 0) or 0)
            bd["city"] = {
                "score_0_1": round(float(city_sim), 4),
                "score_pct": round(float(city_sim) * 100, 2),
                "weight": w_city,
                "contribution": round(float(city_sim) * float(w_city), 4),
                "rule": "same_muni=1.0; same_depto=0.85; else=0.0",
                "query_city": q_city,
                "candidate_city": cand_city,
            }
            r["breakdown"] = bd

            if w_city > 0 and total_weight > 0:
                weighted_sum = 0.0
                for dim, ww in weight_map.items():
                    if ww <= 0:
                        continue
                    dim_score = 0.0
                    if dim == "city":
                        dim_score = float(city_sim)
                    else:
                        dim_score = float((bd.get(dim) or {}).get("score_0_1", 0.0) or 0.0)
                    weighted_sum += dim_score * float(ww)
                r["affinity"] = round((weighted_sum / float(total_weight)) * 100, 2)

        scored.sort(key=lambda x: x.get("affinity", 0), reverse=True)
        scored = scored[:top_k]

    return scored
