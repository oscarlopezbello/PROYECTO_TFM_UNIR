"""
Persistence Manager - Maneja guardado y lectura de MySQL.
Código extraído de api/main.py (líneas 342-380 y 475-607).
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import Engine, text


def safe_int(s: str) -> Optional[int]:
    """
    Convierte string a int de forma segura.
    LÓGICA EXACTA de líneas 136-141 de main.py original.
    """
    try:
        return int(str(s))
    except Exception:
        return None


# Máximo de cargos a mostrar en strings consolidados (UI chips). Coincide con job_name1..5.
MAX_JOB_TITLES_DISPLAY = 5


class PersistenceManager:
    """Maneja persistencia en MySQL."""
    
    def __init__(self, engine: Engine, candidates_table: str = "candidates_prepared"):
        """
        Args:
            engine: SQLAlchemy engine
            candidates_table: Nombre de la tabla de candidatos
        """
        self.engine = engine
        self.candidates_table = candidates_table
    
    def match_city_direct(self, city_query: str) -> List[Dict[str, Any]]:
        """
        Match directo de ciudad en MySQL (sin embeddings).
        
        Args:
            city_query: Ciudad buscada (ej: "Bogotá")
            
        Returns:
            Lista de hits con similitud 1.0 para candidatos que coincidan
        """
        if not self.engine or not city_query.strip():
            return []
        
        # Normalizar búsqueda (case-insensitive, trim)
        city_normalized = city_query.strip().lower()
        
        sql = text(f"""
            SELECT DISTINCT id_candidate, location
            FROM {self.candidates_table}
            WHERE LOWER(TRIM(location)) LIKE :city_pattern
            LIMIT 1000
        """)
        
        df = pd.read_sql(sql, self.engine, params={"city_pattern": f"%{city_normalized}%"})
        
        hits = []
        for _, row in df.iterrows():
            hits.append({
                "id_candidate": str(row["id_candidate"]),
                "dist": 0.0,  # Distancia 0 = match perfecto
                "sim": 1.0,   # Similitud 100%
                "meta": {"dimension": "city", "location": row.get("location", "")},
                "doc": row.get("location", ""),
            })
        
        return hits
    
    def match_experience_direct(self, experience_query: str) -> List[Dict[str, Any]]:
        """
        Match directo de experiencia en MySQL con lógica de rangos.
        
        Args:
            experience_query: Rango buscado (ej: "0-1", "1-3", "3-5", "5+")
            
        Returns:
            Lista de hits con similitud basada en si cae dentro del rango
        """
        if not self.engine or not experience_query.strip():
            return []
        
        # Mapeo de rangos a meses
        range_map = {
            "0-1": (0, 12),
            "1-3": (13, 36),
            "3-5": (37, 60),
            "5+": (61, 999)
        }
        
        exp_range = range_map.get(experience_query.strip())
        if not exp_range:
            return []
        
        min_months, max_months = exp_range
        
        # Obtener todos los candidatos con sus duraciones
        sql = text(f"""
            SELECT 
                id_candidate,
                job_duration1, job_duration2, job_duration3, job_duration4, job_duration5
            FROM {self.candidates_table}
            WHERE job_duration1 IS NOT NULL OR job_duration2 IS NOT NULL
            LIMIT 2000
        """)
        
        df = pd.read_sql(sql, self.engine)
        
        hits = []
        for _, row in df.iterrows():
            # Consolidar todas las duraciones y convertir a meses
            total_months = 0
            durations_text = []
            
            for i in range(1, 6):
                duration = row.get(f"job_duration{i}")
                if pd.notna(duration):
                    duration_str = str(duration).strip().lower()
                    durations_text.append(duration_str)
                    
                    # Parsear duración a meses
                    months = self._parse_duration_to_months(duration_str)
                    total_months += months
            
            # Calcular similitud basada en si cae dentro del rango
            if min_months <= total_months <= max_months:
                # Match perfecto si está en el rango
                similarity = 1.0
            elif total_months < min_months:
                # Penalización si tiene menos experiencia
                similarity = max(0.5, total_months / min_months) if min_months > 0 else 0.5
            else:
                # Penalización menor si tiene más experiencia (sobre-calificado)
                excess = total_months - max_months
                similarity = max(0.7, 1.0 - (excess / 60.0))  # Penalización gradual
            
            hits.append({
                "id_candidate": str(row["id_candidate"]),
                "dist": 1.0 - similarity,
                "sim": similarity,
                "meta": {"dimension": "experience", "total_months": total_months},
                "doc": "; ".join(durations_text[:3]) if durations_text else "Sin experiencia",
            })
        
        return hits
    
    def match_job_title_direct(self, job_title_query: str) -> List[Dict[str, Any]]:
        """
        Match directo de job_title en MySQL con similitud de texto.
        
        Args:
            job_title_query: Cargo buscado (ej: "auxiliar de bodega")
            
        Returns:
            Lista de hits con similitud basada en coincidencia de texto
        """
        if not self.engine or not job_title_query.strip():
            return []
        
        from difflib import SequenceMatcher
        
        # Normalizar búsqueda
        query_normalized = job_title_query.strip().lower()
        
        # Obtener candidatos con job_name
        sql = text(f"""
            SELECT 
                id_candidate,
                job_name1, job_name2, job_name3, job_name4, job_name5
            FROM {self.candidates_table}
            WHERE job_name1 IS NOT NULL
            LIMIT 2000
        """)
        
        df = pd.read_sql(sql, self.engine)
        
        hits = []
        for _, row in df.iterrows():
            # Consolidar job titles limpios
            job_titles = []
            for i in range(1, 6):
                jname = row.get(f"job_name{i}")
                if pd.notna(jname):
                    jname_str = str(jname).strip()
                    if jname_str and jname_str not in ["0", "nan", "None", "null"]:
                        job_titles.append(jname_str)
            
            if not job_titles:
                continue
            
            # Calcular similitud máxima contra cualquiera de los job_titles
            max_similarity = 0.0
            for job_title in job_titles:
                job_title_norm = job_title.lower()
                
                # Usar SequenceMatcher para similitud de texto
                similarity = SequenceMatcher(None, query_normalized, job_title_norm).ratio()
                max_similarity = max(max_similarity, similarity)
            
            # Solo incluir si tiene al menos 50% de similitud
            if max_similarity >= 0.5:
                hits.append({
                    "id_candidate": str(row["id_candidate"]),
                    "dist": 1.0 - max_similarity,
                    "sim": max_similarity,
                    "meta": {"dimension": "job_title"},
                    "doc": "; ".join(job_titles[:MAX_JOB_TITLES_DISPLAY]),
                })
        
        return hits
    
    @staticmethod
    def _parse_duration_to_months(duration_str: str) -> int:
        """
        Convierte texto de duración a meses.
        Ej: "2 años" -> 24, "11 meses" -> 11, "1 año 6 meses" -> 18
        """
        import re
        
        duration_str = duration_str.lower()
        total_months = 0
        
        # Buscar años
        years_match = re.search(r'(\d+)\s*a[ñn]os?', duration_str)
        if years_match:
            total_months += int(years_match.group(1)) * 12
        
        # Buscar meses
        months_match = re.search(r'(\d+)\s*mes(?:es)?', duration_str)
        if months_match:
            total_months += int(months_match.group(1))
        
        return total_months
    
    def fetch_candidates_from_mysql(self, candidate_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Trae datos de candidatos desde MySQL.
        LÓGICA EXACTA de fetch_candidates_from_mysql original (líneas 342-380).
        
        Args:
            candidate_ids: Lista de IDs de candidatos
            
        Returns:
            Dict de candidate_id -> datos del candidato
        """
        if self.engine is None:
            return {}
        
        ints = [safe_int(c) for c in candidate_ids]
        ints = [x for x in ints if x is not None]
        if not ints:
            return {}
        
        # placeholders :id0, :id1, ...
        params = {f"id{i}": v for i, v in enumerate(ints)}
        in_clause = ", ".join([f":id{i}" for i in range(len(ints))])
        
        sql = text(f"""
            SELECT
                id_candidate,
                skills,
                brief_description,
                profile_text,
                last_grade,
                location,
                job_name1, job_name2, job_name3, job_name4, job_name5,
                job_duration1, job_duration2, job_duration3, job_duration4, job_duration5,
                study_area1, study_area2, study_area3,
                language1, language_level1, language2, language_level2
            FROM {self.candidates_table}
            WHERE id_candidate IN ({in_clause})
        """)
        
        df = pd.read_sql(sql, self.engine, params=params)
        
        out = {}
        for _, r in df.iterrows():
            cid = str(r["id_candidate"])
            
            # Consolidar job_title desde job_name1-5 (SOLO valores válidos)
            job_titles = []
            for i in range(1, 6):
                jname = r.get(f"job_name{i}")
                if pd.notna(jname):
                    jname_str = str(jname).strip()
                    # Filtrar valores vacíos, "0", "nan", etc.
                    if jname_str and jname_str not in ["0", "nan", "None", "null"]:
                        job_titles.append(jname_str)
            job_title_text = "; ".join(job_titles[:MAX_JOB_TITLES_DISPLAY]) if job_titles else ""
            
            # Consolidar sector desde study_area (SOLO valores válidos)
            sectors = []
            for i in range(1, 4):
                sarea = r.get(f"study_area{i}")
                if pd.notna(sarea):
                    sarea_str = str(sarea).strip()
                    # Filtrar valores vacíos, "0", "nan", etc.
                    if sarea_str and sarea_str not in ["0", "nan", "None", "null"]:
                        sectors.append(sarea_str)
            sector_text = "; ".join(sectors[:2]) if sectors else ""
            
            # Educación
            education_text = str(r.get("last_grade") or "")
            
            # Idiomas (SOLO valores válidos)
            languages = []
            for i in range(1, 3):
                lang = r.get(f"language{i}")
                level = r.get(f"language_level{i}")
                if pd.notna(lang):
                    lang_str = str(lang).strip()
                    # Filtrar valores vacíos, "0", "nan", etc.
                    if lang_str and lang_str not in ["0", "nan", "None", "null"]:
                        if pd.notna(level):
                            level_str = str(level).strip()
                            if level_str and level_str not in ["0", "nan", "None", "null"]:
                                lang_str += f" ({level_str})"
                        languages.append(lang_str)
            language_text = "; ".join(languages) if languages else ""
            
            # Experiencia: consolidar job_duration1-5 (SOLO valores válidos)
            durations = []
            for i in range(1, 6):
                duration = r.get(f"job_duration{i}")
                if pd.notna(duration):
                    duration_str = str(duration).strip()
                    # Filtrar valores vacíos, "0", "nan", etc.
                    if duration_str and duration_str not in ["0", "nan", "None", "null"]:
                        durations.append(duration_str)
            
            # Si tiene duraciones, mostrar las primeras 3
            if durations:
                experience_text = "; ".join(durations[:3])
            else:
                experience_text = "Sin información"
            
            # Ciudad
            city_text = str(r.get("location") or "")
            
            out[cid] = {
                "skills": r.get("skills") or "",
                "brief_description": r.get("brief_description") or "",
                "profile_text": r.get("profile_text") or "",
                "job_title": job_title_text,
                "sector": sector_text,
                "education": education_text,
                "experience": experience_text,
                "language": language_text,
                "city": city_text,
            }
        return out
    
    def save_job_request_and_results(
        self,
        query_payload: Dict[str, Any],
        weights: Dict[str, int],
        top_k: int,
        scored_results: List[Dict[str, Any]]
    ) -> int:
        """
        Guarda job_request y sus resultados en MySQL.
        LÓGICA EXACTA de líneas 475-607 de main.py original.
        
        Args:
            query_payload: Datos de la consulta
            weights: Pesos usados
            top_k: Top K solicitado
            scored_results: Resultados ranqueados
            
        Returns:
            job_request_id generado
        """
        # Guardar job_request
        with self.engine.begin() as conn:
            r = conn.execute(
                text("""
                    INSERT INTO job_requests (query_text, top_k, weights)
                    VALUES (:query_text, :top_k, :weights)
                """),
                {
                    "query_text": json.dumps(query_payload, ensure_ascii=False),
                    "top_k": top_k,
                    "weights": json.dumps(weights, ensure_ascii=False)
                }
            )
            job_request_id = r.lastrowid
        
        # Guardar resultados
        with self.engine.begin() as conn:
            for rank, r in enumerate(scored_results, start=1):
                cid_int = safe_int(r["candidate_id"])
                if cid_int is None:
                    # evita crashear por IDs no numéricos
                    continue
                conn.execute(
                    text("""
                        INSERT INTO job_request_results
                        (id_request, candidate_id, affinity, rank_position)
                        VALUES (:id_request, :candidate_id, :affinity, :rank)
                    """),
                    {
                        "id_request": job_request_id,
                        "candidate_id": cid_int,
                        "affinity": r["affinity"],
                        "rank": rank
                    }
                )
        
        return job_request_id
    
    def save_match_execution(
        self,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> Optional[int]:
        """
        Guarda una ejecución de match en la tabla match_executions.
        Usado para trazabilidad cuando el match es orquestado por LLM.
        
        Args:
            request_payload: Payload completo enviado (skills, weights, top_k, etc.)
            response_payload: Response completo (job_request_id, query, results, etc.)
            
        Returns:
            id_execution generado, o None si falla
        """
        try:
            top_k = request_payload.get("top_k") or response_payload.get("query", {}).get("top_k")
            results = response_payload.get("results", [])
            num_candidates = len(results)
            job_request_id = response_payload.get("job_request_id")
            weights_used = (request_payload.get("weights") or 
                           response_payload.get("query", {}).get("weights"))
            
            with self.engine.begin() as conn:
                r = conn.execute(
                    text("""
                        INSERT INTO match_executions
                        (request_payload, response_payload, top_k, num_candidates, job_request_id, weights_used)
                        VALUES (:request_payload, :response_payload, :top_k, :num_candidates, :job_request_id, :weights_used)
                    """),
                    {
                        "request_payload": json.dumps(request_payload, ensure_ascii=False),
                        "response_payload": json.dumps(response_payload, ensure_ascii=False),
                        "top_k": top_k,
                        "num_candidates": num_candidates,
                        "job_request_id": job_request_id,
                        "weights_used": json.dumps(weights_used or {}, ensure_ascii=False),
                    }
                )
                return r.lastrowid
        except Exception:
            return None
    
    def enrich_candidates(self, scored_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriquece resultados con datos de MySQL.
        LÓGICA EXACTA de líneas 581-587 de main.py original.
        
        Args:
            scored_results: Lista de resultados a enriquecer
            
        Returns:
            Lista enriquecida con todos los campos de dimensiones
        """
        candidate_ids = [s["candidate_id"] for s in scored_results]
        enrich = self.fetch_candidates_from_mysql(candidate_ids)
        
        for s in scored_results:
            cid = s["candidate_id"]
            candidate_data = enrich.get(cid, {})
            s["skills"] = candidate_data.get("skills", "")
            s["brief_description"] = candidate_data.get("brief_description", "")
            s["job_title"] = candidate_data.get("job_title", "")
            s["sector"] = candidate_data.get("sector", "")
            s["education"] = candidate_data.get("education", "")
            s["experience"] = candidate_data.get("experience", "")
            s["language"] = candidate_data.get("language", "")
            s["city"] = candidate_data.get("city", "")
        
        return scored_results
