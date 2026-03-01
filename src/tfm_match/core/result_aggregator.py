"""
Result Aggregator - Combina resultados de múltiples dimensiones y calcula scoring.
Código extraído de api/main.py (líneas 490-579).
"""

from typing import List, Dict, Any


class ResultAggregator:
    """Agrega y rankea resultados de múltiples dimensiones."""
    
    @staticmethod
    def collect_candidates(hits_by_dim: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Recolecta todos los candidate_ids únicos de todas las dimensiones.
        LÓGICA EXACTA de líneas 503-510 de main.py original.
        
        Args:
            hits_by_dim: Dict de dimensión -> lista de hits
            
        Returns:
            Lista de candidate_ids únicos
        """
        candidate_ids = []
        seen = set()
        for dim, hits in hits_by_dim.items():
            for h in hits:
                c = h["id_candidate"]
                if c not in seen:
                    seen.add(c)
                    candidate_ids.append(c)
        return candidate_ids
    
    @staticmethod
    def combine_and_rank(
        hits_by_dim: Dict[str, List[Dict[str, Any]]],
        weight_map: Dict[str, int],
        candidate_ids: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combina resultados y calcula scoring ponderado.
        LÓGICA EXACTA de líneas 524-579 de main.py original.
        
        Args:
            hits_by_dim: Resultados por dimensión
            weight_map: Pesos por dimensión
            candidate_ids: Lista de candidatos a considerar
            top_k: Número de top candidatos a retornar
            
        Returns:
            Lista de candidatos ranqueados con affinity y breakdown
        """
        # Indexa rápido sim por candidato/dim
        sim_idx: Dict[str, Dict[str, float]] = {c: {} for c in candidate_ids}
        for dim, hits in hits_by_dim.items():
            for h in hits:
                c = h["id_candidate"]
                if c in sim_idx:
                    sim_idx[c][dim] = max(sim_idx[c].get(dim, 0.0), float(h["sim"]))
        
        # Calcular peso total de TODAS las dimensiones con peso > 0
        total_weight = sum(ww for ww in weight_map.values() if ww > 0)
        
        if total_weight <= 0:
            return []
        
        scored = []
        for c in candidate_ids:
            breakdown = {}
            weighted_sum = 0.0
            
            # Procesar TODAS las dimensiones con peso
            for dim, ww in weight_map.items():
                if ww <= 0:
                    continue
                
                # Si el candidato tiene match en esta dimensión
                if dim in sim_idx[c]:
                    dim_score = float(sim_idx[c][dim])  # 0..1
                else:
                    # Si no tiene match, score = 0
                    dim_score = 0.0
                
                weighted_sum += dim_score * ww
                breakdown[dim] = {
                    "score_0_1": round(dim_score, 4),
                    "score_pct": round(dim_score * 100, 2),
                    "weight": ww,
                    "contribution": round(dim_score * ww, 4),
                }
            
            # Score final = suma ponderada / total de pesos
            final_0_1 = weighted_sum / total_weight
            scored.append({
                "candidate_id": c,
                "affinity": round(final_0_1 * 100, 2),
                "breakdown": breakdown,
            })
        
        scored.sort(key=lambda x: x["affinity"], reverse=True)
        scored = scored[:top_k]
        
        return scored
