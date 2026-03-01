#!/usr/bin/env python3
"""
Script para validar consistencia entre baseline y nuevos resultados.
Compara candidatos, affinities y scores por dimensión.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_json(filepath: Path) -> Dict:
    """Carga archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_case(baseline_case: Dict, new_case: Dict, tolerance: float = 0.001) -> Dict:
    """
    Valida un caso de prueba específico.
    
    Returns:
        Dict con resultado de validación
    """
    case_id = baseline_case['case_id']
    errors = []
    warnings = []
    
    baseline_output = baseline_case['output']
    new_output = new_case['output']
    
    # 1. Validar número total de resultados
    baseline_total = baseline_output['total_results']
    new_total = new_output['total_results']
    
    if baseline_total != new_total:
        warnings.append({
            "type": "total_results_mismatch",
            "baseline": baseline_total,
            "new": new_total,
            "diff": abs(baseline_total - new_total)
        })
    
    # 2. Validar top 10 candidatos
    baseline_top10 = {c['candidate_id'] for c in baseline_output['top_10_candidates']}
    new_top10 = {c['candidate_id'] for c in new_output['top_10_candidates']}
    
    missing = baseline_top10 - new_top10
    extra = new_top10 - baseline_top10
    
    if missing or extra:
        errors.append({
            "type": "top10_candidates_mismatch",
            "missing": list(missing),
            "extra": list(extra)
        })
    
    # 3. Validar affinities de candidatos comunes
    baseline_affinities = {
        c['candidate_id']: c['affinity']
        for c in baseline_output['top_10_candidates']
    }
    new_affinities = {
        c['candidate_id']: c['affinity']
        for c in new_output['top_10_candidates']
    }
    
    for cid in baseline_top10.intersection(new_top10):
        baseline_aff = baseline_affinities[cid]
        new_aff = new_affinities[cid]
        diff = abs(baseline_aff - new_aff)
        
        if diff > tolerance * 100:  # tolerance en porcentaje
            errors.append({
                "type": "affinity_mismatch",
                "candidate_id": cid,
                "baseline": baseline_aff,
                "new": new_aff,
                "diff": diff
            })
    
    # 4. Validar scores por dimensión (dimension_details)
    baseline_details = {
        d['candidate_id']: d['dimensions']
        for d in baseline_output['dimension_details']
    }
    new_details = {
        d['candidate_id']: d['dimensions']
        for d in new_output['dimension_details']
    }
    
    for cid in baseline_top10.intersection(new_top10):
        if cid in baseline_details and cid in new_details:
            baseline_dims = baseline_details[cid]
            new_dims = new_details[cid]
            
            # Comparar cada dimensión
            all_dims = set(baseline_dims.keys()).union(set(new_dims.keys()))
            
            for dim in all_dims:
                if dim not in baseline_dims:
                    warnings.append({
                        "type": "dimension_missing_in_baseline",
                        "candidate_id": cid,
                        "dimension": dim
                    })
                    continue
                
                if dim not in new_dims:
                    warnings.append({
                        "type": "dimension_missing_in_new",
                        "candidate_id": cid,
                        "dimension": dim
                    })
                    continue
                
                baseline_score = baseline_dims[dim]['score_0_1']
                new_score = new_dims[dim]['score_0_1']
                score_diff = abs(baseline_score - new_score)
                
                if score_diff > tolerance:
                    errors.append({
                        "type": "dimension_score_mismatch",
                        "candidate_id": cid,
                        "dimension": dim,
                        "baseline": baseline_score,
                        "new": new_score,
                        "diff": score_diff
                    })
    
    return {
        "case_id": case_id,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def validate_consistency(baseline_path: str, new_path: str, tolerance: float = 0.001) -> bool:
    """
    Valida consistencia entre baseline y nuevos resultados.
    
    Args:
        baseline_path: Ruta al baseline snapshot
        new_path: Ruta a los nuevos resultados
        tolerance: Tolerancia para diferencias en scores (default: 0.1%)
    
    Returns:
        True si todos los casos pasan, False si hay errores
    """
    
    print("=" * 70)
    print("VALIDACIÓN DE CONSISTENCIA - TFM Match")
    print("=" * 70)
    print()
    
    # Cargar archivos
    script_dir = Path(__file__).parent
    baseline_file = script_dir / baseline_path
    new_file = script_dir / new_path
    
    try:
        baseline = load_json(baseline_file)
    except FileNotFoundError:
        print(f"Error: No se encontró baseline en {baseline_file}")
        return False
    
    try:
        new_results = load_json(new_file)
    except FileNotFoundError:
        print(f"Error: No se encontraron nuevos resultados en {new_file}")
        return False
    
    print(f"Baseline: {baseline_file}")
    print(f"Nuevos: {new_file}")
    print(f"Tolerancia: {tolerance * 100}%")
    print()
    
    # Validar cada caso
    baseline_results = baseline.get('results', [])
    new_results_list = new_results.get('results', [])
    
    if len(baseline_results) != len(new_results_list):
        print(f"   Advertencia: Número de casos difiere")
        print(f"   Baseline: {len(baseline_results)} casos")
        print(f"   Nuevos: {len(new_results_list)} casos")
        print()
    
    # Crear índice por case_id
    new_results_idx = {r['case_id']: r for r in new_results_list}
    
    all_validations = []
    passed_count = 0
    failed_count = 0
    
    print("-" * 70)
    print("Validando casos...")
    print("-" * 70)
    print()
    
    for baseline_case in baseline_results:
        case_id = baseline_case['case_id']
        
        if case_id not in new_results_idx:
            print(f" {case_id}: No encontrado en nuevos resultados")
            failed_count += 1
            continue
        
        new_case = new_results_idx[case_id]
        validation = validate_case(baseline_case, new_case, tolerance)
        all_validations.append(validation)
        
        if validation['passed']:
            print(f"{case_id}: OK")
            passed_count += 1
        else:
            print(f"{case_id}: FALLÓ ({len(validation['errors'])} errores)")
            failed_count += 1
            
            # Mostrar primeros 3 errores
            for error in validation['errors'][:3]:
                print(f"   - {error['type']}: {error}")
            
            if len(validation['errors']) > 3:
                print(f"   ... y {len(validation['errors']) - 3} errores más")
    
    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total casos: {len(baseline_results)}")
    print(f"Pasados: {passed_count}")
    print(f"Fallados: {failed_count}")
    
    # Contar warnings totales
    total_warnings = sum(len(v['warnings']) for v in all_validations)
    if total_warnings > 0:
        print(f" Warnings: {total_warnings}")
    
    print()
    
    # Guardar reporte detallado
    report_file = script_dir / "validation_report.json"
    report = {
        "baseline_file": str(baseline_file),
        "new_file": str(new_file),
        "tolerance": tolerance,
        "total_cases": len(baseline_results),
        "passed": passed_count,
        "failed": failed_count,
        "total_warnings": total_warnings,
        "validations": all_validations
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Reporte detallado guardado en: {report_file}")
    print()
    
    if failed_count == 0:
        print("¡Todos los casos pasaron la validación!")
        return True
    else:
        print(" Algunos casos fallaron. Revisa el reporte para más detalles.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Valida consistencia de resultados")
    parser.add_argument(
        "--baseline",
        default="baseline_snapshot.json",
        help="Archivo baseline (default: baseline_snapshot.json)"
    )
    parser.add_argument(
        "--new",
        default="refactored_results.json",
        help="Archivo nuevos resultados (default: refactored_results.json)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Tolerancia para diferencias en scores (default: 0.001 = 0.1%%)"
    )
    
    args = parser.parse_args()
    
    success = validate_consistency(args.baseline, args.new, args.tolerance)
    sys.exit(0 if success else 1)
