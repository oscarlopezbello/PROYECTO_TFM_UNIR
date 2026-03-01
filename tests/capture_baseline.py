#!/usr/bin/env python3
"""
Script para capturar el baseline de resultados actuales.
Ejecuta todos los casos de prueba y guarda los resultados como referencia.
"""

import json
import requests
import sys
from datetime import datetime
from pathlib import Path


API_URL = "http://127.0.0.1:8000/match"


def extract_dimension_scores(result):
    """
    Extrae los scores por dimensión de cada candidato
    para validación granular después.
    """
    dimension_data = []
    
    for candidate in result.get("results", []):
        cid = candidate["candidate_id"]
        breakdown = candidate.get("breakdown", {})
        
        dimension_data.append({
            "candidate_id": cid,
            "dimensions": {
                dim: {
                    "score_0_1": breakdown[dim]["score_0_1"],
                    "weight": breakdown[dim]["weight"],
                    "contribution": breakdown[dim]["contribution"]
                }
                for dim in breakdown.keys()
            }
        })
    
    return dimension_data


def capture_baseline(output_file="baseline_snapshot.json"):
    """
    Ejecuta todos los casos de prueba contra la API ACTUAL
    y guarda los resultados como baseline de referencia.
    """
    
    script_dir = Path(__file__).parent
    cases_file = script_dir / "baseline_cases.json"
    output_path = script_dir / output_file
    
    print("=" * 60)
    print("CAPTURA DE BASELINE - TFM Match API")
    print("=" * 60)
    print()
    
    # Cargar casos de prueba
    try:
        with open(cases_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró {cases_file}")
        sys.exit(1)
    
    print(f"Casos de prueba cargados: {len(test_cases)}")
    print(f"API URL: {API_URL}")
    print()
    
    # Verificar que la API esté disponible
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code != 200:
            print("Error: La API no está respondiendo correctamente")
            print("   Por favor, asegúrate de que el servidor esté corriendo:")
            print("   poetry run uvicorn tfm_match.api.main:app --reload")
            sys.exit(1)
        print("API disponible y funcionando")
    except requests.exceptions.RequestException as e:
        print(f"Error: No se puede conectar a la API: {e}")
        print("   Por favor, asegúrate de que el servidor esté corriendo:")
        print("   poetry run uvicorn tfm_match.api.main:app --reload")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("Ejecutando casos de prueba...")
    print("-" * 60)
    print()
    
    baseline_results = {
        "timestamp": datetime.now().isoformat(),
        "api_version": "original",
        "results": []
    }
    
    for i, case in enumerate(test_cases, 1):
        case_id = case['case_id']
        print(f"[{i}/{len(test_cases)}] Ejecutando {case_id}...", end=" ")
        
        try:
            # Llamar API actual
            response = requests.post(API_URL, json=case["input"], timeout=30)
            
            if response.status_code != 200:
                print(f"Error HTTP {response.status_code}")
                print(f"    Respuesta: {response.text[:200]}")
                continue
            
            result = response.json()
            
            # Capturar información relevante
            baseline_results["results"].append({
                "case_id": case["case_id"],
                "description": case["description"],
                "input": case["input"],
                "output": {
                    "job_request_id": result.get("job_request_id"),
                    "total_results": len(result.get("results", [])),
                    "top_10_candidates": [
                        {
                            "candidate_id": r["candidate_id"],
                            "affinity": r["affinity"],
                            "breakdown": r["breakdown"]
                        }
                        for r in result.get("results", [])[:10]
                    ],
                    "dimension_details": extract_dimension_scores(result)
                }
            })
            
            print(f"{len(result.get('results', []))} candidatos")
            
        except requests.exceptions.Timeout:
            print("Timeout")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")
    
    print()
    print("-" * 60)
    print("Guardando baseline...")
    print("-" * 60)
    
    # Guardar baseline
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        
        print(f"Baseline capturado exitosamente")
        print(f"Archivo: {output_path}")
        print(f"Casos completados: {len(baseline_results['results'])}/{len(test_cases)}")
        print()
        
        # Resumen
        print("=" * 60)
        print("RESUMEN")
        print("=" * 60)
        for result in baseline_results["results"]:
            print(f"• {result['case_id']}: {result['output']['total_results']} candidatos")
        
        return True
        
    except Exception as e:
        print(f"Error al guardar baseline: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Captura baseline de resultados")
    parser.add_argument(
        "--output",
        default="baseline_snapshot.json",
        help="Archivo de salida (default: baseline_snapshot.json)"
    )
    
    args = parser.parse_args()
    
    success = capture_baseline(args.output)
    sys.exit(0 if success else 1)
