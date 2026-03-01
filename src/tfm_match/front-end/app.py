import streamlit as st
import requests
from pathlib import Path
import re
import html
import os
from urllib.parse import urljoin

DEFAULT_API_BASE = os.getenv("TFM_MATCH_API_BASE_URL", "http://127.0.0.1:8000")

def build_api_urls(api_base: str) -> tuple[str, str, str]:
    base = api_base.rstrip("/") + "/"
    return (
        urljoin(base, "match"),
        urljoin(base, "cities"),
        urljoin(base, "health"),
    )


def resolve_api_base() -> str:
    """
    Resuelve base URL del API.
    - Usa TFM_MATCH_API_BASE_URL si existe
    - Si no conecta y es 8000, intenta 8001 automáticamente (útil en Windows cuando el puerto cambia).
    """
    api_base = DEFAULT_API_BASE
    match_url, _, health_url = build_api_urls(api_base)
    try:
        requests.get(health_url, timeout=2)
        return api_base
    except Exception:
        pass

    if api_base.endswith(":8000"):
        api_base2 = api_base[:-4] + "8001"
        _, _, health_url2 = build_api_urls(api_base2)
        try:
            requests.get(health_url2, timeout=2)
            st.warning(f"No se pudo conectar al API en {api_base}. Usando {api_base2}.")
            return api_base2
        except Exception:
            pass

    return api_base


API_BASE = resolve_api_base()
API_URL, API_CITIES_URL, API_HEALTH_URL = build_api_urls(API_BASE)

# -----------------------------
# Helpers para cargar datos del API
# -----------------------------
@st.cache_data(ttl=3600)
def load_cities(api_cities_url: str):
    """
    Carga ciudades desde el API. Cache por 1 hora.
    """
    try:
        r = requests.get(api_cities_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            cities = data.get("cities", [])
            return ["No aplica"] + cities  # Agregar "No aplica" al inicio
        else:
            return ["No aplica", "Bogotá", "Medellín", "Cali", "Remoto"]  # Fallback
    except Exception:
        return ["No aplica", "Bogotá", "Medellín", "Cali", "Remoto"]  # Fallback


# -----------------------------
# Configuración y CSS
# -----------------------------
st.set_page_config(
    page_title="Plataforma de Emparejamiento de Vacantes",
    layout="wide"
)

def load_css():
    """
    Carga CSS relativo al archivo actual para evitar fallos por cwd.
    Asume: front-end/styles.css está junto a app.py
    """
    try:
        css_path = Path(__file__).parent / "styles.css"
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception:
        # No bloquea la app si no está el archivo
        pass

load_css()


# -----------------------------
# Helpers UI/estado
# -----------------------------
DEFAULT_STATE = {
    "job_name": "",
    "city": "No aplica",
    "education": "No aplica",
    "skills": "",
    "sector": "",
    "experience": "No aplica",
    "language": "No aplica",
    "top_k": 10,
    "w_education": 0.0,
    "w_skills": 2.5,
    "w_sector": 2.5,
    "w_exp": 0.0,
    "w_lang": 0.0,
    "w_job_title": 5.0,
    "w_city": 0.0,
    "results": None,
    "last_payload": None,
    "last_error": None,
}

def init_state():
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_form():
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v

def safe_post(url: str, payload: dict, timeout: int = 30):
    """
    POST con manejo básico de errores.
    Retorna: (ok: bool, json_or_text: Any)
    """
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code >= 200 and r.status_code < 300:
            return True, r.json()
        # Si no es JSON, devuelve texto para diagnóstico
        try:
            return False, r.json()
        except Exception:
            return False, r.text
    except Exception as e:
        return False, str(e)

def _split_skills(raw: str, max_items: int = 80):
    """
    Convierte un string "sucio" de skills (con comas/puntos/guiones/saltos de línea)
    en lista de ítems para mostrar (no afecta embeddings ni el matching).
    """
    if not raw:
        return []

    t = str(raw)
    # Normaliza separadores comunes a coma
    t = t.replace("\r", ",").replace("\n", ",").replace("\t", ",")
    t = re.sub(r"[|;/•·]+", ",", t)
    # En muchos CVs el punto se usa como separador de ítems
    t = t.replace(".", ",")
    # Guiones como separador
    t = t.replace(" - ", ",").replace(" -", ",").replace("- ", ",")

    parts = [p.strip(" \u00A0-–—_") for p in t.split(",")]
    parts = [p for p in parts if p]

    # Dedup preservando orden (case-insensitive)
    seen = set()
    out = []
    for p in parts:
        key = p.casefold()
        if key not in seen:
            out.append(p)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def _split_list_like_text(raw: str, max_items: int = 80):
    """
    Split genérico para textos tipo lista (job_title, sector, language, etc.)
    Ej: "RAC CALL CENTER; Call center; Secretaria" -> ["RAC CALL CENTER", "Call center", "Secretaria"]
    """
    if not raw:
        return []
    t = str(raw)
    t = t.replace("\r", ";").replace("\n", ";").replace("\t", ";")
    t = re.sub(r"[|,/•·]+", ";", t)
    parts = [p.strip(" \u00A0-–—_") for p in t.split(";")]
    parts = [p for p in parts if p]

    seen = set()
    out = []
    for p in parts:
        key = p.casefold()
        if key not in seen:
            out.append(p)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def list_to_chips_html(raw: str, max_chips: int = 12) -> str:
    items = _split_list_like_text(raw)
    if not items:
        return "<span class='muted'>No especificado</span>"

    shown = items[:max_chips]
    chips = " ".join([f"<span class='skill-chip'>{html.escape(s)}</span>" for s in shown])
    remaining = len(items) - len(shown)
    if remaining > 0:
        chips += f" <span class='skill-chip skill-chip--more'>+{remaining} más</span>"
    return f"<div class='skill-chip-wrap'>{chips}</div>"

def skills_to_chips_html(raw: str, max_chips: int = 18) -> str:
    skills = _split_skills(raw)
    if not skills:
        return "<span class='muted'>No especificado</span>"

    shown = skills[:max_chips]
    chips = " ".join([f"<span class='skill-chip'>{html.escape(s)}</span>" for s in shown])
    remaining = len(skills) - len(shown)
    if remaining > 0:
        chips += f" <span class='skill-chip skill-chip--more'>+{remaining} más</span>"
    return f"<div class='skill-chip-wrap'>{chips}</div>"

init_state()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0072CE, #003A8F);
                padding: 22px;
                border-radius: 12px;
                color: white;">
        <h1 style="margin:0;">Plataforma de Emparejamiento de Vacantes</h1>
        <p style="margin:6px 0 0 0;">Proyecto TFM - UNIR</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Mostrar base del API (útil para debug)
with st.expander("Configuración API", expanded=False):
    st.code(f"API_BASE={API_BASE}\nMATCH={API_URL}\nCITIES={API_CITIES_URL}", language="text")


# -----------------------------
# Layout principal
# -----------------------------
left, right = st.columns([1.05, 1.45], gap="large")

with left:
    st.subheader("Información de la vacante")

    # Cargar ciudades disponibles desde el API
    available_cities = load_cities(API_CITIES_URL)

    # Campos de entrada (sin form para actualización en tiempo real de pesos)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.job_name = st.text_input("Nombre del cargo", value=st.session_state.job_name, key="job_name_input")
    with col2:
        # Determinar índice actual
        try:
            current_index = available_cities.index(st.session_state.city or "No aplica")
        except ValueError:
            current_index = 0
        
        st.session_state.city = st.selectbox(
            "Ciudad",
            available_cities,
            index=current_index,
            key="city_input"
        )

    # Sliders de peso para cargo y ciudad 
    col1b, col2b = st.columns(2)
    with col1b:
        st.session_state.w_job_title = st.slider("Peso de nombre del cargo", 0.0, 10.0, st.session_state.w_job_title, step=0.5, key="w_job_slider")
    with col2b:
        st.session_state.w_city = st.slider("Peso de ciudad", 0.0, 10.0, st.session_state.w_city, step=0.5, key="w_city_slider")

    st.session_state.education = st.selectbox(
        "Nivel de formación requerido",
        ["No aplica", "Bachiller", "Técnico", "Tecnólogo", "Profesional", "Posgrado"],
        index=["No aplica", "Bachiller", "Técnico", "Tecnólogo", "Profesional", "Posgrado"].index(st.session_state.education or "No aplica"),
        key="education_input"
    )
    st.session_state.w_education = st.slider("Peso de formación", 0.0, 10.0, st.session_state.w_education, step=0.5, key="w_edu_slider")

    st.session_state.skills = st.text_area(
        "Habilidades requeridas",
        value=st.session_state.skills,
        placeholder="Ej: Excel, Office, atención al cliente, ventas...",
        help="Tip: escribe varias habilidades separadas por coma (,) o punto y coma (;). Ej: Excel, Office, CRM, ventas.",
        key="skills_input"
    )
    st.caption("Puedes pegar una lista: separa habilidades por coma o punto y coma. Entre más específicas (p. ej. 'Excel', 'Power BI', 'CRM'), mejor el match.")
    st.session_state.w_skills = st.slider("Peso de habilidades", 0.0, 10.0, st.session_state.w_skills, step=0.5, key="w_skills_slider")

    st.session_state.sector = st.text_area(
        "Área o sector de la vacante",
        value=st.session_state.sector,
        placeholder="Ej: BPO/Contact Center, Retail, Logística",
        key="sector_input"
    )
    st.session_state.w_sector = st.slider("Peso de área/sector", 0.0, 10.0, st.session_state.w_sector, step=0.5, key="w_sector_slider")

    col3, col4 = st.columns(2)
    with col3:
        st.session_state.experience = st.selectbox(
            "Experiencia requerida (años)",
            ["No aplica", "0-1", "1-3", "3-5", "5+"],
            index=["No aplica", "0-1", "1-3", "3-5", "5+"].index(st.session_state.experience or "No aplica"),
            key="exp_input"
        )
    with col4:
        st.session_state.language = st.selectbox(
            "Idioma requerido",
            ["No aplica", "Inglés", "Francés", "Portugués"],
            index=["No aplica", "Inglés", "Francés", "Portugués"].index(st.session_state.language or "No aplica"),
            key="lang_input"
        )
    
    col3b, col4b = st.columns(2)
    with col3b:
        st.session_state.w_exp = st.slider("Peso de experiencia", 0.0, 10.0, st.session_state.w_exp, step=0.5, key="w_exp_slider")
    with col4b:
        st.session_state.w_lang = st.slider("Peso de idioma", 0.0, 10.0, st.session_state.w_lang, step=0.5, key="w_lang_slider")
    
    # Calcular suma total de pesos EN TIEMPO REAL
    total_weight = (
        st.session_state.w_job_title + 
        st.session_state.w_city + 
        st.session_state.w_education + 
        st.session_state.w_skills + 
        st.session_state.w_sector + 
        st.session_state.w_exp + 
        st.session_state.w_lang
    )
    
    # Mostrar indicador de suma total con colores dinámicos EN TIEMPO REAL
    st.markdown("---")
    if total_weight > 10:
        st.error(f"**Suma de pesos: {total_weight:.1f} / 10** EXCEDE EL LÍMITE")
    elif total_weight == 0:
        st.warning(f"**Suma de pesos: {total_weight:.1f} / 10** - Debe tener al menos un peso > 0")
    else:
        st.success(f"**Suma de pesos: {total_weight:.1f} / 10** - Correcto")
    st.markdown("---")

    st.session_state.top_k = st.number_input("Top-K candidatos", min_value=1, max_value=50, value=st.session_state.top_k, key="topk_input")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        submitted = st.button("Emparejar Candidatos", type="primary")
    with col_btn2:
        if st.button("Borrar Campos"):
            reset_form()
            st.rerun()

    # Validaciones antes de ejecutar matching
    invalid = False
    if submitted:
        # Calcular suma total de pesos
        total_weight = (
            st.session_state.w_job_title + 
            st.session_state.w_city + 
            st.session_state.w_education + 
            st.session_state.w_skills + 
            st.session_state.w_sector + 
            st.session_state.w_exp + 
            st.session_state.w_lang
        )
        
        # Validación 1: Suma de pesos no puede exceder 10
        if total_weight > 10:
            invalid = True
            st.error(f"La suma de pesos es {total_weight:.1f}, pero debe ser ≤ 10. Por favor ajusta los pesos.")
        
        # Validación 2: Al menos una dimensión con contenido
        if not st.session_state.skills and not st.session_state.sector and not st.session_state.job_name:
            invalid = True
            st.warning("Ingresa al menos Habilidades, Área/Sector o Nombre del cargo para buscar candidatos.")

    # Construcción de payload + llamada a API
    if submitted and not invalid:
        with st.spinner("Buscando candidatos..."):
            # Convertir "No aplica" a vacío para el backend
            education_val = "" if st.session_state.education == "No aplica" else st.session_state.education
            experience_val = "" if st.session_state.experience == "No aplica" else st.session_state.experience
            language_val = "" if st.session_state.language == "No aplica" else st.session_state.language
            city_val = "" if st.session_state.city == "No aplica" else st.session_state.city
            
            # Payload compatible con el esquema de la API (campos planos)
            payload = {
                "skills": st.session_state.skills,
                "experience": experience_val,
                "education": education_val,
                "language": language_val,
                "sector": st.session_state.sector,
                "job_title": st.session_state.job_name,  # ← CAMPO CLAVE
                "city": city_val,  # ← Ciudad
                "top_k": int(st.session_state.top_k),
                "weights": {
                    "skills": int(st.session_state.w_skills),
                    "experience": int(st.session_state.w_exp),
                    "education": int(st.session_state.w_education),
                    "language": int(st.session_state.w_lang),
                    "sector": int(st.session_state.w_sector),
                    "job_title": int(st.session_state.w_job_title),  # Peso para job_title
                    "city": int(st.session_state.w_city),  # Peso para city
                }
            }

            ok, data = safe_post(API_URL, payload, timeout=45)

            if ok:
                st.session_state.last_payload = payload
                st.session_state.last_error = None
                st.session_state.results = data.get("results", [])
            else:
                st.session_state.last_payload = payload
                st.session_state.last_error = data
                st.session_state.results = None


with right:
    st.subheader("Resultados de Emparejamiento")

    # Mensaje de error si hubo fallo
    if st.session_state.last_error:
        st.error("No fue posible obtener resultados del servicio de matching.")
        with st.expander("Ver detalle del error"):
            st.write(st.session_state.last_error)
        with st.expander("Ver payload enviado"):
            st.json(st.session_state.last_payload)

    results = st.session_state.results

    if results is None:
        st.info("Aún no has ejecutado una búsqueda.")
    elif not results:
        st.warning("No se encontraron candidatos.")
    else:
        # Resumen rápido
        st.caption(f"Se muestran {len(results)} candidatos (Top-K).")

        # Tabla compacta (si el API no trae más campos, lo maneja igual)
        table_rows = []
        for r in results:
            table_rows.append({
                "candidate_id": r.get("candidate_id"),
                "afinidad_%": r.get("affinity"),
                "skills_resumen": (r.get("skills") or "")[:120]
            })
        # st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Cards detalladas
        for r in results:
            affinity = r.get("affinity", 0)
            cand_id = r.get("candidate_id", "NA")
            skills_txt = r.get("skills", "")
            skills_html = skills_to_chips_html(skills_txt)

            with st.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <strong> Candidato ID: {cand_id}</strong>
                        <div style="float:right; color:#0072CE;"><strong>{affinity}% afinidad</strong></div>
                        <div class="affinity-bar">
                            <div class="affinity-fill" style="width:{affinity}%"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Desglose dinámico por dimensión (solo muestra las que se buscaron)
                breakdown = r.get("breakdown")
                if breakdown and st.session_state.last_payload:
                    with st.expander("Ver desglose por dimensión"):
                        payload = st.session_state.last_payload
                        
                        # Mapeo de dimensiones con sus nombres amigables
                        dimension_map = {
                            "job_title": {"label": "Nombre del Cargo", "query_key": "job_title", "candidate_key": "job_title"},
                            "city": {"label": "Ciudad", "query_key": "city", "candidate_key": "city"},
                            "skills": {"label": "Habilidades", "query_key": "skills", "candidate_key": "skills"},
                            "sector": {"label": "Área/Sector", "query_key": "sector", "candidate_key": "sector"},
                            "education": {"label": "Nivel de Formación", "query_key": "education", "candidate_key": "education"},
                            "experience": {"label": "Experiencia", "query_key": "experience", "candidate_key": "experience"},
                            "language": {"label": "Idioma", "query_key": "language", "candidate_key": "language"}
                        }
                        
                        # Solo mostrar dimensiones que fueron buscadas (tienen valor en payload)
                        for dim_key, dim_info in dimension_map.items():
                            query_value = payload.get(dim_info["query_key"])
                            
                            # Si esta dimensión fue usada en la búsqueda
                            if query_value:
                                # Extraer el score (puede ser objeto o número directo)
                                dim_data = breakdown.get(dim_key, {})
                                if isinstance(dim_data, dict):
                                    dim_score = float(dim_data.get("score_pct", 0))
                                else:
                                    dim_score = float(dim_data) if dim_data else 0.0
                                
                                # Obtener el valor real del candidato desde el resultado
                                candidate_value = r.get(dim_info["candidate_key"], "No especificado")
                                # Formato especial para skills: chips/tags para legibilidad
                                if dim_key == "skills":
                                    candidate_value = skills_to_chips_html(candidate_value)
                                elif dim_key in ("job_title", "sector", "language", "experience"):
                                    candidate_value = list_to_chips_html(candidate_value)
                                elif dim_key == "city":
                                    candidate_value = html.escape(str(candidate_value))
                                else:
                                    candidate_value = html.escape(str(candidate_value))
                                
                                # Tarjeta visual por dimensión
                                st.markdown(f"**{dim_info['label']}** (Similitud: {round(dim_score, 1)}%)")
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    qv = str(query_value)
                                    if dim_key == "skills":
                                        q_html = skills_to_chips_html(qv, max_chips=12)
                                    elif dim_key in ("job_title", "sector", "language", "experience"):
                                        q_html = list_to_chips_html(qv, max_chips=10)
                                    else:
                                        q_html = html.escape(qv)
                                    st.markdown(f"<small><strong>Buscado:</strong></small><br>{q_html}", unsafe_allow_html=True)
                                with col2:
                                    st.markdown(
                                        f"<small><strong>Candidato tiene:</strong></small><br>{candidate_value}",
                                        unsafe_allow_html=True
                                    )
                                
                                # Extra info: experiencia total en meses (si viene del backend)
                                if dim_key == "experience" and isinstance(dim_data, dict):
                                    total_m = dim_data.get("total_months")
                                    if total_m is not None:
                                        try:
                                            total_m_int = int(total_m)
                                            years = total_m_int // 12
                                            months = total_m_int % 12
                                            st.caption(f"Total experiencia (estimada): {years} años {months} meses")
                                        except Exception:
                                            pass
                                
                                # Barra de progreso visual
                                st.progress(min(dim_score / 100.0, 1.0))
                                st.markdown("---")



        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("⬅ Nueva búsqueda"):
                st.session_state.results = None
                st.session_state.last_error = None
                st.session_state.last_payload = None
                st.rerun()
        with col_b:
            # Export rápido
            if st.session_state.results:
                st.download_button(
                    "⬇ Exportar JSON",
                    data=str({"results": st.session_state.results}),
                    file_name="matching_results.json"
                )
