import os, sys
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY="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"

SYSTEM_INSTRUCTIONS = """\
Lee EXCLUSIVAMENTE estos dos enlaces oficiales de Cisco y contesta la consulta del usuario.
Si en los enlaces no hay información suficiente, responde exactamente:
"No tengo la informacion suficiente para responder tu consulta".
Enlaces permitidos:
1) https://www.cisco.com/site/us/en/learn/training-certifications/exams/retired.html
2) https://www.cisco.com/site/us/en/learn/training-certifications/exams/list.html
Cita la(s) URL(s) usada(s) al final de tu respuesta.
"""

def ask(query: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_API_ORGANIZATION"))

    # Truco para forzar que la herramienta busque solo en esas páginas:
    # - Instrucciones del sistema
    # - “site:” en la consulta para limitar el dominio y mencionar explícitamente las URLs
    guarded_query = (
        f"Consulta del usuario: {query}\n\n"
        "Usa solo estas fuentes (si no están, responde que no hay info suficiente):\n"
        "site:cisco.com "
        "https://www.cisco.com/site/us/en/learn/training-certifications/exams/retired.html "
        "https://www.cisco.com/site/us/en/learn/training-certifications/exams/list.html"
    )

    resp = client.responses.create(
        model="gpt-5-nano",               # también funciona con gpt-4o / gpt-4.1 (según disponibilidad) "gpt-4.1-mini"
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": guarded_query},
        ],
        tools=[{"type": "web_search"}],    # <-- habilita la búsqueda web gestionada por OpenAI
    )
    # La API expone helpers como .output_text; si no, navega el objeto para extraer texto.
    return getattr(resp, "output_text", str(resp))

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Falta OPENAI_API_KEY en tu entorno o .env", file=sys.stderr)
        sys.exit(1)

    consulta = input("Escribe tu consulta (se responderá SOLO con las dos URLs de Cisco): ").strip()
    print("\n--- Respuesta ---")
    print(ask(consulta))
