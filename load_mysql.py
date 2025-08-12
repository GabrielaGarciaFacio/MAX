"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import os
import argparse
from dotenv import find_dotenv, load_dotenv
from models.hybrid_search_retreiver import HybridSearchRetriever

hsr = HybridSearchRetriever()

# Cargar .env
dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
else:
    raise FileNotFoundError("No .env file found in root directory of repository")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Carga datos SQL o PDFs en el/los namespace(s) especificado(s)"
    )
    parser.add_argument(
        "namespace",
        nargs="?",
        help="Namespace para la carga (Cursos, Examenes, Laboratorios) o 'Temarios' para PDFs. Si se omite, ejecuta todos.",
        default=None
    )
    args = parser.parse_args()

    namespaces = [args.namespace] if args.namespace else ["Cursos", "Examenes", "Laboratorios", "Temarios"]

    for ns in namespaces:
        ns_clean = ns.strip()
        print(f"\n=== Procesando «{ns_clean}» ===")

        # 1) Consultar y eliminar namespace si ya existe
        try:
            stats = hsr.pinecone.index.describe_index_stats()
            existing_namespaces = stats.get("namespaces", {}).keys()
            if ns_clean in existing_namespaces:
                print(f"  → Namespace '{ns_clean}' ya existe. Eliminando contenido…")
                hsr.pinecone.index.delete(delete_all=True, namespace=ns_clean)
        except Exception as e:
            print(f"  → No se pudo comprobar/eliminar '{ns_clean}': {e}")

        # 2) Volver a cargar
        if ns_clean.lower() == "temarios":
            print("  → Llamando a pdf_loader()…")
            hsr.pdf_loader()
        else:
            print(f"  → Llamando a load_sql(namespace='{ns_clean}')…")
            hsr.load_sql(namespace=ns_clean)
