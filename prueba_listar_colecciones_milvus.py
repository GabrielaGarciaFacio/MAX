#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Listar colecciones en Milvus.

Uso básico:
  python listar_colecciones_milvus.py

Con filtros y métricas:
  python listar_colecciones_milvus.py --host localhost --port 19530 --prefix netec_ --show-counts --show-schema

Salida en JSON (útil para pipelines):
  python listar_colecciones_milvus.py --as-json --show-counts
"""

import argparse
import json
import sys

from pymilvus import connections, utility, Collection
from pymilvus.exceptions import MilvusException


def connect_milvus(host: str, port: int, user: str | None, password: str | None, secure: bool) -> None:
    """
    Establece conexión con Milvus usando el alias 'default'.
    """
    params = {
        "host": host,
        "port": str(port),
    }
    # Autenticación opcional (si tu Milvus la tiene habilitada)
    if user:
        params["user"] = user
    if password:
        params["password"] = password
    # TLS opcional
    if secure:
        params["secure"] = True

    connections.connect(alias="default", **params)


def list_collections(prefix: str | None, contains: str | None) -> list[str]:
    """
    Devuelve la lista de colecciones, aplicando filtros simples si se indican.
    """
    names = utility.list_collections()
    if prefix:
        names = [n for n in names if n.startswith(prefix)]
    if contains:
        names = [n for n in names if contains in n]
    return sorted(names)


def describe_collection(name: str, show_counts: bool, show_schema: bool) -> dict:
    """
    Construye un dict con metadatos de la colección de forma segura.
    """
    info: dict = {"name": name}
    try:
        col = Collection(name)
        if show_counts:
            try:
                info["num_entities"] = col.num_entities
            except Exception as e:
                info["num_entities_error"] = str(e)

        if show_schema:
            try:
                info["schema"] = [
                    {
                        "field": f.name,
                        "dtype": str(f.dtype),
                        "is_primary": getattr(f, "is_primary", False),
                        "auto_id": getattr(f, "auto_id", False),
                        "max_length": getattr(f, "max_length", None),
                        "dim": getattr(f, "dim", None),
                    }
                    for f in col.schema.fields
                ]
                # Índices (si existen)
                idx_list = []
                for idx in getattr(col, "indexes", []):
                    try:
                        idx_list.append(
                            {
                                "field_name": idx.field_name,
                                "index_name": idx.index_name,
                                "params": idx.params,
                            }
                        )
                    except Exception:
                        pass
                if idx_list:
                    info["indexes"] = idx_list
            except Exception as e:
                info["schema_error"] = str(e)
    except Exception as e:
        info["open_error"] = str(e)

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Lista colecciones de Milvus y opcionalmente muestra recuentos y esquema."
    )
    parser.add_argument("--host", default="localhost", help="Host de Milvus (por defecto: localhost)")
    parser.add_argument("--port", type=int, default=19530, help="Puerto de Milvus (por defecto: 19530)")
    parser.add_argument("--user", default=None, help="Usuario (si Milvus tiene auth habilitada)")
    parser.add_argument("--password", default=None, help="Password (si Milvus tiene auth habilitada)")
    parser.add_argument("--secure", action="store_true", help="Activa TLS (si tu Milvus está con HTTPS/TLS)")
    parser.add_argument("--prefix", default=None, help="Filtrar solo colecciones cuyo nombre empieza con este prefijo")
    parser.add_argument("--contains", default=None, help="Filtrar solo colecciones que contengan esta subcadena")
    parser.add_argument("--show-counts", action="store_true", help="Muestra número de entidades por colección")
    parser.add_argument("--show-schema", action="store_true", help="Muestra campos del esquema e índices (si hay)")
    parser.add_argument("--as-json", action="store_true", help="Imprime la salida en formato JSON")

    args = parser.parse_args()

    try:
        connect_milvus(args.host, args.port, args.user, args.password, args.secure)
    except MilvusException as e:
        print(f"[ERROR] No se pudo conectar a Milvus: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Fallo inesperado de conexión: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        names = list_collections(args.prefix, args.contains)
    except Exception as e:
        print(f"[ERROR] No se pudieron listar colecciones: {e}", file=sys.stderr)
        sys.exit(3)

    if args.as_json:
        # Construir salida detallada
        out = [describe_collection(n, args.show_counts, args.show_schema) for n in names]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # Salida legible
    if not names:
        print("No hay colecciones que coincidan con el filtro actual.")
        return

    print("Colecciones en Milvus:")
    for n in names:
        line = f"- {n}"
        if args.show_counts:
            try:
                count = Collection(n).num_entities
                line += f"  (entidades: {count})"
            except Exception as e:
                line += f"  (error al obtener entidades: {e})"
        print(line)

    if args.show_schema:
        print("\nEsquemas:")
        for n in names:
            info = describe_collection(n, show_counts=False, show_schema=True)
            if "schema" in info:
                print(f"\n[nombre: {n}]")
                for f in info["schema"]:
                    print(
                        f"  • {f['field']} | {f['dtype']}"
                        f"{' | PRIMARY KEY' if f.get('is_primary') else ''}"
                        f"{' | auto_id' if f.get('auto_id') else ''}"
                        f"{f' | max_length={f['max_length']}' if f.get('max_length') else ''}"
                        f"{f' | dim={f['dim']}' if f.get('dim') else ''}"
                    )
                if info.get("indexes"):
                    print("  Índices:")
                    for idx in info["indexes"]:
                        print(f"    - {idx['index_name']} (campo={idx['field_name']}, params={idx['params']})")
            else:
                print(f"\n[nombre: {n}] (no se pudo leer esquema: {info.get('schema_error', 'desconocido')})")


if __name__ == "__main__":
    main()
