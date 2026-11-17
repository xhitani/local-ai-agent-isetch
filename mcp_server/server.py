"""Serveur MCP de l'assistant de scolarité (transport stdio).
Il expose des outils de LECTURE et de GÉNÉRATION de brouillon.
Il n'a volontairement AUCUN outil d'envoi : l'envoi reste une décision humaine."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))

from mcp.server.fastmcp import FastMCP

import tools as T

mcp = FastMCP("scolarite-iset")


@mcp.tool()
def get_calendrier(module: str) -> str:
    """Dates d'examen et de rattrapage d'un module (session de janvier 2027)."""
    return T.get_calendrier(module)


@mcp.tool()
def search_reglement(query: str) -> str:
    """Cherche dans le règlement des études (absences, certificat médical,
    rattrapage, stages, réclamations). Renvoie les articles pertinents."""
    return T.search_reglement(query)


@mcp.tool()
def certificat_medical(module: str, hospitalisation: bool = False) -> str:
    """Étudiant malade ou absent à un examen : date limite du certificat médical,
    date de rattrapage et démarche."""
    return T.certificat_medical(module, hospitalisation)


@mcp.tool()
def verifier_module(module: str) -> str:
    """Vérifie qu'un module a une session de rattrapage. Renvoie du JSON."""
    return T.verifier_module(module)


@mcp.tool()
def generate_demande(module: str, nom: str, numero_etudiant: str,
                     hospitalisation: bool = False) -> str:
    """Génère le BROUILLON d'une demande de rattrapage (Article 8). N'envoie rien."""
    return T.generate_demande(module, nom, numero_etudiant, hospitalisation)


if __name__ == "__main__":
    mcp.run(transport="stdio")
