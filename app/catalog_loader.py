import json
from typing import List, Dict, Any
import re

def load_catalog(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("assessments", [])
    return data

def extract_skills(desc: str) -> List[str]:
    # Simple keyword extraction – extend as needed
    keywords = [
        "java", "python", ".net", "c#", "javascript", "typescript", "react", "angular",
        "vue", "node.js", "express", "django", "flask", "spring", "aws", "azure", "gcp",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "db2",
        "stakeholder", "communication", "leadership", "management", "agile", "scrum",
        "waterfall", "devops", "ci/cd", "docker", "kubernetes", "jenkins",
        "data analysis", "excel", "power bi", "tableau", "machine learning",
        "deep learning", "nlp", "computer vision", "embedded", "iot"
    ]
    found = []
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', desc.lower()):
            found.append(kw)
    return found

def enrich_assessment(assess: Dict) -> Dict:
    """Add computed fields for retrieval."""
    text_parts = [
        assess.get("title", ""),
        ", ".join(assess.get("test_types", [])),
        ", ".join(assess.get("job_levels", [])),
        assess.get("description", "")
    ]
    assess["embedding_text"] = " | ".join(filter(None, text_parts)).lower()
    assess["skills"] = extract_skills(assess.get("description", ""))
    return assess

def load_and_prepare(filepath: str) -> List[Dict]:
    catalog = load_catalog(filepath)
    return [enrich_assessment(a) for a in catalog]