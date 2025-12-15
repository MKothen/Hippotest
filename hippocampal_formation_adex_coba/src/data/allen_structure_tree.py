from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

ALLEN_STRUCTURE_TREE_URL = (
    "https://api.brain-map.org/api/v2/structure_graph_download/1.json"
)
# Graph 1 = Allen Mouse Brain Atlas structure graph.

@dataclass
class AllenStructureTree:
    id_to_children: Dict[int, List[int]]

    @staticmethod
    def _cache_path(cache_dir: Path) -> Path:
        return cache_dir / "allen_structure_tree_graph_1.json"

    @classmethod
    def load(cls, cache_dir: Path) -> "AllenStructureTree":
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cls._cache_path(cache_dir)
        if not p.exists():
            r = requests.get(ALLEN_STRUCTURE_TREE_URL, timeout=60)
            r.raise_for_status()
            p.write_text(r.text, encoding="utf-8")

        data = json.loads(p.read_text(encoding="utf-8"))

        # The payload is a tree of nodes; each node has 'id' and 'children'
        # We build adjacency for fast descendant queries.
        id_to_children: Dict[int, List[int]] = {}

        def walk(node):
            nid = int(node["id"])
            kids = node.get("children", []) or []
            id_to_children[nid] = [int(ch["id"]) for ch in kids]
            for ch in kids:
                walk(ch)

        # Some responses wrap at different keys, handle robustly:
        root = None
        if isinstance(data, dict) and "msg" in data:
            root = data["msg"]
        elif isinstance(data, list):
            root = data
        else:
            root = data

        # root can be list of nodes
        if isinstance(root, list):
            for n in root:
                walk(n)
        else:
            walk(root)

        return cls(id_to_children=id_to_children)

    def descendants(self, root_id: int) -> List[int]:
        out: List[int] = []
        stack = list(self.id_to_children.get(int(root_id), []))
        seen = set(stack)
        while stack:
            nid = stack.pop()
            out.append(nid)
            for ch in self.id_to_children.get(nid, []):
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
        return out
