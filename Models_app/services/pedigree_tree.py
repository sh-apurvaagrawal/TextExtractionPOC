from dataclasses import dataclass, field

@dataclass
class PedigreeTree:
    nodes: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)
    image_path: str = ""
    image_id : int = 0
