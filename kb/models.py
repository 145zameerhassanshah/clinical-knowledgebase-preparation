from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
@dataclass
class ExtractedUnit:
    unit_type:str; index:int; text:str; metadata:Dict[str,Any]=field(default_factory=dict)
@dataclass
class ExtractedDocument:
    document_id:str; source_path:str; file_name:str; file_type:str; extraction_method:str; text:str; units:List[ExtractedUnit]; metadata:Dict[str,Any]=field(default_factory=dict); warnings:List[str]=field(default_factory=list)
@dataclass
class Section:
    section_id:str; document_id:str; title:str; level:int; text:str; metadata:Dict[str,Any]=field(default_factory=dict)
@dataclass
class Chunk:
    chunk_id:str; document_id:str; section_id:str; section_title:str; chunk_index:int; text:str; source_file:str; source_type:str; extraction_method:str; metadata:Dict[str,Any]=field(default_factory=dict)
    def to_dict(self): return asdict(self)
