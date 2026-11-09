import re
from .models import Chunk
def split_sentences(text): return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9•])',text) if s.strip()]
def structure_aware_chunks(doc,section,target_chars=900,overlap_sentences=1):
 sentences=split_sentences(section.text)
 if not sentences: return []
 groups=[]; current=[]
 for s in sentences:
  cand=' '.join(current+[s]).strip()
  if current and len(cand)>target_chars:
   groups.append(current); current=(current[-overlap_sentences:] if overlap_sentences else [])+[s]
  else: current.append(s)
 if current: groups.append(current)
 out=[]
 for idx,g in enumerate(groups,start=1):
  text=' '.join(g).strip()
  if text: out.append(Chunk(f'{section.section_id}_c{idx:03d}',doc.document_id,section.section_id,section.title,idx,text,doc.file_name,doc.file_type,doc.extraction_method,{'section_level':section.level,'char_count':len(text),'document_metadata':doc.metadata,'document_warnings':doc.warnings}))
 return out
