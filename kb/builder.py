from pathlib import Path
import json,csv,statistics
from .extractors import extract_document
from .sectioning import build_sections
from .chunking import structure_aware_chunks
SUPPORTED={'.pdf','.docx','.txt','.html','.htm','.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}
def build_knowledgebase(input_dir,output_dir):
 inp=Path(input_dir); outp=Path(output_dir); outp.mkdir(parents=True,exist_ok=True)
 docs=[]; secs=[]; chunks=[]; failures=[]; files=[p for p in inp.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED]
 for p in files:
  try:
   doc=extract_document(str(p)); docs.append({'document_id':doc.document_id,'file_name':doc.file_name,'file_type':doc.file_type,'source_path':doc.source_path,'extraction_method':doc.extraction_method,'metadata':doc.metadata,'warnings':doc.warnings,'text_char_count':len(doc.text)})
   for s in build_sections(doc):
    secs.append({'section_id':s.section_id,'document_id':s.document_id,'title':s.title,'level':s.level,'text':s.text,'char_count':len(s.text),'metadata':s.metadata})
    chunks.extend([c.to_dict() for c in structure_aware_chunks(doc,s)])
  except Exception as e: failures.append({'file':str(p),'error':str(e)})
 (outp/'documents.json').write_text(json.dumps(docs,indent=2,ensure_ascii=False),encoding='utf-8')
 (outp/'sections.json').write_text(json.dumps(secs,indent=2,ensure_ascii=False),encoding='utf-8')
 with (outp/'chunks.jsonl').open('w',encoding='utf-8') as f:
  for r in chunks: f.write(json.dumps(r,ensure_ascii=False)+'\n')
 with (outp/'chunks.csv').open('w',newline='',encoding='utf-8-sig') as f:
  fields=['chunk_id','document_id','section_id','section_title','chunk_index','text','source_file','source_type','extraction_method','char_count']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for r in chunks: w.writerow({**{k:r[k] for k in fields if k!='char_count'},'char_count':r['metadata']['char_count']})
 sizes=[r['metadata']['char_count'] for r in chunks]
 report={'documents_discovered':len(files),'documents_processed':len(docs),'sections_created':len(secs),'chunks_created':len(chunks),'failed_documents':len(failures),'average_chunk_chars':round(statistics.mean(sizes),2) if sizes else 0,'smallest_chunk_chars':min(sizes) if sizes else 0,'largest_chunk_chars':max(sizes) if sizes else 0,'retrieval_ready':bool(chunks),'failures':failures}
 (outp/'build_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8'); return report
