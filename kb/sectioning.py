import re
from .models import Section
HEADING_RE=re.compile(r'^(?:\d+(?:\.\d+){0,3}\s+.+|[A-Z][A-Z\s/&-]{4,})$')
def build_sections(doc):
 sections=[]; counter=1
 if doc.file_type=='docx':
  current='Document Content'; level=1; buf=[]
  for u in doc.units:
   style=(u.metadata.get('style') or '').lower()
   if 'heading' in style:
    if buf:
     sections.append(Section(f'{doc.document_id}_s{counter:03d}',doc.document_id,current,level,'\n\n'.join(buf).strip(),{})); counter+=1; buf=[]
    current=u.text; m=re.search(r'(\d+)',style); level=int(m.group(1)) if m else 1
   else: buf.append(u.text)
  if buf or not sections: sections.append(Section(f'{doc.document_id}_s{counter:03d}',doc.document_id,current,level,'\n\n'.join(buf).strip() or doc.text,{}))
  return sections
 if doc.file_type=='html':
  current='Document Content'; level=1; buf=[]
  for u in doc.units:
   if u.unit_type in {'h1','h2','h3'}:
    if buf:
     sections.append(Section(f'{doc.document_id}_s{counter:03d}',doc.document_id,current,level,'\n\n'.join(buf).strip(),{})); counter+=1; buf=[]
    current=u.text; level=int(u.unit_type[-1])
   else: buf.append(u.text)
  if buf or not sections: sections.append(Section(f'{doc.document_id}_s{counter:03d}',doc.document_id,current,level,'\n\n'.join(buf).strip() or doc.text,{}))
  return sections
 current='Document Content'; buf=[]
 for line in [x.strip() for x in doc.text.splitlines()]:
  if not line: continue
  if HEADING_RE.match(line) and len(line)<=120:
   if buf:
    sections.append(Section(f'{doc.document_id}_s{counter:03d}',doc.document_id,current,1,'\n'.join(buf).strip(),{})); counter+=1; buf=[]
   current=line
  else: buf.append(line)
 if buf or not sections: sections.append(Section(f'{doc.document_id}_s{counter:03d}',doc.document_id,current,1,'\n'.join(buf).strip() or doc.text,{}))
 return sections
