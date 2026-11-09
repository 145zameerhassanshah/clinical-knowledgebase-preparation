from pathlib import Path
import mimetypes,re,hashlib,shutil
import pymupdf
from docx import Document
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from .models import ExtractedDocument,ExtractedUnit
IMAGE_EXTS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}
def slugify(t): return re.sub(r'[^a-zA-Z0-9]+','_',t.lower()).strip('_') or 'document'
def make_document_id(path):
 p=Path(path); return f"{slugify(p.stem)}_{hashlib.sha1(str(p).encode()).hexdigest()[:8]}"
def configure_tesseract():
 for c in [shutil.which('tesseract'),r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe']:
  if c and Path(c).exists(): pytesseract.pytesseract.tesseract_cmd=str(c); return str(c)
 return None
def detect_file_type(path):
 ext=Path(path).suffix.lower()
 if ext=='.pdf': return 'pdf'
 if ext=='.docx': return 'docx'
 if ext=='.txt': return 'txt'
 if ext in {'.html','.htm'}: return 'html'
 if ext in IMAGE_EXTS: return 'image'
 raise ValueError(f'Unsupported file type: {Path(path).name}')
def extract_pdf(path,min_text_chars=40):
 doc=pymupdf.open(path); units=[]; full=[]; warnings=[]; ocr_pages=0; tess=configure_tesseract()
 for idx,page in enumerate(doc,start=1):
  text=page.get_text('text').strip(); method='pymupdf_text'
  if len(text)<min_text_chars and tess:
   pix=page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False); image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples); text=pytesseract.image_to_string(image).strip(); method='tesseract_ocr'; ocr_pages+=1
  elif len(text)<min_text_chars: warnings.append(f'Page {idx} appears scan-like; Tesseract not available.')
  units.append(ExtractedUnit('page',idx,text,{'page_number':idx,'method':method})); full.append(text)
 meta={'page_count':len(doc),'ocr_pages':ocr_pages,'pdf_metadata':dict(doc.metadata or {})}; doc.close()
 return ExtractedDocument(make_document_id(path),str(path),Path(path).name,'pdf','PyMuPDF + optional Tesseract OCR fallback','\n\n'.join(full),units,meta,warnings)
def extract_docx(path):
 doc=Document(path); units=[]; full=[]
 for idx,p in enumerate(doc.paragraphs,start=1):
  text=p.text.strip()
  if text: units.append(ExtractedUnit('paragraph',idx,text,{'style':p.style.name if p.style else None})); full.append(text)
 cp=doc.core_properties; meta={'title':cp.title,'author':cp.author,'subject':cp.subject,'created':str(cp.created) if cp.created else None,'modified':str(cp.modified) if cp.modified else None}
 return ExtractedDocument(make_document_id(path),str(path),Path(path).name,'docx','python-docx','\n\n'.join(full),units,meta,[])
def extract_txt(path):
 p=Path(path); warnings=[]
 try: text=p.read_text(encoding='utf-8'); enc='utf-8'
 except UnicodeDecodeError: text=p.read_text(encoding='latin-1'); enc='latin-1'; warnings.append('UTF-8 failed; latin-1 fallback used.')
 return ExtractedDocument(make_document_id(path),str(path),p.name,'txt','Python text reader',text.strip(),[ExtractedUnit('text',1,text.strip(),{})],{'encoding':enc},warnings)
def extract_html(path):
 p=Path(path); soup=BeautifulSoup(p.read_text(encoding='utf-8',errors='replace'),'html.parser')
 for tag in soup(['script','style','noscript']): tag.decompose()
 units=[]; full=[]; idx=1
 for tag in soup.find_all(['h1','h2','h3','p','li']):
  text=tag.get_text(' ',strip=True)
  if text: units.append(ExtractedUnit(tag.name,idx,text,{'html_tag':tag.name})); full.append(text); idx+=1
 title=soup.title.get_text(' ',strip=True) if soup.title else None
 return ExtractedDocument(make_document_id(path),str(path),p.name,'html','BeautifulSoup','\n\n'.join(full),units,{'title':title},[])
def extract_image(path):
 if not configure_tesseract(): raise RuntimeError('Tesseract OCR is required for image extraction.')
 p=Path(path)
 with Image.open(path) as image:
  text=pytesseract.image_to_string(image).strip(); meta={'width':image.width,'height':image.height,'format':image.format,'mode':image.mode}
 return ExtractedDocument(make_document_id(path),str(path),p.name,'image','Tesseract OCR',text,[ExtractedUnit('image',1,text,meta)],meta,[])
def extract_document(path):
 k=detect_file_type(path)
 return {'pdf':extract_pdf,'docx':extract_docx,'txt':extract_txt,'html':extract_html,'image':extract_image}[k](path)
