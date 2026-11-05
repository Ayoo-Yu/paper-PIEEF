"""Export all 67 reported paper items from the frozen public results.

This entry point never trains, benchmarks, downloads or contacts a service.
Tables export authoritative stored table cells, workbook sheets export stored
values without recalculation, schematics export editable final SVG, and
quantitative figures redraw the published numerical source values. Portable
redraw layout can differ from the retained figures/paper exports.
"""
from pathlib import Path
import argparse,csv,hashlib,json,re,sys,zipfile,xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
ITEMS=json.loads((ROOT/'results/paper/items.json').read_text(encoding='utf-8-sig'))
X='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def workbook_rows(sheet_name):
 with zipfile.ZipFile(ROOT/'results/Supplementary_Data_S1.xlsx') as z:
  wb=ET.fromstring(z.read('xl/workbook.xml'));rels={x.attrib['Id']:x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))};ss=[]
  if 'xl/sharedStrings.xml' in z.namelist():ss=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
  sh=next(s for s in wb.find(X+'sheets') if s.attrib['name']==sheet_name);target=rels[sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']];target=target.lstrip('/') if target.startswith('/') else 'xl/'+target;rows=[]
  for row in ET.fromstring(z.read(target)).iter(X+'row'):
   vals=[]
   for c in row.findall(X+'c'):
    i=0
    for t in re.match('[A-Z]+',c.attrib['r'])[0]:i=i*26+ord(t)-64
    while len(vals)<i:vals.append('')
    v=c.find(X+'v');val=v.text if v is not None and v.text is not None else ''.join(c.find(X+'is').itertext()) if c.find(X+'is') is not None else ''
    vals[i-1]=ss[int(val)] if c.attrib.get('t')=='s' else val
   rows.append(vals)
  return rows

def reproduce(item):
 operation=item['operation'];key=item['id'];previous={p:digest(ROOT/p) for p in item['outputs'] if (ROOT/p).exists()}
 if operation=='export_authoritative_table_cells':
  tables=json.loads((ROOT/'results/paper/table_cells.json').read_text(encoding='utf-8'));rows=tables[key]['rows'];p=ROOT/item['outputs'][0];p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('w',encoding='utf-8',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
  assert list(csv.reader(p.open(encoding='utf-8',newline='')))==rows
  outputs=item['outputs'];details={'rows':len(rows),'cell_equality':True}
 elif operation=='export_stored_workbook_cells':
  p=ROOT/item['outputs'][0];rows=workbook_rows(p.stem);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('w',encoding='utf-8',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
  assert list(csv.reader(p.open(encoding='utf-8',newline='')))==rows
  outputs=item['outputs'];details={'rows':len(rows),'cell_equality':True}
 elif operation=='export_editable_schematic_svg':
  sys.path.insert(0,str(ROOT/'scripts/figures'));from render_schematics import render
  outputs=render(key);details={'source_svg_byte_equality':(ROOT/outputs[0]).read_bytes()==(ROOT/item['inputs'][0]).read_bytes(),'rendered_emf':False}
 else:
  sys.path.insert(0,str(ROOT/'scripts/figures'));from public_figures import render
  outputs=render(key);details={'data_source':'published frozen source values','layout':'portable redraw; paper export retained separately'}
 result={'item':item['item'],'id':key,'operation':operation,'status':'PASS','inputs':[],'outputs':[],'details':details}
 for n in item['inputs']:
  p=ROOT/n
  if p.exists():result['inputs'].append({'path':n,'sha256':digest(p)})
 for n in outputs:
  n=Path(n).as_posix();p=ROOT/n;result['outputs'].append({'path':n,'bytes':p.stat().st_size,'sha256':digest(p),'unchanged_from_previous_export':previous.get(n)==digest(p) if n in previous else None})
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--item',default='all');p.add_argument('--all',action='store_true');p.add_argument('--from-results',action='store_true',required=True);p.add_argument('--report',default='results/reproduction_report.json');a=p.parse_args()
 selected=ITEMS if a.all or a.item=='all' else [v for v in ITEMS if v['id']==a.item]
 if not selected:p.error('unknown paper item')
 report={'training':False,'network':False,'items':[]}
 for item in selected:
  try:r=reproduce(item)
  except Exception as e:r={'item':item['item'],'id':item['id'],'status':'FAIL','error_type':type(e).__name__,'error':str(e).replace(str(ROOT),'repository')}
  report['items'].append(r);print(f"{r['id']}: {r['status']}",flush=True)
 report['passed']=sum(x['status']=='PASS' for x in report['items']);report['failed']=len(selected)-report['passed'];dest=ROOT/a.report;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(f"{report['passed']}/{len(selected)} paper items reproduced; {report['failed']} failed")
 return bool(report['failed'])
if __name__=='__main__':raise SystemExit(main())
