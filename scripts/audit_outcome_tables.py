"""Read archived tables, distinguish access failure from a passing test.

NonDex's header has more fields than data rows. Use only the first five
documented positions, then validate failures against original diagnostics.
"""
import collections, csv, io, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
csv.field_size_limit(10_000_000)

def parse_table(text,meta):
    rows=list(csv.reader(io.StringIO(text)));out=[];issues=[]
    if not rows: return out,['empty_table']
    if rows[0][:2]==['Iteration','Execution ID']:
        for n,row in enumerate(rows[1:],2):
            if len(row)<5: issues.append(f'short_row_{n}');continue
            try: iteration=int(row[0])
            except ValueError: issues.append(f'bad_iteration_{n}');continue
            out.append(dict(iteration=iteration,status=row[4].strip().lower(),execution_id=row[1],xml_path=row[3],row=n,scope='NonDex report; target testcase needs validation'))
    else:
        target=meta['configured_test'].replace('#','.')
        for n,row in enumerate(rows,1):
            if len(row)<4: issues.append(f'short_row_{n}');continue
            if row[0]!=target and not re.fullmatch(re.escape(target)+r'\[[^\r\n]*\]',row[0]):
                issues.append(f'target_name_mismatch_row_{n}');continue
            item=dict(status=row[1].strip().lower(),row=n,scope='exact target')
            if row[3].lstrip().startswith('{'):
                try: item.update(embedded_nio=json.loads(row[3]),iteration=n-1)
                except json.JSONDecodeError: issues.append(f'bad_json_{n}');continue
            else:
                try: item['iteration']=int(row[3])
                except ValueError: issues.append(f'bad_iteration_{n}');continue
                item['xml_path']=row[4] if len(row)>4 else None
            out.append(item)
    return out,issues
