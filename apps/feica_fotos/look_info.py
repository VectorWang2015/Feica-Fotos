"""Verbatim official FOTOS descriptions; no agent-written Look copy fallback."""
import json
from pathlib import Path
from .catalog import LOOKS, LOOK_PREVIEW

OFFICIAL_TEXT_PATH=Path(__file__).resolve().parent/'assets'/'official-look-descriptions.json'
PREVIEW_NOTICE='本滤镜为 preview，其强度机制可能与官方有偏差。'

def _official_records():
    try:
        data=json.loads(OFFICIAL_TEXT_PATH.read_text(encoding='utf-8'))
        records=data.get('records',{})
        return records if isinstance(records,dict) else {}
    except (OSError,ValueError,TypeError):
        return {}

records=_official_records()
LOOK_INFO={}
for spec in LOOKS:
    record=records.get(spec.id,{})
    body=record.get('description') if isinstance(record,dict) else None
    LOOK_INFO[spec.id]={
        'title':spec.title,
        'body':body if isinstance(body,str) else '',
        'official_available':isinstance(body,str) and bool(body),
        'official_name':record.get('name') if isinstance(record,dict) else None,
        'official_locale':record.get('locale') if isinstance(record,dict) else None,
        'resource_key':record.get('resource_key') if isinstance(record,dict) else None,
        'preview':LOOK_PREVIEW[spec.id],
    }
