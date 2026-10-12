"""Group detection phrases supplied by the conversational agent; no model calls."""


def normalize(label):
    return " ".join(label.lower().replace("_", " ").replace("-", " ").split())


def build_prompt(labels, hints=None):
    hints = {} if hints is None else hints
    if not isinstance(hints, dict):
        raise ValueError("detection_hints 必须是物体名到英文描述列表的映射")
    cleaned = {}
    for label, phrases in hints.items():
        if not isinstance(label, str) or normalize(label) not in labels:
            raise ValueError("detection_hints 的物体名必须属于本次扫描")
        if (not isinstance(phrases, list) or
                any(not isinstance(p, str) or not p.strip() or '.' in p for p in phrases)):
            raise ValueError("detection_hints 的每个值必须是独立英文描述的列表")
        cleaned.setdefault(normalize(label), []).extend(normalize(p) for p in phrases)
    # Broad colour phrases cannot identify two targets of that same colour.
    generic_words = {"fruit", "fruits", "object", "objects", "vegetable", "vegetables", "item", "thing"}
    is_generic = lambda p: len(p.split())>1 and p.split()[-1] in generic_words
    prefixes = {p.rsplit(' ', 1)[0] for phrases in cleaned.values() for p in phrases if is_generic(p)}
    owners = {prefix: {label for label in labels if any(p.startswith(prefix+' ')
                      for p in [label]+cleaned.get(label, []))} for prefix in prefixes}
    aliases, phrases = {}, list(labels)
    for label in labels:
        names, generic = [label], []
        for phrase in cleaned.get(label, []):
            if is_generic(phrase):
                if len(owners[phrase.rsplit(' ', 1)[0]])==1:
                    generic.append(phrase)
            else:
                names.append(phrase)
        names, generic = list(dict.fromkeys(names)), list(dict.fromkeys(generic))
        aliases[label] = {"names": names, "generic": generic}
        phrases += names+generic
    return '.'.join(dict.fromkeys(phrases)), aliases
