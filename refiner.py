import re
import textwrap
import tree_sitter_language_pack          # same dependency you already use


import re
import tree_sitter_language_pack

def _text(node, doc_bytes):
    return doc_bytes[node.start_byte:node.end_byte].decode("utf8")

def analyze_snippet(document_text: str, snippet: str, language: str = "python"):
    """
    Language‑agnostic snippet analyzer using only Tree‑sitter AST:
      - finds the enclosing function/method & optional parent class/struct
      - builds an import‑table by spotting any AST node whose type name
        contains “import”, “include”, or “use”
      - detects internal deps via “self”/“this” member accesses
      - detects external deps by matching those against the import table
    """
    # 1) parse
    parser = tree_sitter_language_pack.get_parser(language)
    tree   = parser.parse(document_text.encode("utf8"))
    root   = tree.root_node
    doc_b  = document_text.encode("utf8")

    # 2) locate snippet in bytes
    snip_b = snippet.encode("utf8")
    start  = doc_b.find(snip_b)
    if start < 0:
        raise ValueError("Snippet not found verbatim")
    end = start + len(snip_b)

    # 3) find smallest covering function‑like node
    def is_func_like(n):
        return "function" in n.type or "method" in n.type
    snippet_node = None
    def walk(n):
        nonlocal snippet_node
        if n.start_byte <= start and end <= n.end_byte:
            if is_func_like(n):
                if (snippet_node is None or
                    (n.end_byte-n.start_byte) < (snippet_node.end_byte-snippet_node.start_byte)):
                    snippet_node = n
            for c in n.named_children:
                walk(c)
    walk(root)
    if snippet_node is None:
        raise ValueError("No enclosing function/method found")

    # 4) extract func name & optional class
    name_n = snippet_node.child_by_field_name("name")
    func_name = _text(name_n, doc_b) if name_n else "<lambda>"

    # climb up for a class/struct
    def is_type_def(n):
        return bool(re.search(r"(class|struct|interface|enum)", n.type, re.I))
    cls_node, p = None, snippet_node.parent
    while p:
        if is_type_def(p):
            cls_node = p
            break
        p = p.parent
    if cls_node:
        cn = cls_node.child_by_field_name("name")
        class_name = _text(cn, doc_b)
        includes = [f"{class_name}.{func_name}"]
    else:
        includes = [func_name]

    # 5) build import table from any node whose type mentions import/include/use
    import_table = {}
    def extract_imports(n):
        text = document_text[n.start_byte:n.end_byte]
        # local parse just for this node:
        # pull out dotted names and identifiers
        for tok in re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", text):
            alias = tok.split(".")[-1]
            import_table[alias] = tok
    stack = [root]
    while stack:
        node = stack.pop()
        if re.search(r"(import|include|use)", node.type, re.I):
            extract_imports(node)
        stack.extend(node.named_children)

    # 6) gather snippet parameters to avoid false positives
    params = set()
    for n in snippet_node.named_children:
        if "parameter" in n.type:
            for idn in n.named_children:
                if idn.type == "identifier":
                    params.add(_text(idn, doc_b))

    # 7) scan snippet subtree for deps
    internal, external = set(), set()
    def scan(n):
        # member accesses: any node whose type has 'member'|'field'|'attribute'
        if re.search(r"(member|field|attribute|property)", n.type, re.I):
            # grab object & property/name children
            obj  = n.child_by_field_name("object") or n.child_by_field_name("operand")
            prop = n.child_by_field_name("property") or n.child_by_field_name("name")
            if obj and prop:
                o, p = _text(obj, doc_b), _text(prop, doc_b)
                if o in ("self", "this"):
                    internal.add(f"{o}.{p}")
                elif o in import_table:
                    external.add(f"{o}.{p}")
        # bare identifier
        elif n.type == "identifier":
            txt = _text(n, doc_b)
            if txt in import_table and txt not in params:
                external.add(import_table[txt])
        for c in n.named_children:
            scan(c)
    scan(snippet_node)

    return {
        "includes":       includes,
        "internal_deps":  sorted(internal),
        "external_deps":  sorted(external),
        "imports":        import_table,
        "snippet_node":   snippet_node,
        "class_node":     cls_node,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Augmenter – builds the *minimal* context
# ──────────────────────────────────────────────────────────────────────────────
def augment_snippet(document_text: str,
                    snippet      : str,
                    language     : str = "python") -> str:
    """
    Returns a trimmed‑down, executable slice of the original document that:
      • keeps *just* the import lines needed by the snippet
      • keeps the parent class / struct *header* (+ a stubbed __init__)
      • keeps the snippet unchanged
      • replaces everything else with '# ...'
    """
    ctx = analyze_snippet(document_text, snippet, language)

    # -- 1.  pick only the import lines that define the external deps ------------
    needed_symbols  = {d.split('.')[0] for d in ctx["external_deps"]}
    import_lines, seen = [], set()
    for line in document_text.splitlines():
        if (line.lstrip().startswith(("import", "from")) and
            any(re.search(rf"\b{re.escape(sym)}\b", line) for sym in needed_symbols)):
            if line not in seen:
                import_lines.append(line.rstrip())
                seen.add(line)

    # -- 2a.  If snippet lives in a class, carve that class out ------------------
    if ctx["class_node"]:
        cls          = ctx["class_node"]
        lines        = document_text.splitlines()
        cls_start_ln = document_text[:cls.start_byte].count('\n')
        cls_end_ln   = document_text[:cls.end_byte].count('\n')
        cls_block    = lines[cls_start_ln:cls_end_ln + 1]

        # keep only:  class header,  __init__ header + attr assignments needed,
        #             snippet function,  '# ...' placeholders
        indent       = len(cls_block[0]) - len(cls_block[0].lstrip())
        needed_attrs = [a.split('.')[1] for a in ctx["internal_deps"]]

        new_block, in_init, in_target = [], False, False
        target_def_pat = re.compile(rf"\s*def\s+{re.escape(ctx['includes'][0].split('.')[-1])}\b")

        for ln in cls_block:
            stripped = ln.lstrip()

            # class line   -------------------------------------------------------
            if stripped.startswith("class "):
                new_block.append(ln.rstrip())
                continue

            # __init__ header ----------------------------------------------------
            if re.match(r"\s*def\s+__init__\b", stripped):
                in_init = True
                new_block.append(ln.rstrip())
                new_block.append(" " * (indent + 4) + "# ...")
                continue

            # other method header ends __init__
            if in_init and stripped.startswith("def "):
                in_init = False

            # collect attribute assignments we actually need
            if in_init and any(f"self.{attr}" in stripped for attr in needed_attrs):
                # place them *before* the '# ...' we just inserted
                new_block.insert(-1, ln.rstrip())
                continue

            # target function header --------------------------------------------
            if target_def_pat.match(stripped):
                in_target = True
                new_block.append(ln.rstrip())
                continue

            # inside target func – keep verbatim until dedent
            if in_target:
                new_block.append(ln.rstrip())
                # dedent back to class level?
                if (ln.startswith(" " * indent) and ln.lstrip().startswith("def ") and
                        not target_def_pat.match(ln.lstrip())):
                    in_target = False
                continue

        # add '# ...' after class header if we ended up with only header+snippet
        if len(new_block) > 2 and not new_block[1].strip():
            new_block.insert(1, " " * (indent + 4) + "# ...")

        parent_chunk = "\n".join(new_block)

    # -- 2b.  top‑level function (no class) --------------------------------------
    else:
        parent_chunk = snippet

    # -- 3.  Stitch the parts together -------------------------------------------
    pieces = []
    if import_lines:
        pieces.extend(import_lines)
        pieces.append("# ...")

    pieces.append(parent_chunk)
    return "\n".join(pieces).rstrip() + "\n"     # final newline for POSIX‑y style



if __name__ == '__main__':
    doc = """
from utilities import format_complex
class ComplexNumber:
    \"\"\"Represents a complex number with real and imaginary parts.\"\"\"
    def __init__(self, real, imag, dick_size):
        self.dick_size = dick_size
        self.real = real
        self.imag = imag

    def modulus(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def add(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def multiply(self, other):
        new_real = self.real * other.real - self.imag * other.imag 
        new_imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(new_real, new_imag)

    def __str__(self): 
        return format_complex(self.real, self.imag)
        """

    # snipper is def __str__(self):
    snip = 'def __str__' + doc.split('def __str__')[1].strip()
    print('snippet:', snip)

    print(analyze_snippet(doc, snip, 'python'))
    print(augment_snippet(doc, snip))

    # return;
    # ctx = analyze_snippet(doc, snip, 'python')
    # print("includes:", ctx['includes'])
    # print("internal_deps:", ctx['internal_deps'])
    # print("external_deps:", ctx['external_deps'])