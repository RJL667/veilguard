"""Experiment: how much faster is redaction if we trim more of the spaCy
pipeline, and does PERSON/ORG/LOCATION detection survive?

EMAIL/PHONE/SA-ID/etc are regex recognizers — unaffected.  Only the
NER-derived entities (PERSON/ORG/LOCATION) could regress.  We measure
analyze time on a 19KB payload + check detection per config.
"""
import os, sys, time, tempfile
os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_trim_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii.redactor import PII_ENTITIES  # noqa: E402
from pii import get_redactor  # noqa: E402

r = get_redactor()
nlp = r.analyzer.nlp_engine.nlp["en"]
analyze = lambda t: r.analyzer.analyze(text=t, entities=PII_ENTITIES, language="en",
                                       score_threshold=r.min_score,
                                       allow_list=r.allow_list or None)

# 19KB realistic payload
NAMES = ["Alice Johnson","Brian Naidoo","Priya Pillay","Thabo Mokoena","Nomsa Khumalo"]
big = "\n".join(
    f"On the {i}th, {NAMES[i%5]} ({NAMES[i%5].split()[0].lower()}{i}@acme.co.za) at Acme Corporation "
    f"in Cape Town raised ticket {7000+i}; phone +27 21 555 {1000+i}; ID 800101500908{i%9}."
    for i in range(60))

# detection probes (entity we EXPECT)
PROBES = [
    ("Alice Johnson met the team", "PERSON"),
    ("Dr Fatima Patel reviewed it", "PERSON"),
    ("spoke to Thabo about the deal", "PERSON"),
    ("contact alice.johnson@acme.co.za today", "EMAIL_ADDRESS"),
    ("call +27 21 555 1234 now", "SA_PHONE_NUMBER"),
    ("my ID number is 8001015009087 ok", "SA_ID"),
    ("card 4111 1111 1111 1111 expires", "CREDIT_CARD"),
    ("server at 192.168.1.42 down", "IP_ADDRESS"),
]
def found_types(text):
    return {res.entity_type for res in analyze(text)}

def detect_report():
    out = []
    for text, want in PROBES:
        got = found_types(text)
        out.append(f"{want}={'OK' if want in got else 'MISS('+','.join(got or {'-'})+')'}")
    return "  ".join(out)

def timed():
    analyze(big)  # warm
    t = min(_t() for _ in range(3))
    return t
def _t():
    s = time.perf_counter(); analyze(big); return (time.perf_counter()-s)*1000

def cfg(name, disable):
    for p in disable:
        if p in nlp.pipe_names: nlp.disable_pipe(p)
    ms = timed()
    det = detect_report()
    print(f"{name:38s} active={nlp.pipe_names}")
    print(f"   19KB analyze: {ms:6.1f} ms   detection: {det}\n")
    return ms

print("(baseline = parser already dropped by the redactor)\n")
base = cfg("[A] current (parser dropped)", [])
t2   = cfg("[B] +drop tagger,attribute_ruler", ["tagger", "attribute_ruler"])
t3   = cfg("[C] +drop lemmatizer too", ["lemmatizer"])
print(f"speedup B vs A: {base/t2:.2f}x   C vs A: {base/t3:.2f}x")
print("WATCH: any 'MISS' on PERSON/ORG/LOCATION above means that trim hurts recall.")
