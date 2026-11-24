"""Build a 3-domain byte corpus (prose / code / json) from locally available files."""
import glob, os, random, json
random.seed(0)
out = os.path.join(os.path.dirname(__file__))
def take(files, limit):
    buf = bytearray()
    for f in files:
        try: b = open(f, 'rb').read()
        except Exception: continue
        buf += b + b'\n'
        if len(buf) >= limit: break
    return bytes(buf[:limit])
prose = open(os.path.join(out, 'raw/shakespeare.txt'), 'rb').read()
code_files = sorted(glob.glob('/usr/lib/python3.13/**/*.py', recursive=True)); random.shuffle(code_files)
code = take(code_files, 3_000_000)
js = sorted(f for f in glob.glob('/usr/**/*.json', recursive=True) + glob.glob('/root/.local/**/*.json', recursive=True) + glob.glob('/root/.rustup/**/*.json', recursive=True) if os.path.getsize(f) > 2000)
random.shuffle(js)
jsonb = take(js, 3_000_000)
for name, b in [('prose', prose), ('code', code), ('json', jsonb)]:
    n = len(b); cut = int(n * 0.95)
    open(os.path.join(out, f'{name}.train'), 'wb').write(b[:cut])
    open(os.path.join(out, f'{name}.val'), 'wb').write(b[cut:])
    print(name, n, 'train', cut, 'val', n - cut)
