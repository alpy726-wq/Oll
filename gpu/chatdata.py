"""Real, human-written data for Oll-Chat (built on Modal into /vol/chat, or locally with --small for tests).
  general : FineWeb-Edu (shard 001, never seen by Oll-1) + English Wikipedia (one shard)          -> general.npy (bytes)
  dialog  : OpenAssistant oasst1 (English, best-ranked path of each human conversation tree),
            Dolly-15k (instruction, optional context, response; written by Databricks employees),
            SQuAD v1.1 train (Wikipedia paragraph, crowd-worker question, answer)                 -> dialog.npy (bytes)
  eval    : SQuAD v1.1 validation (held out; the model never trains on it)                         -> squad_val.json
Format of every exchange: "Document: ...\\n" (only when there is one) "User: ...\\nOll: ...\\n\\n"."""
import gzip, io, json, os, sys, urllib.request
import numpy as np

HF = "https://huggingface.co/datasets"
UA = {"User-Agent": "oll-research/0.1"}


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(path + ".part", "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.rename(path + ".part", path)
    return path


def ex(user, answer, doc=None):
    s = (f"Document: {doc.strip()}\n" if doc and doc.strip() else "") + f"User: {user.strip()}\nOll: {answer.strip()}\n\n"
    return s.encode("utf-8")


def build(out="/vol/chat", general_bytes=1_500_000_000, small=False):
    import pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    meta = {}
    # ---- dialog
    if not os.path.exists(f"{out}/dialog.npy"):
        parts = []
        tr = fetch(f"{HF}/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz", f"{out}/raw/oasst_trees.jsonl.gz")
        n_oa = 0
        for line in gzip.open(tr, "rt", encoding="utf-8"):
            t = json.loads(line)
            p = t["prompt"]
            if p.get("lang") != "en":
                continue
            conv, node = [], p
            while node is not None:  # follow the best-ranked reply at every turn
                conv.append((node["role"], node["text"]))
                reps = [r for r in node.get("replies", []) if r.get("text")]
                node = min(reps, key=lambda r: (r.get("rank") if r.get("rank") is not None else 99)) if reps else None
            s = b""
            for i in range(0, len(conv) - 1, 2):
                if conv[i][0] == "prompter" and conv[i + 1][0] == "assistant":
                    s += f"User: {conv[i][1].strip()}\nOll: {conv[i + 1][1].strip()}\n".encode("utf-8")
            if s:
                parts.append(s + b"\n"); n_oa += 1
        dl = fetch(f"{HF}/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl", f"{out}/raw/dolly.jsonl")
        n_do = 0
        for line in open(dl, encoding="utf-8"):
            d = json.loads(line); parts.append(ex(d["instruction"], d["response"], d.get("context"))); n_do += 1
        sq = fetch(json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://huggingface.co/api/datasets/rajpurkar/squad/parquet/plain_text/train", headers=UA)).read())[0], f"{out}/raw/squad_train.parquet")
        tab = pq.read_table(sq).to_pydict(); n_sq = 0
        for c, q, a in zip(tab["context"], tab["question"], tab["answers"]):
            parts.append(ex(q, a["text"][0], c)); n_sq += 1
        rng = np.random.default_rng(0); order = rng.permutation(len(parts))
        arr = np.frombuffer(b"".join(parts[i] for i in order), dtype=np.uint8)
        np.save(f"{out}/dialog.npy", arr)
        meta.update(oasst_conversations=n_oa, dolly=n_do, squad_train=n_sq)
    # ---- held-out evaluation questions
    if not os.path.exists(f"{out}/squad_val.json"):
        sv = fetch(json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://huggingface.co/api/datasets/rajpurkar/squad/parquet/plain_text/validation", headers=UA)).read())[0], f"{out}/raw/squad_val.parquet")
        tab = pq.read_table(sv).to_pydict()
        val = [dict(context=c, question=q, answers=a["text"]) for c, q, a in zip(tab["context"], tab["question"], tab["answers"])]
        json.dump(val, open(f"{out}/squad_val.json", "w"))
    # ---- general text
    if small and not os.path.exists(f"{out}/general.npy"):  # local tests: our committed book corpus instead of downloads
        np.save(f"{out}/general.npy", np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/v4/text_train.npz"))["bytes"][:20_000_000].astype(np.uint8))
    if not os.path.exists(f"{out}/general.npy"):
        srcs = [f"{HF}/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet"]
        if not small:
            srcs.insert(0, f"{HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/001_00000.parquet")
        buf, n = [], 0
        for k, url in enumerate(srcs):
            p = fetch(url, f"{out}/raw/general_{k}.parquet")
            pf = pq.ParquetFile(p)
            cap = general_bytes * (0.75 if (k == 0 and not small) else 1.0) if not small else 20_000_000
            got = 0
            for rg in range(pf.num_row_groups):
                for txt in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                    b = txt.encode("utf-8") + b"\n\n"; buf.append(b); n += len(b); got += len(b)
                if got >= cap:
                    break
        rng = np.random.default_rng(1); order = rng.permutation(len(buf))
        np.save(f"{out}/general.npy", np.frombuffer(b"".join(buf[i] for i in order), dtype=np.uint8))
    meta["general_bytes"] = int(np.load(f"{out}/general.npy", mmap_mode="r").shape[0])
    meta["dialog_bytes"] = int(np.load(f"{out}/dialog.npy", mmap_mode="r").shape[0])
    json.dump(meta, open(f"{out}/META.json", "w"), indent=1)
    return meta


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/vol/chat", small="--small" in sys.argv))
