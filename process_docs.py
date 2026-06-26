"""Process dataset documents with the preprocessing pipeline and save them to JSON.

Each output record looks like:
    {"doc_id": "...", "raw_text": "...", "tokens": ["...", ...]}

Usage (defaults shown):
    python process_docs.py
    python process_docs.py --dataset "argsme/1.0/touche-2020-task-1/uncorrected" --limit 5000
    python process_docs.py --limit 0          # 0 means process ALL documents (can be huge / slow)
"""

import argparse
import json

import ir_win_fix  # noqa: F401  Windows fix for ir_datasets temp-file rename bug (apply before loading datasets)
import ir_datasets

from preprocessing import preprocess_text


def _doc_text(doc):
    """Return the best available text for an ir_datasets document.

    Every ir_datasets doc type implements default_text(); fall back to common
    attributes just in case.
    """
    if hasattr(doc, 'default_text'):
        return doc.default_text()
    for attr in ('text', 'body', 'title'):
        if hasattr(doc, attr):
            return getattr(doc, attr)
    return ''


def process_dataset_to_json(dataset_name,
                            out_path,
                            limit=1000,
                            keep_raw=True,
                            **preprocess_kwargs):
    """Iterate a dataset's docs, preprocess each, and write the results to JSON.

    Args:
        dataset_name: ir_datasets identifier (e.g. 'argsme/1.0/touche-2020-task-1/uncorrected').
        out_path: path of the JSON file to write.
        limit: max number of docs to process; 0 or None processes everything.
        keep_raw: include the original document text in each record.
        preprocess_kwargs: forwarded to preprocess_text (e.g. do_stemming=True).
    """
    print(f"Loading dataset: {dataset_name} ...")
    dataset = ir_datasets.load(dataset_name)

    records = []
    processed = 0
    for doc in dataset.docs_iter():
        raw_text = _doc_text(doc)
        tokens = preprocess_text(raw_text, **preprocess_kwargs)

        record = {'doc_id': doc.doc_id, 'tokens': tokens}
        if keep_raw:
            record['raw_text'] = raw_text
        records.append(record)

        processed += 1
        if processed % 1000 == 0:
            print(f"  processed {processed} docs ...")
        if limit and processed >= limit:
            break

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(records)} processed docs to {out_path}")
    return out_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset docs and save to JSON.")
    parser.add_argument('--dataset', default='argsme/1.0/touche-2020-task-1/uncorrected',
                        help="ir_datasets identifier")
    parser.add_argument('--out', default=None,
                        help="output JSON path (defaults to <dataset-slug>_processed.json)")
    parser.add_argument('--limit', type=int, default=1000,
                        help="max docs to process; 0 = all (can be very large)")
    parser.add_argument('--no-raw', action='store_true',
                        help="do not store the original text, only tokens")
    parser.add_argument('--no-stemming', action='store_true',
                        help="disable stemming (on by default)")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    out_path = args.out
    if out_path is None:
        slug = args.dataset.replace('/', '_').replace('.', '-')
        out_path = f"{slug}_processed.json"

    process_dataset_to_json(
        args.dataset,
        out_path,
        limit=args.limit,
        keep_raw=not args.no_raw,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        do_stemming=not args.no_stemming,
    )
