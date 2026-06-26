import ir_win_fix  # noqa: F401  Windows fix for ir_datasets temp-file rename bug (apply before loading datasets)
import ir_datasets

print("Checking ArgsMe dataset...")
try:
    argsme = ir_datasets.load("argsme/1.0/touche-2020-task-1/uncorrected")
    print(f"ArgsMe is ready. Number of documents available to read: {argsme.docs_count()}")
    
    q = next(argsme.queries_iter())
    print(f"Sample query ID: {q.query_id}")
    # ArgsMe queries usually support fallback attributes
    query_text = getattr(q, 'title', getattr(q, 'text', 'No text field found'))
    print(f"Sample query content: {query_text}")
except Exception as e:
    print(f"Error with ArgsMe: {e}")