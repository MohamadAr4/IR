import ir_datasets

def download_dataset(dataset_name):
    print(f"\n Downlaoding the Data set : {dataset_name} ...")
    try:
        dataset = ir_datasets.load(dataset_name)

        print("Now I am downloading the Documnents..")
        docs_count = dataset.docs_count()
        print(f"Now after scaning the number of the docs is : {docs_count}")
        
        print("Now I am saving the queries..")
        queries_count = sum(1 for _ in dataset.queries_iter())
        print(f"I just downloaded the queries and the count is : {queries_count}")
        
        print("Now I am downloading the Qrels..")
        qrels_count = sum(1 for _ in dataset.qrels_iter())
        print(f"I just downloaded the Qrels and the count is : {qrels_count}")
        
        print(f"The data set is ready {dataset_name}")
        
    except Exception as e:
        print(f"there was an error with this data set : {dataset_name}: {e}")

download_dataset("argsme/1.0/touche-2020-task-1/uncorrected")