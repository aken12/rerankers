import os
import argparse
from rerankers import Reranker
import srsly
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def qrel_reader(qrel_path):
    qrels = {}
    with open(qrel_path) as f:
        for res in f:
            qid,_,did,rel = res.strip().split('\t')
            if qid not in qrels:
                qrels[qid] = [{did:rel}]
            else:
                qrels[qid].append({did:rel})
    return qrels

def result_reader(result_path,topk=100):
    top100 = {}
    with open(result_path) as f:
        for res in f:
            qid,_,did,rank,_,_ = res.strip().split(' ')
            
            if int(rank) > topk:
                continue
            
            if qid not in top100:
                top100[qid] = [did]
            else:
                top100[qid].append(did) 
    return top100

def write_result(output_path,scores,reranker):
    with open(output_path,"w")as fw:
        for qid, ranked_results in scores.items():
            for rank,result in enumerate(ranked_results,start=1):
                fw.write(f'{qid} 0 {result.doc_id} {rank} {result.score} {reranker}\n')
                
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir',help='',default='/home/ace14788tj/extdisk/data/',type=str)
    parser.add_argument('--task',default="nq",type=str)
    parser.add_argument('--first_stage',default="bm25",type=str)
    parser.add_argument('--rewrite',default="default",type=str)
    parser.add_argument('--topk',default=100,type=int)
    parser.add_argument('--model_name',default="castorini/monot5-base-msmarco-10k",type=str)
    parser.add_argument('--debug',action="store_true")
    args = parser.parse_args()
    
    task = args.task
    first_stage = args.first_stage
    rewrite = args.rewrite
    model_name = args.model_name
    topk = args.topk
    
    if "monot5" in model_name:
        reranker = "monot5"
    else:
        reranker = "monobert"
    
    corpus_dir = os.path.join(args.data_dir,task,"corpus")
    query_path = os.path.join(args.data_dir,task,f"query/{rewrite}_query.json")
    result_path = os.path.join(args.data_dir,task,"runs",first_stage,f'{task}_{first_stage}_{rewrite}.txt')
    qrel_path = os.path.join(args.data_dir,task,"qrels","qrel.tsv")
    output_path = os.path.join(args.data_dir,task,"runs",reranker,f'{task}_{reranker}_{rewrite}.txt')

    # os.makedirs(os.path.dirname(output_path),exist_ok=True)

    queries = [x for x in srsly.read_json(query_path)]
    
    top100 = result_reader(result_path,topk)
    qrels_dict = qrel_reader(qrel_path)

    queries = [q for q in queries if q['query_id'] in qrels_dict]
    
    if args.debug:
        queries = queries[:1]
    
    logger.info('corpus loading start!!!')
    corpus = []
    for f in os.listdir(corpus_dir):
        if f.endswith('.jsonl'):
            corpus_path = os.path.join(corpus_dir, f)
            corpus.extend(srsly.read_jsonl(corpus_path))
            
    logger.info('corpus loaded... docid mapping start!!!')

    corpus_map = {x['id']: f"{x['title']} {x['contents']}" for x in tqdm(corpus,total=len(corpus))}
    
    scores = {}
    
    ranker = Reranker(model_name, device='cuda', batch_size=128, verbose=0)
    logger.info("reranking start...")
    for q in tqdm(queries):
        doc_ids = top100[q['query_id']]
        docs = [corpus_map[x] for x in doc_ids]
        scores[q['query_id']] = ranker.rank(q['query'], docs, doc_ids=doc_ids)
        
    write_result(output_path,scores,reranker)

if __name__=="__main__":
    main()