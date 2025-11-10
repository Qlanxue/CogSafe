import argparse
import threading
from generator import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # evaluation from category
    parser.add_argument('--from_category', type=str, default='Representation & Toxicity - Toxic Content-Harass, Threaten, or Bully An Individual', help='the safety category to use')
    parser.add_argument('--eval_num', type=int, default=40, help='the evaluation number for current category')
    parser.add_argument('--generate_system', type=bool, default=False, help='whether to generate system for current category')

    # evaluation from file
    parser.add_argument('--from_file', type=str, default='../data/eval_prompt.jsonl')
    parser.add_argument('--batch_size', type=int, default=40, help='the batch size for evaluation')

    # save file path of results
    parser.add_argument('--result_file', type=str, default='gpt-3.5-turbo_results.jsonl', help='the file used to save the results')

    parser.add_argument('--strategy_file', type=str, default='../data/strategy_collection_emb.jsonl', help=' the preprocessed embedding of the strategy description')
    parser.add_argument('--retrieval_model', type=str, default='retrieval_strategy_epoch', help='the retrieval model')
    parser.add_argument('--max_tree_depth', type=int, default=3, help='the maximum depth of the tree')
    parser.add_argument('--selected_ratio', type=float, default=0.8, help='the ratio of selected items')
    parser.add_argument('--max_round', type=int, default=5, help='the maximum number of dialogue round')
    parser.add_argument('--omni_model', type=str, default='gpt-4o', help='the model for Omini')
    parser.add_argument('--omni_model_path', type=str, default='')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo', help='the model needed to evaluated')
    parser.add_argument('--llm_model_path', type=str, default='')

    args = parser.parse_args()

    # test once
    # omni_agent = OmniAgent(args)
    # omni_agent.evaluation_process()

    if args.from_file != '':
        # begin evaluation from file
        batch_cate = []
        batch_agent = []
        line_cnt = 0
        with jsonlines.open(args.from_file, 'r') as fr:
            for line in fr:
                batch_cate.append(copy.copy(line))
                batch_agent.append(OmniAgent(args))
                if len(batch_cate) == args.batch_size:
                    threads = []
                    for k in range(len(batch_cate)):
                        t = threading.Thread(target=batch_agent[k].evaluation_process, args=(batch_cate[k],))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    batch_cate = []
                    batch_agent = []

        if len(batch_cate):
            threads = []
            for k in range(len(batch_cate)):
                t = threading.Thread(target=batch_agent[k].evaluation_process, args=(batch_cate[k],))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
    else:
        # begin evaluation from category
        batch_agent = []
        line_cnt = 0
        for i in range(args.eval_num):
            batch_agent.append(OmniAgent(args))
            if len(batch_agent) == args.batch_size:
                threads = []
                for k in range(len(batch_agent)):
                    t = threading.Thread(target=batch_agent[k].evaluation_process, args=(None,))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
                batch_agent = []

        if len(batch_agent):
            threads = []
            for k in range(len(batch_agent)):
                t = threading.Thread(target=batch_agent[k].evaluation_process, args=(None, ))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()



