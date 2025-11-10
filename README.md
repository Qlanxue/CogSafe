# CogSafe: A Safety Dynamic Evaluation Benchmark

Official repository for our ACL 2025 paper "[CogSafe: Dynamic Evaluation with Cognitive Reasoning for Multi-turn Safety of Large Language Models](https://aclanthology.org/2025.acl-long.963/)
".

We will release a training set soon!

## Evaluation from the file

```
python cogsafe_main.py --from_file ../data/eval_prompt.jsonl
```

## Evaluation from the target category

```
python cogsafe_main.py --from_file '' --from_category 'Representation & Toxicity - Toxic Content-Harass, Threaten, or Bully An Individual'
```

## Citation

If you use our technique or are inspired by our work, welcome to cite our paper and provide valuable suggestions.

```
@inproceedings{zhang-etal-2025-dynamic-evaluation,
    title = "Dynamic Evaluation with Cognitive Reasoning for Multi-turn Safety of Large Language Models",
    author = "Lanxue Zhang, Yanan Cao, Yuqiang Xie, Fang Fang  and Yangxi Li",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.963/",
    doi = "10.18653/v1/2025.acl-long.963",
    pages = "19588--19608",
}
```
