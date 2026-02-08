def reciprocal_rank_fusion(results_list, top_k=50, k=60):
    """Reciprocal Rank Fusion"""
    scores = {}
    for result in results_list:
        text = result["text"]
        if text not in scores:
            scores[text] = {"score": 0, "source": result["source"], "text": text}
    
    # 为每个结果分配排名（按输入顺序视为排名）
    from collections import defaultdict
    source_groups = defaultdict(list)
    for r in results_list:
        source_groups[r["source"]].append(r)
    
    for source, group in source_groups.items():
        for rank, item in enumerate(group):
            text = item["text"]
            scores[text]["score"] += 1 / (rank + k)
    
    sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return sorted_items[:top_k]