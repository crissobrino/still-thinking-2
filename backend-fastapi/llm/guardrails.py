def should_refuse(retrieved_docs, scores=None, min_docs=1, min_score=0.35) -> bool:
    if not retrieved_docs or len(retrieved_docs) < min_docs:
        return True

    if scores is not None and len(scores) > 0 and max(scores) < min_score:
        return True

    return False