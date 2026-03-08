from .base import BaseReranker
from .listwise import ListwiseReranker

__all__ = ['BaseReranker', 'ListwiseReranker', 'create_reranker']


def create_reranker(args):
    """
    Factory function to create a reranker from parsed arguments.

    Args:
        args: Parsed arguments containing reranker configuration

    Returns:
        An instance of a reranker subclass, or None if no reranker is specified
    """
    if not hasattr(args, 'reranker_type') or args.reranker_type is None:
        return None

    if args.reranker_type == 'listwise':
        return ListwiseReranker(args)
    else:
        raise ValueError(f"Unknown reranker type: {args.reranker_type}")
