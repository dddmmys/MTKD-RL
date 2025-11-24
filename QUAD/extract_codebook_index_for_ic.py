#!/usr/bin/env python3

import torch
import os
from example import get_parsers
from vq_utils_for_ic import CodebookIndexExtractor

def get_world_size():
    warn_message = (
        "It's better to use GPU to extrac codebook indices"
        "Please set with commonds like: export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    assert (torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" in os.environ), warn_message
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert world_size > 0, warn_message
    return world_size

def main():
    world_size = get_world_size()
    parser = get_parsers()
    CodebookIndexExtractor.add_arguments(parser)
    args = parser.parse_args()
    args.device = torch.device("cuda", 0)
    args.world_size = world_size
    extractor = CodebookIndexExtractor(params=args)
    if not args.use_extracted_codebook:
        extractor.extract_and_save_embedding()
        extractor.train_quantizer()
        extractor.extract_codebook_indexes()


if __name__ == "__main__":
    main()
