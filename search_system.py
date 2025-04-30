import os

from pathlib import Path
from yandex_cloud_ml_sdk.search_indexes import (
    StaticIndexChunkingStrategy,
    HybridSearchIndexType,
    ReciprocalRankFusionIndexCombinationStrategy,
)


def upload_data(sdk, folder_path: str):
    uploaded = []
    for file in Path(folder_path).glob('**/*'):
        if os.path.isdir(file): continue
        uploaded.append(sdk.files.upload(file.resolve(), ttl_days=1, expiration_policy="static"))
    return uploaded
    

def make_search_tool(sdk, uploaded_files: list):
    op = sdk.search_indexes.create_deferred(
        uploaded_files,
        index_type=HybridSearchIndexType(
            chunking_strategy=StaticIndexChunkingStrategy(
                max_chunk_size_tokens=1000, chunk_overlap_tokens=100
            ),
            combination_strategy=ReciprocalRankFusionIndexCombinationStrategy()
        ),
    )
    index = op.wait()
    return sdk.tools.search_index(index)

