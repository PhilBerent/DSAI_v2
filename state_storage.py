import json
import os
import zstandard as zstd
from typing import Any, Dict
from UtilityFunctions import *
from globals import *
from DSAIParams import *
from enums_and_constants import StateStoragePoints, CodeStages

def _compress_json_to_file(data: Dict[str, Any], file_path: str, level: int = 10):
    os.makedirs(StateStorageDirectory, exist_ok=True)
    try:
        json_bytes = json.dumps(data, indent=4, ensure_ascii=True).encode('utf-8')
        with open(file_path, 'wb') as f_out:
            cctx = zstd.ZstdCompressor(level=level)
            f_out.write(cctx.compress(json_bytes))
        print(f"State successfully saved with Zstandard compression to {file_path}")
    except Exception as e:
        print(f"Error saving compressed state to {file_path}: {e}")

def _decompress_json_from_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'rb') as f_in:
        dctx = zstd.ZstdDecompressor()
        json_bytes = dctx.decompress(f_in.read())
    return json.loads(json_bytes.decode('utf-8'))

def save_state(data_to_save: Dict[str, Any], file_path: str):
    _compress_json_to_file(data_to_save, file_path)

def loadStateLBA():
    loaded_state = _decompress_json_from_file(LargeBlockAnalysisCompletedFile)
    large_blocks = loaded_state.get("large_blocks")
    map_results = loaded_state.get("map_results")
    final_entities = loaded_state.get("final_entities")
    raw_text = loaded_state.get("raw_text")
    return large_blocks, map_results, final_entities, raw_text

def loadStateIA():
    loaded_state = _decompress_json_from_file(IterativeAnalysisCompletedFile)
    doc_analysis_result = loaded_state.get("doc_analysis_result")
    large_blocks = loaded_state.get("large_blocks")
    map_results = loaded_state.get("map_results") 
    raw_text = loaded_state.get("raw_text")
    return doc_analysis_result, large_blocks, map_results, raw_text

def loadStateDBA():
    loaded_state = _decompress_json_from_file(DetailedBlockAnalysisCompletedFile)
    file_id = loaded_state.get("file_id")
    chunks_with_analysis = loaded_state.get("chunks_with_analysis")
    map_results = loaded_state.get("map_results") 
    doc_analysis_result = loaded_state.get("doc_analysis_result")
    return file_id, chunks_with_analysis, doc_analysis_result, map_results

def loadStateEA():
    loaded_state = _decompress_json_from_file(EmbeddingsCompletedFile)
    file_id = loaded_state.get("file_id")   
    embeddings_dict = loaded_state.get("embeddings_dict")
    chunks_with_analysis = loaded_state.get("chunks_with_analysis")
    doc_analysis_result = loaded_state.get("doc_analysis_result")
    map_results = loaded_state.get("map_results")
    return file_id, embeddings_dict, chunks_with_analysis, doc_analysis_result, map_results

def loadStateGA():
    loaded_state = _decompress_json_from_file(GraphAnalysisCompletedFile)
    loaded_state = json.loads(json_bytes.decode('utf-8'))
    file_id = loaded_state.get("file_id")
    graph_nodes = loaded_state.get("graph_nodes")
    graph_edges = loaded_state.get("graph_edges")
    embeddings_dict = loaded_state.get("embeddings_dict")
    chunks_with_analysis = loaded_state.get("chunks_with_analysis")
    doc_analysis_result = loaded_state.get("doc_analysis_result")
    map_results = loaded_state.get("map_results")
    return file_id, graph_nodes, graph_edges, embeddings_dict, chunks_with_analysis, doc_analysis_result, map_results
