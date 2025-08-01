from multiprocessing.pool import Pool
import os
from typing import BinaryIO, Iterable, Iterator
from collections import Counter, defaultdict
import regex as re
from itertools import pairwise, islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import heapq
import copy
import json


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(input_path: str, start: int, end: int, token_split_pattern: re.Pattern, special_tokens: bytes) -> defaultdict[tuple, int]:
    local_token_counts = defaultdict(int)
    
    with open(input_path, "rb") as f:
        f.seek(start) # start from start_idx
        chunk_text = f.read(end - start).decode("utf-8", errors='replace')
        text_parts = [chunk_text]

        # Split the chunk into subchunks by special tokens
        for token in special_tokens:
            new_parts = []
            for part in text_parts:
                new_parts.extend(part.split(token))
            text_parts = new_parts

        # Count all occurrences of tokens in all subchunks
        for sub_chunk in text_parts:
            if not sub_chunk:
                continue
            tokens = token_split_pattern.finditer(sub_chunk)
            for match in tokens:
                token = match.group().encode('utf-8')
                token_tuple = tuple(token)
                local_token_counts[token_tuple] += 1
            
    return local_token_counts

class PreTokenizer:
    def __init__(self, special_tokens: list[str], num_processes: int, num_chunks: int):
        self.num_processes = num_processes
        self.num_chunks = num_chunks
        self.special_tokens = None
        if special_tokens:
            self.special_tokens = {token: token.encode('utf-8') for token in special_tokens}
        if self.special_tokens:
            sorted_dec_special_tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)
            special_pat_str = '|'.join(re.escape(token) for token in sorted_dec_special_tokens)
            self.special_pat = re.compile(f"({special_pat_str})")
        else:
            self.special_pat = None
        self.pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+") # the one used in GPT-2
        self.sent_split_pat = b"<|endoftext|>"
    
    def pretokenize(self, text_chunk: str) -> list[bytes]:
        if self.special_pat:
            parts = self.special_pat.split(text_chunk)
        else:
            parts = [text_chunk]
        pretokenized_tokens = []
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                pretokenized_tokens.append(self.special_tokens[part])
            elif part:
                parsed_text = [match.group(0).encode('utf-8') for match in re.finditer(self.pat, part)]
                pretokenized_tokens.extend(parsed_text)
        return pretokenized_tokens
   
    def process(self, input_path:str) -> list[bytes]:
        with open(input_path, "rb") as f:
           chunk_bounds = find_chunk_boundaries(f, self.num_chunks, self.sent_split_pat)
           all_chunks = []
           for start_pos, end_pos in zip(chunk_bounds[:-1], chunk_bounds[1:]):
               f.seek(start_pos)
               chunk = f.read(end_pos - start_pos).decode("utf-8", errors="ignore")
               all_chunks.append(chunk)
        pretokenized_tokens = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            for chunk in all_chunks:
                futures.append(executor.submit(self.pretokenize, chunk))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                pretokenized_tokens.extend(future.result())
        return pretokenized_tokens

class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str], num_processes: int = 1, num_chunks: int = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes
        self.num_chunks = num_chunks if num_chunks is not None else num_processes
        self.pre_tokenizer = PreTokenizer(special_tokens, num_processes, self.num_chunks)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for token in self.special_tokens:
            encoded_token = token.encode('utf-8')
            if encoded_token not in self.vocab.values():
                 self.vocab[len(self.vocab)] = encoded_token

    def _find_best_pair(self, freq_max_heap: list[tuple[int, tuple[bytes, bytes]]], pair_freqs: defaultdict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        while freq_max_heap:
            neg_freq, pair = heapq.heappop(freq_max_heap)
            freq = -neg_freq

            # skip outdated node
            if pair not in pair_freqs or pair_freqs[pair] != freq:
                continue
            best_pair = pair
            top_cands = []

            # tie-breaking if the frequency is the same. if tie, then pick the one which is lexicographically larger
            while freq_max_heap and freq_max_heap[0][0] == neg_freq:
                _, next_pair = heapq.heappop(freq_max_heap)
                # valid entry
                if next_pair in pair_freqs and pair_freqs[next_pair] == freq:
                    if next_pair > best_pair:
                        top_cands.append(best_pair)
                        best_pair = next_pair
                    else:
                        top_cands.append(next_pair)
            
            # readd unselected top candidates back to heap
            for cand in top_cands:
                heapq.heappush(freq_max_heap, (neg_freq, cand))
            return best_pair
        return None


    def _update_pair_freq(self, pair: tuple[bytes, bytes], freq_delta: int, pair_freqs: dict, pair_to_tokens: dict, token: bytes, freq_max_heap: list):
        pair_freqs[pair] += freq_delta
        
        if pair_freqs[pair] > 0:
            pair_to_tokens[pair].add(token)
            heapq.heappush(freq_max_heap, (-pair_freqs[pair], pair))
        else:
            del pair_freqs[pair]
            if pair in pair_to_tokens:
                pair_to_tokens[pair].discard(token)
                if not pair_to_tokens[pair]:
                    del pair_to_tokens[pair]


    def _merge_best_pair_and_update_data_structures(self, best_pair: tuple[bytes, bytes], new_token: bytes, pair_freqs: defaultdict[tuple[bytes, bytes], int],
                                                    pair_to_tokens: defaultdict[bytes, set[bytes]], token_to_chars: dict[bytes, list[bytes]], 
                                                    non_special_tokens_freq: dict[bytes, int], freq_max_heap: list):
        for token in pair_to_tokens.get(best_pair, []):
            chars = token_to_chars[token]
            token_freq = non_special_tokens_freq[token]
            curr_idx = 0
            while curr_idx < len(chars) - 1:
                if chars[curr_idx] == best_pair[0] and chars[curr_idx + 1] == best_pair[1]:
                    # Merge the best byte pair into one
                    chars[curr_idx] = new_token
                    chars.pop(curr_idx + 1)

                    # Update frequencies of prev adjacent pair (old)
                    if curr_idx > 0:
                        self._update_pair_freq((chars[curr_idx - 1], best_pair[0]), -token_freq, pair_freqs, pair_to_tokens, token, freq_max_heap)
                    # Update frequencies of next adjacent pair (old)
                    if curr_idx < len(chars) - 1:
                        self._update_pair_freq((best_pair[1], chars[curr_idx + 1]), -token_freq, pair_freqs, pair_to_tokens, token, freq_max_heap)

                    # Update new adjacent prev pair (old)
                    if curr_idx > 0:
                        self._update_pair_freq((chars[curr_idx - 1], chars[curr_idx]), token_freq, pair_freqs, pair_to_tokens, token, freq_max_heap)
                    if curr_idx < len(chars) - 1:
                        self._update_pair_freq((chars[curr_idx], chars[curr_idx + 1]), token_freq, pair_freqs, pair_to_tokens, token, freq_max_heap)
                else:
                    curr_idx += 1

        if best_pair in pair_freqs:
            del pair_freqs[best_pair]
        if best_pair in pair_to_tokens:
            del pair_to_tokens[best_pair]
             

    def train(self, input_path:str):

        # step 1: pretokenize the text
        pretokenized_tokens = self.pre_tokenizer.process(input_path)
        special_tokens_bytes = set(self.pre_tokenizer.special_tokens.values())
        non_special_tokens_freq = Counter(token for token in pretokenized_tokens if token not in special_tokens_bytes)
        
        # step 2: count pairs of characters in non-special tokens
        token_to_chars = {token:[bytes([ch]) for ch in token] for token in non_special_tokens_freq}
        pair_freqs = defaultdict(int)
        pair_to_tokens = defaultdict(set)
        freq_max_heap = []
        for token, freq in non_special_tokens_freq.items():
            chars = token_to_chars[token]
            if len(chars) > 1:
                for p1, p2 in zip(chars[:-1], chars[1:]):
                    pair = (p1, p2)
                    pair_freqs[pair] += freq
                    pair_to_tokens[pair].add(token)
        
        for pair, freq in pair_freqs.items():
            heapq.heappush(freq_max_heap, (-freq, pair))
        
        merges = []
        num_merges_required = self.vocab_size - len(self.vocab)

        # step 3: subword merging operations
        while len(merges) < num_merges_required:
            best_pair = self._find_best_pair(freq_max_heap, pair_freqs)
            if best_pair is None:
                break

            # merge best pair into a new token
            merges.append(best_pair)
            merged_bytes = best_pair[0] + best_pair[1]
            self.vocab[len(self.vocab)] = merged_bytes
            
        
            self._merge_best_pair_and_update_data_structures(
                best_pair, merged_bytes, pair_freqs, pair_to_tokens, token_to_chars
                , non_special_tokens_freq, freq_max_heap
            )
        
        self.merges = merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.id_to_token = {k: v for k, v in vocab.items()}
        self.special_tokens = {}
        self.special_pat = None
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            for token_str in sorted_tokens:
                self.special_tokens[token_str] = self.token_to_id[token_str.encode("utf-8")]
            self.special_pat = "(" + "|".join(re.escape(k) for k in sorted_tokens) + ")"
        self.pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        self.pretokenizer = PreTokenizer(special_tokens, num_processes=1, num_chunks=None)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]) -> 'Tokenizer':
        with open(vocab_filepath, 'r') as f:
            vocab_str = json.load(f)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_str.items()}
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_str = [line.strip() for line in f]
            merges = [tuple(part.encode('utf-8') for part in merge.split(' ')) for merge in merges_str]
        return cls(vocab, merges, special_tokens)
    
    def _encode_tokens(self, text) -> list[int]:
        pretokenized_tokens = self.pretokenizer.pretokenize(text)
        #pretokenized_tokens = [s.encode('utf-8') for s in self.pat.findall(text_bytes.decode('utf-8', errors='replace'))]
        token_ids = []
        for token in pretokenized_tokens:
            if not token:
                continue
            parts = [bytes([ch]) for ch in token]
            while len(parts) > 1:
                best_pair_info = min(
                    ((self.merge_ranks.get((parts[i], parts[i + 1]), float('inf')), (parts[i], parts[i + 1]))
                     for i in range(len(parts) - 1)),
                    key=lambda x: x[0]
                )
                if best_pair_info[0] == float('inf'):
                    break
                best_pair = best_pair_info[1]
                new_parts = []
                i = 0
                while i < len(parts):
                    if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                        new_parts.append(parts[i] + parts[i+1])
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts
            for part in parts:
                token_ids.append(self.token_to_id[part])
        return token_ids

    def encode(self, text: str) -> list[int]:
        if not self.special_pat:
            token_ids = self._encode_tokens(text)
        else:
            chunks = re.split(self.special_pat, text)
            token_ids = []
            for chunk in chunks:
                if not chunk:
                    continue
                if chunk in self.special_tokens:
                    token_ids.append(self.special_tokens[chunk])
                else:
                    token_ids.extend(self._encode_tokens(chunk))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            encoded_ids = self.encode(text_chunk)
            for token_id in encoded_ids:
                yield token_id

    def decode(self, token_ids: list[int]) -> str:
        all_bytes = b"".join(self.vocab.get(token_id, b'') for token_id in token_ids)
        text = all_bytes.decode('utf-8', errors='replace')
        return text

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    trainer = BPETrainer(vocab_size, special_tokens, num_processes=1, num_chunks=None)
    trainer.train(input_path)
    return trainer.vocab, trainer.merges