from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.io import loadmat


@dataclass
class StoryData:
    story_id: str
    text: str
    tokens: list[str]
    normalized_tokens: list[str]
    period_block_ids: list[int]


@dataclass
class SourceSegment:
    story_id: str
    participant_id: str
    segment_index: int
    source_text: str
    scanpath_tokens: list[str]
    normalized_scanpath_tokens: list[str]


PUNCT_TRANSLATION = str.maketrans(
    {
        '—': '',
        '‒': '',
        '−': '',
        '-': '',
        '«': '',
        '»': '',
        '“': '',
        '”': '',
        '‘': '',
        '’': '',
        '"': '',
        "'": '',
        '(': '',
        ')': '',
        ';': '',
        ',': '',
        ':': '',
        '.': '',
        '…': '',
        '¿': '',
        '?': '',
        '¡': '',
        '!': '',
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a text/scanpath-aligned dataset from results_all scanpaths and stimuli.'
    )
    parser.add_argument('--results_dir', default='results_all', help='Directory produced by em_analysis.py')
    parser.add_argument('--stimuli_dir', default='stimuli', help='Directory containing story .mat files')
    parser.add_argument('--output_dir', default='aligned_output', help='Where aligned outputs will be written')
    parser.add_argument(
        '--mirrored_output_dir',
        default='results_all_alligned',
        help='Folder mirroring results_all/scanpaths with one jsonl file per story/participant.',
    )
    parser.add_argument(
        '--full_scanpaths_dir',
        default='data/processed/scanpaths',
        help='Fixation-level scanpaths with global word ids. Used for exact word_id recovery.',
    )
    return parser.parse_args()


def resolve_case_insensitive_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if path.exists():
        return path

    parent = path.parent if path.parent.exists() else None
    if parent is not None:
        for candidate in parent.iterdir():
            if candidate.name.lower() == path.name.lower():
                return candidate

    if not path.is_absolute() and path.parent == base_dir:
        for candidate in base_dir.iterdir():
            if candidate.name.lower() == path.name.lower():
                return candidate

    return path


def normalize_token(token: str) -> str:
    normalized = unicodedata.normalize('NFKD', token).lower()
    normalized = normalized.translate(PUNCT_TRANSLATION)
    return ''.join(ch for ch in normalized if ch.isalnum())


def lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    prev = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        curr = [0]
        for j, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def match_quality(source_tokens: list[str], aligned_tokens: list[str]) -> tuple[str, int, float]:
    source_norm = [normalize_token(token) for token in source_tokens]
    aligned_norm = [normalize_token(token) for token in aligned_tokens]
    matched = lcs_length(source_norm, aligned_norm)
    coverage = matched / len(source_norm) if source_norm else 1.0
    if coverage >= 0.999:
        quality = 'high'
    elif coverage >= 0.9:
        quality = 'medium'
    elif coverage >= 0.5:
        quality = 'low'
    else:
        quality = 'failed'
    return quality, matched, coverage


def load_story_data(stimulus_file: Path) -> StoryData:
    mat = loadmat(str(stimulus_file), simplify_cells=True)
    raw_lines = mat['lines']
    sorted_lines = sorted(raw_lines, key=lambda line: (int(line['screen']), int(line['linenumber'])))

    tokens: list[str] = []
    normalized_tokens: list[str] = []
    period_block_ids: list[int] = []
    block_id = 0
    for line in sorted_lines:
        for token in line['text'].split():
            tokens.append(token)
            normalized_tokens.append(normalize_token(token))
            period_block_ids.append(block_id)
            if '.' in token:
                block_id += 1

    return StoryData(
        story_id=stimulus_file.stem,
        text=' '.join(tokens),
        tokens=tokens,
        normalized_tokens=normalized_tokens,
        period_block_ids=period_block_ids,
    )


def load_stories(stimuli_dir: Path) -> dict[str, StoryData]:
    stories: dict[str, StoryData] = {}
    for stimulus_file in sorted(stimuli_dir.glob('*.mat')):
        if stimulus_file.stem == 'Test':
            continue
        stories[stimulus_file.stem] = load_story_data(stimulus_file)
    return stories


def load_results_segments(results_scanpaths_dir: Path) -> dict[tuple[str, str], list[SourceSegment]]:
    segments_by_story_participant: dict[tuple[str, str], list[SourceSegment]] = {}
    for story_dir in sorted(path for path in results_scanpaths_dir.iterdir() if path.is_dir()):
        for scanpath_file in sorted(story_dir.glob('*.json')):
            segments: list[SourceSegment] = []
            for segment_index, raw_line in enumerate(scanpath_file.read_text(encoding='utf-8').splitlines()):
                payload = json.loads(raw_line)
                scanpath_tokens = payload['text'].split()
                segments.append(
                    SourceSegment(
                        story_id=story_dir.name,
                        participant_id=scanpath_file.stem,
                        segment_index=segment_index,
                        source_text=payload['text'],
                        scanpath_tokens=scanpath_tokens,
                        normalized_scanpath_tokens=[normalize_token(token) for token in scanpath_tokens],
                    )
                )
            segments_by_story_participant[(story_dir.name, scanpath_file.stem)] = segments
    return segments_by_story_participant


def load_full_scanpath(full_scanpath_file: Path) -> dict[str, Any]:
    return json.loads(full_scanpath_file.read_text(encoding='utf-8'))


def slice_scanpath_by_segments(
    story: StoryData,
    source_segments: list[SourceSegment],
    full_scanpath: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    global_word_ids = full_scanpath['word_ids']
    full_words = full_scanpath['words']

    expected_length = sum(len(segment.scanpath_tokens) for segment in source_segments)
    if expected_length != len(global_word_ids):
        return [], {
            'reason': 'token_count_mismatch',
            'expected_scanpath_tokens': expected_length,
            'full_scanpath_tokens': len(global_word_ids),
        }

    examples: list[dict[str, Any]] = []
    offset = 0
    for segment in source_segments:
        seg_len = len(segment.scanpath_tokens)
        seg_global_word_ids = global_word_ids[offset:offset + seg_len]
        seg_full_words = full_words[offset:offset + seg_len]
        offset += seg_len

        if not seg_global_word_ids:
            continue

        invalid_word_ids = [word_id for word_id in seg_global_word_ids if word_id < 0 or word_id >= len(story.tokens)]
        if invalid_word_ids:
            return [], {
                'reason': 'global_word_id_out_of_range',
                'segment_index': segment.segment_index,
                'invalid_word_ids': invalid_word_ids[:10],
            }

        fragment_start = min(seg_global_word_ids)
        fragment_end = max(seg_global_word_ids)
        text_tokens = story.tokens[fragment_start:fragment_end + 1]
        word_id = [global_word_id - fragment_start + 1 for global_word_id in seg_global_word_ids]
        aligned_scanpath_tokens = [story.tokens[global_word_id] for global_word_id in seg_global_word_ids]

        quality, matched, coverage = match_quality(segment.scanpath_tokens, aligned_scanpath_tokens)
        block_ids = [story.period_block_ids[global_word_id] for global_word_id in seg_global_word_ids]
        source_full_quality, _, _ = match_quality(segment.scanpath_tokens, seg_full_words)

        example = {
            'story_id': segment.story_id,
            'participant_id': segment.participant_id,
            'trial_id': f"{segment.story_id}::{segment.participant_id}::seg_{segment.segment_index:04d}",
            'segment_index': segment.segment_index,
            'text': ' '.join(text_tokens),
            'word_id': word_id,
            'scanpath_tokens': segment.scanpath_tokens,
            'aligned_scanpath_tokens': aligned_scanpath_tokens,
            'text_tokens': text_tokens,
            'match_quality': quality,
            'coverage': round(coverage, 6),
            'n_scanpath_tokens': len(segment.scanpath_tokens),
            'n_aligned_tokens': matched,
            'period_block_start': min(block_ids),
            'period_block_end': max(block_ids),
            'global_word_start': fragment_start,
            'global_word_end': fragment_end,
            'source_text': segment.source_text,
            'source_vs_full_quality': source_full_quality,
        }
        examples.append(example)

    return examples, None


def export_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write('\n')


def export_results_like_structure(mirrored_output_dir: Path, examples: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault((example['story_id'], example['participant_id']), []).append(example)

    for (story_id, participant_id), participant_examples in grouped.items():
        story_dir = mirrored_output_dir / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for example in sorted(participant_examples, key=lambda item: item['segment_index']):
            rows.append(
                {
                    'scanpath_text': example['source_text'],
                    'text': example['text'],
                    'word_id': example['word_id'],
                    'scanpath_tokens': example['scanpath_tokens'],
                    'text_tokens': example['text_tokens'],
                    'trial_id': example['trial_id'],
                    'segment_index': example['segment_index'],
                    'match_quality': example['match_quality'],
                    'coverage': example['coverage'],
                }
            )
        export_jsonl(story_dir / f'{participant_id}.json', rows)


def build_dataset(
    results_scanpaths_dir: Path,
    stimuli_dir: Path,
    output_dir: Path,
    mirrored_output_dir: Path,
    full_scanpaths_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stories = load_stories(stimuli_dir)
    source_segments = load_results_segments(results_scanpaths_dir)

    examples: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for (story_id, participant_id), participant_segments in sorted(source_segments.items()):
        if story_id not in stories:
            issues.append(
                {
                    'story_id': story_id,
                    'participant_id': participant_id,
                    'reason': 'missing_stimulus_story',
                }
            )
            continue

        full_scanpath_file = full_scanpaths_dir / story_id / f'{participant_id}.json'
        if not full_scanpath_file.exists():
            issues.append(
                {
                    'story_id': story_id,
                    'participant_id': participant_id,
                    'reason': 'missing_full_scanpath_file',
                    'expected_file': str(full_scanpath_file),
                }
            )
            continue

        full_scanpath = load_full_scanpath(full_scanpath_file)
        participant_examples, participant_error = slice_scanpath_by_segments(
            story=stories[story_id],
            source_segments=participant_segments,
            full_scanpath=full_scanpath,
        )
        if participant_error is not None:
            participant_error['story_id'] = story_id
            participant_error['participant_id'] = participant_id
            issues.append(participant_error)
            continue

        examples.extend(participant_examples)

    quality_counts = Counter(example['match_quality'] for example in examples)
    summary = {
        'stories': len({example['story_id'] for example in examples}),
        'participants': len({(example['story_id'], example['participant_id']) for example in examples}),
        'examples': len(examples),
        'quality_counts': dict(sorted(quality_counts.items())),
        'issues': len(issues),
        'results_scanpaths_dir': str(results_scanpaths_dir),
        'stimuli_dir': str(stimuli_dir),
        'full_scanpaths_dir': str(full_scanpaths_dir),
    }

    non_high_examples = [
        {
            'story_id': example['story_id'],
            'participant_id': example['participant_id'],
            'trial_id': example['trial_id'],
            'match_quality': example['match_quality'],
            'coverage': example['coverage'],
            'text': example['text'],
            'scanpath_tokens': example['scanpath_tokens'],
            'aligned_scanpath_tokens': example['aligned_scanpath_tokens'],
            'word_id': example['word_id'],
        }
        for example in examples
        if example['match_quality'] != 'high'
    ]
    issues.extend(non_high_examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    mirrored_output_dir.mkdir(parents=True, exist_ok=True)
    export_jsonl(output_dir / 'aligned_scanpaths.jsonl', examples)
    export_jsonl(output_dir / 'alignment_issues.jsonl', issues)
    export_results_like_structure(mirrored_output_dir, examples)
    (output_dir / 'alignment_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    return examples, issues, summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    results_dir = resolve_case_insensitive_path(args.results_dir, repo_root)
    stimuli_dir = resolve_case_insensitive_path(args.stimuli_dir, repo_root)
    output_dir = resolve_case_insensitive_path(args.output_dir, repo_root)
    mirrored_output_dir = resolve_case_insensitive_path(args.mirrored_output_dir, repo_root)
    full_scanpaths_dir = resolve_case_insensitive_path(args.full_scanpaths_dir, repo_root)

    results_scanpaths_dir = results_dir / 'scanpaths' if (results_dir / 'scanpaths').is_dir() else results_dir
    if not results_scanpaths_dir.is_dir():
        raise SystemExit(f'Results scanpaths directory not found: {results_scanpaths_dir}')
    if not stimuli_dir.is_dir():
        raise SystemExit(f'Stimuli directory not found: {stimuli_dir}')
    if not full_scanpaths_dir.is_dir():
        raise SystemExit(f'Full scanpaths directory not found: {full_scanpaths_dir}')

    examples, issues, summary = build_dataset(
        results_scanpaths_dir=results_scanpaths_dir,
        stimuli_dir=stimuli_dir,
        output_dir=output_dir,
        mirrored_output_dir=mirrored_output_dir,
        full_scanpaths_dir=full_scanpaths_dir,
    )

    print(f"Wrote {len(examples)} aligned examples to {output_dir / 'aligned_scanpaths.jsonl'}")
    print(f"Wrote {len(issues)} issue records to {output_dir / 'alignment_issues.jsonl'}")
    print(f"Wrote mirrored per-subject files to {mirrored_output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    preview = examples[:3]
    for index, example in enumerate(preview):
        print(f"PREVIEW {index + 1}: {example['trial_id']}")
        print(f"  text: {example['text']}")
        print(f"  scanpath_tokens: {example['scanpath_tokens'][:20]}")
        print(f"  word_id: {example['word_id'][:20]}")
        print(f"  match_quality: {example['match_quality']} coverage={example['coverage']}")


if __name__ == '__main__':
    main()
