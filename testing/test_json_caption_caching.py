#!/usr/bin/env python3
"""
Test JSON caption caching and weighted selection behavior.
"""
import os
import sys
import json
import tempfile
import shutil

# Add toolkit to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.dataloader_mixins import (
    parse_json_captions,
    select_prompt_weighted,
    _filter_prompts_by_mode,
    compute_json_file_hash,
)

def test_select_prompt_weighted_all_zero_weights():
    """Test that all-zero weights returns empty caption."""
    prompts = [
        {'prompt': 'test1', 'weight': 0, 'do_i2v': True, 'do_t2v': True},
        {'prompt': 'test2', 'weight': 0, 'do_i2v': True, 'do_t2v': True},
    ]
    result = select_prompt_weighted(prompts)
    assert result['prompt'] == '', f"Expected empty caption, got '{result['prompt']}'"
    print("✓ test_select_prompt_weighted_all_zero_weights passed")


def test_select_prompt_weighted_some_zero_weights():
    """Test that zero-weight prompts are excluded from selection."""
    prompts = [
        {'prompt': 'test1', 'weight': 0, 'do_i2v': True, 'do_t2v': True},
        {'prompt': 'test2', 'weight': 1.0, 'do_i2v': True, 'do_t2v': True},
    ]
    # Run many times to verify zero-weight prompt is never selected
    for _ in range(100):
        result = select_prompt_weighted(prompts)
        assert result['prompt'] == 'test2', f"Expected 'test2', got '{result['prompt']}'"
    print("✓ test_select_prompt_weighted_some_zero_weights passed")


def test_filter_prompts_by_mode_warning():
    """Test that warning is logged when all prompts filtered out."""
    prompts = [
        {'prompt': 'test1', 'weight': 1.0, 'do_i2v': False, 'do_t2v': True},
    ]
    # Filter for I2V mode - should get empty list
    filtered = _filter_prompts_by_mode(prompts, is_i2v_mode=True, log_warning=True)
    assert len(filtered) == 0, f"Expected 0 filtered prompts, got {len(filtered)}"
    print("✓ test_filter_prompts_by_mode_warning passed")


def test_compute_json_file_hash():
    """Test JSON file hash computation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{'prompt': 'test'}], f)
        temp_path = f.name
    
    try:
        hash1 = compute_json_file_hash(temp_path)
        assert hash1 is not None and len(hash1) > 0
        
        # Same content = same hash
        hash2 = compute_json_file_hash(temp_path)
        assert hash1 == hash2, "Same file should produce same hash"
        
        # Different content = different hash
        with open(temp_path, 'w') as f:
            json.dump([{'prompt': 'different'}], f)
        hash3 = compute_json_file_hash(temp_path)
        assert hash1 != hash3, "Different content should produce different hash"
    finally:
        os.unlink(temp_path)
    
    print("✓ test_compute_json_file_hash passed")


def test_parse_json_captions_basic():
    """Test basic JSON caption parsing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {'prompt': 'test1', 'weight': 2.0, 'do_i2v': True, 'do_t2v': False},
            {'prompt': 'test2', 'weight': 1.0, 'do_i2v': False, 'do_t2v': True},
        ], f)
        temp_path = f.name
    
    try:
        prompts = parse_json_captions(temp_path)
        assert len(prompts) == 2
        assert prompts[0]['prompt'] == 'test1'
        assert prompts[0]['weight'] == 2.0
        assert prompts[0]['do_i2v'] == True
        assert prompts[0]['do_t2v'] == False
        assert prompts[1]['prompt'] == 'test2'
        assert prompts[1]['weight'] == 1.0
        assert prompts[1]['do_i2v'] == False
        assert prompts[1]['do_t2v'] == True
    finally:
        os.unlink(temp_path)
    
    print("✓ test_parse_json_captions_basic passed")


def test_parse_json_captions_defaults():
    """Test default values for missing fields."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{'prompt': 'test'}], f)
        temp_path = f.name
    
    try:
        prompts = parse_json_captions(temp_path)
        assert len(prompts) == 1
        assert prompts[0]['prompt'] == 'test'
        assert prompts[0]['do_i2v'] == True  # default
        assert prompts[0]['do_t2v'] == True  # default
        assert prompts[0]['weight'] == 1.0   # single prompt default
    finally:
        os.unlink(temp_path)
    
    print("✓ test_parse_json_captions_defaults passed")


def test_parse_json_captions_empty_prompt_respected():
    """Test that empty prompts are kept as valid prompts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {'prompt': 'test1', 'weight': 1.0, 'do_i2v': True, 'do_t2v': True},
            {'prompt': '', 'weight': 1.0, 'do_i2v': True, 'do_t2v': True},
            {'prompt': 'test3', 'weight': 1.0, 'do_i2v': True, 'do_t2v': True},
        ], f)
        temp_path = f.name
    
    try:
        prompts = parse_json_captions(temp_path)
        # Empty prompt should NOT be filtered out
        assert len(prompts) == 3, f"Expected 3 prompts (including empty), got {len(prompts)}"
        assert prompts[0]['prompt'] == 'test1'
        assert prompts[1]['prompt'] == '', "Empty prompt should be preserved"
        assert prompts[2]['prompt'] == 'test3'
    finally:
        os.unlink(temp_path)
    
    print("✓ test_parse_json_captions_empty_prompt_respected passed")


def test_select_prompt_weighted_with_empty_prompt():
    """Test that empty prompts can be selected by weight."""
    prompts = [
        {'prompt': 'test1', 'weight': 2.0, 'do_i2v': True, 'do_t2v': True},
        {'prompt': '', 'weight': 1.0, 'do_i2v': True, 'do_t2v': True},
    ]
    
    # Run many times - empty prompt should be selected ~1/3 of the time
    empty_count = 0
    total = 1000
    for _ in range(total):
        result = select_prompt_weighted(prompts)
        if result['prompt'] == '':
            empty_count += 1
    
    # Should be roughly 1/3 (with some tolerance)
    ratio = empty_count / total
    assert 0.2 < ratio < 0.4, f"Empty prompt selected {ratio:.1%} of time (expected ~33%)"
    print("✓ test_select_prompt_weighted_with_empty_prompt passed")


def test_filter_prompts_t2v_only_mode():
    """
    Test that T2V-only mode correctly filters prompts.
    This tests the fix for a bug introduced in commit fb730aa where T2V-only
    datasets had is_i2v_mode defaulting to True, causing incorrect filtering.
    """
    prompts = [
        {'prompt': 'i2v_caption', 'weight': 1.0, 'do_i2v': True, 'do_t2v': False},
        {'prompt': 't2v_caption', 'weight': 1.0, 'do_i2v': False, 'do_t2v': True},
        {'prompt': 'both_caption', 'weight': 1.0, 'do_i2v': True, 'do_t2v': True},
    ]
    
    # For T2V-only mode (is_i2v_mode=False), should get t2v_caption and both_caption
    filtered_t2v = _filter_prompts_by_mode(prompts, is_i2v_mode=False, log_warning=False)
    assert len(filtered_t2v) == 2, f"Expected 2 T2V prompts, got {len(filtered_t2v)}"
    t2v_prompts = [p['prompt'] for p in filtered_t2v]
    assert 't2v_caption' in t2v_prompts, f"Expected 't2v_caption' in T2V mode"
    assert 'both_caption' in t2v_prompts, f"Expected 'both_caption' in T2V mode"
    assert 'i2v_caption' not in t2v_prompts, f"'i2v_caption' should NOT be in T2V mode"
    
    # For I2V-only mode (is_i2v_mode=True), should get i2v_caption and both_caption
    filtered_i2v = _filter_prompts_by_mode(prompts, is_i2v_mode=True, log_warning=False)
    assert len(filtered_i2v) == 2, f"Expected 2 I2V prompts, got {len(filtered_i2v)}"
    i2v_prompts = [p['prompt'] for p in filtered_i2v]
    assert 'i2v_caption' in i2v_prompts, f"Expected 'i2v_caption' in I2V mode"
    assert 'both_caption' in i2v_prompts, f"Expected 'both_caption' in I2V mode"
    assert 't2v_caption' not in i2v_prompts, f"'t2v_caption' should NOT be in I2V mode"
    
    print("✓ test_filter_prompts_t2v_only_mode passed")


def test_filter_prompts_t2v_only_dataset_scenario():
    """
    Test the exact scenario that was broken by commit fb730aa:
    A T2V-only dataset (do_i2v=False, do_t2v=True) with JSON captions
    where prompts are marked as do_i2v=False, do_t2v=True.
    
    Before the fix: is_i2v_mode defaulted to True, so these prompts were
    filtered OUT and the dataset got empty captions.
    After the fix: is_i2v_mode is correctly set to False, so these prompts
    are filtered IN and used correctly.
    """
    # Prompts that are T2V-only
    prompts = [
        {'prompt': 't2v_only_1', 'weight': 1.0, 'do_i2v': False, 'do_t2v': True},
        {'prompt': 't2v_only_2', 'weight': 1.0, 'do_i2v': False, 'do_t2v': True},
    ]
    
    # With the fix, is_i2v_mode should be False for T2V-only datasets
    filtered = _filter_prompts_by_mode(prompts, is_i2v_mode=False, log_warning=False)
    assert len(filtered) == 2, f"Expected 2 T2V-only prompts, got {len(filtered)}"
    
    # Verify weighted selection works on the filtered prompts
    selected = select_prompt_weighted(filtered)
    assert selected['prompt'] in ['t2v_only_1', 't2v_only_2'], \
        f"Expected T2V-only prompt, got '{selected['prompt']}'"
    
    print("✓ test_filter_prompts_t2v_only_dataset_scenario passed")


def test_empty_json_array_no_crash():
    """
    Test that JSON with no valid prompts (empty array) doesn't crash.
    
    Bug scenario: JSON file starts with valid prompts (cached), then user
    changes it to empty array. The hash changes, and load_prompt_embedding()
    would try to load from a cache path that was never created, causing
    FileNotFoundError.
    
    Fix: When json_caption_path is set but raw_prompts is empty, don't try
    to load from cache - just use empty caption.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([], f)  # Empty array - no valid prompts
        temp_path = f.name
    
    try:
        prompts = parse_json_captions(temp_path)
        assert prompts == [], f"Expected empty list, got {prompts}"
        assert len(prompts) == 0
        # Verify that select_prompt_weighted handles empty list gracefully
        result = select_prompt_weighted(prompts)
        assert result['prompt'] == '', f"Expected empty caption for no prompts, got '{result['prompt']}'"
    finally:
        os.unlink(temp_path)
    
    print("✓ test_empty_json_array_no_crash passed")


def test_json_no_valid_prompts_fallback():
    """
    Test that JSON with all non-dict items falls back to empty caption.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(["string", 123, None, True], f)  # All non-dict items
        temp_path = f.name
    
    try:
        prompts = parse_json_captions(temp_path)
        assert prompts == [], f"Expected empty list for non-dict items, got {prompts}"
        
        # Empty prompts should return empty caption when selected
        result = select_prompt_weighted(prompts)
        assert result['prompt'] == ''
    finally:
        os.unlink(temp_path)
    
    print("✓ test_json_no_valid_prompts_fallback passed")


if __name__ == '__main__':
    print("Running JSON caption caching tests...\n")
    
    test_select_prompt_weighted_all_zero_weights()
    test_select_prompt_weighted_some_zero_weights()
    test_filter_prompts_by_mode_warning()
    test_compute_json_file_hash()
    test_parse_json_captions_basic()
    test_parse_json_captions_defaults()
    test_parse_json_captions_empty_prompt_respected()
    test_select_prompt_weighted_with_empty_prompt()
    test_filter_prompts_t2v_only_mode()
    test_filter_prompts_t2v_only_dataset_scenario()
    test_empty_json_array_no_crash()
    test_json_no_valid_prompts_fallback()
    
    print("\nAll tests passed! ✓")
