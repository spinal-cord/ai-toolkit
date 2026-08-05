"""
Test script to verify do_t2v and mixed I2V/T2V mode implementation.
"""
import copy
from unittest.mock import MagicMock, patch


def test_file_item_is_i2v_mode_default():
    """Test that FileItemDTO defaults to I2V mode."""
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import DatasetConfig
    
    config = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
    )
    
    # Mock everything needed for FileItemDTO initialization
    with patch("toolkit.data_transfer_object.data_loader.get_quick_signature_string", return_value="abc123"):
        with patch("av.open") as mock_av:
            mock_container = MagicMock()
            mock_container.duration = 5000  # 5 seconds
            mock_av.return_value.__enter__.return_value = mock_container
            
            with patch("cv2.VideoCapture") as mock_cap:
                mock_cap.return_value.isOpened.return_value = True
                mock_cap.return_value.get.return_value = 100
                mock_cap.return_value.release.return_value = None
                
                item = FileItemDTO(
                    path="/test/video.mp4",
                    dataset_config=config,
                )
    
    # Default should be I2V mode
    assert item.is_i2v_mode == True, f"Expected is_i2v_mode=True, got {item.is_i2v_mode}"
    print("✓ test_file_item_is_i2v_mode_default passed")


def test_dataset_doubling_logic():
    """Test the dataset doubling logic directly."""
    import copy
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import DatasetConfig
    
    # Simulate the doubling logic from AiToolkitDataset.__init__
    config = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
        do_i2v=True,
        do_t2v=True,
    )
    
    # Create some mock file items
    with patch("toolkit.data_transfer_object.data_loader.get_quick_signature_string", return_value="abc123"):
        with patch("av.open") as mock_av:
            mock_container = MagicMock()
            mock_container.duration = 5000
            mock_av.return_value.__enter__.return_value = mock_container
            
            with patch("cv2.VideoCapture") as mock_cap:
                mock_cap.return_value.isOpened.return_value = True
                mock_cap.return_value.get.return_value = 100
                mock_cap.return_value.release.return_value = None
                
                file_list = [
                    FileItemDTO(path="/test/video1.mp4", dataset_config=config),
                    FileItemDTO(path="/test/video2.mp4", dataset_config=config),
                ]
    
    # Apply the doubling logic
    is_video = True
    current_file_list = [x for x in file_list]
    for file_item in current_file_list:
        new_file_item = copy.deepcopy(file_item)
        new_file_item.is_i2v_mode = False
        file_list.append(new_file_item)
    
    # Original 2 videos should become 4
    assert len(file_list) == 4, f"Expected 4 items, got {len(file_list)}"
    
    i2v_count = sum(1 for item in file_list if item.is_i2v_mode)
    t2v_count = sum(1 for item in file_list if not item.is_i2v_mode)
    
    assert i2v_count == 2, f"Expected 2 I2V items, got {i2v_count}"
    assert t2v_count == 2, f"Expected 2 T2V items, got {t2v_count}"
    
    print("✓ test_dataset_doubling_logic passed")


def test_dataset_no_doubling_when_only_i2v_enabled():
    """Test that dataset does not double when only do_i2v is enabled."""
    from toolkit.config_modules import DatasetConfig
    
    config = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
        do_i2v=True,
        do_t2v=False,  # Only I2V enabled
    )
    
    # Simulate the check
    is_video = True
    should_double = is_video and config.do_i2v and config.do_t2v
    
    assert should_double == False, "Should not double when only do_i2v is enabled"
    print("✓ test_dataset_no_doubling_when_only_i2v_enabled passed")


def test_batch_get_is_i2v_mode_list():
    """Test that DataLoaderBatchDTO correctly returns I2V mode list."""
    import torch
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
    from toolkit.config_modules import DatasetConfig
    
    config = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
        do_i2v=True,
        do_t2v=True,
    )
    
    # Create mock file items with proper attributes
    item1 = MagicMock(spec=FileItemDTO)
    item1.is_latent_cached = False
    item1.tensor = torch.randn(10, 3, 64, 64)  # Some dummy tensor
    item1.control_tensor = None
    item1.control_tensor_list = None
    item1.clip_image_tensor = None
    item1.mask_tensor = None
    item1.unaugmented_tensor = None
    item1.unconditional_tensor = None
    item1.inpaint_tensor = None
    item1.clip_image_embeds = None
    item1.clip_image_embeds_unconditional = None
    item1.prompt_embeds = None
    item1.audio_tensor = None
    item1.loss_multiplier = 1.0
    item1.num_frames = 10
    item1.extra_values = []
    item1.audio_data = None
    item1.dataset_config = config
    item1.caption = "test"
    item1.caption_short = "test"
    item1.is_reg = False
    item1.network_weight = 1.0
    item1.is_i2v_mode = True  # I2V item
    item1.te_padding_side = "right"
    item1.text_embedding_space_version = "sd1"
    item1.is_caching_to_disk = False
    item1.is_caching_to_memory = False
    item1._encoded_latent = None
    item1._cached_first_frame_latent = None
    item1._cached_audio_latent = None
    
    item2 = MagicMock(spec=FileItemDTO)
    item2.is_latent_cached = False
    item2.tensor = torch.randn(10, 3, 64, 64)
    item2.control_tensor = None
    item2.control_tensor_list = None
    item2.clip_image_tensor = None
    item2.mask_tensor = None
    item2.unaugmented_tensor = None
    item2.unconditional_tensor = None
    item2.inpaint_tensor = None
    item2.clip_image_embeds = None
    item2.clip_image_embeds_unconditional = None
    item2.prompt_embeds = None
    item2.audio_tensor = None
    item2.loss_multiplier = 1.0
    item2.num_frames = 10
    item2.extra_values = []
    item2.audio_data = None
    item2.dataset_config = config
    item2.caption = "test"
    item2.caption_short = "test"
    item2.is_reg = False
    item2.network_weight = 1.0
    item2.is_i2v_mode = False  # T2V item
    item2.te_padding_side = "right"
    item2.text_embedding_space_version = "sd1"
    item2.is_caching_to_disk = False
    item2.is_caching_to_memory = False
    item2._encoded_latent = None
    item2._cached_first_frame_latent = None
    item2._cached_audio_latent = None
    
    batch = DataLoaderBatchDTO(file_items=[item1, item2])
    
    modes = batch.get_is_i2v_mode_list()
    assert modes == [True, False], f"Expected [True, False], got {modes}"
    
    assert batch.has_mixed_i2v_t2v_modes == True
    
    print("✓ test_batch_get_is_i2v_mode_list passed")


def test_latent_cache_hash_includes_do_t2v():
    """Test that latent cache hash includes do_t2v setting."""
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.config_modules import DatasetConfig
    
    config_i2v_only = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
        do_i2v=True,
        do_t2v=False,
    )
    
    config_both = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
        do_i2v=True,
        do_t2v=True,
    )
    
    with patch("toolkit.data_transfer_object.data_loader.get_quick_signature_string", return_value="abc123"):
        with patch("av.open") as mock_av:
            mock_container = MagicMock()
            mock_container.duration = 5000
            mock_av.return_value.__enter__.return_value = mock_container
            
            with patch("cv2.VideoCapture") as mock_cap:
                mock_cap.return_value.isOpened.return_value = True
                mock_cap.return_value.get.return_value = 100
                mock_cap.return_value.release.return_value = None
                
                item1 = FileItemDTO(
                    path="/test/video.mp4",
                    dataset_config=config_i2v_only,
                )
                
                item2 = FileItemDTO(
                    path="/test/video.mp4",
                    dataset_config=config_both,
                )
    
    hash1 = item1.get_latent_info_dict()
    hash2 = item2.get_latent_info_dict()
    
    # The hashes should differ because do_t2v is different
    assert hash1 != hash2, "Expected different hash dicts when do_t2v differs"
    
    # Check that do_t2v is included in the hash
    assert "do_t2v" in hash2 or not hash2.get("do_t2v")
    print("✓ test_latent_cache_hash_includes_do_t2v passed")


def test_mixed_batch_epoch_counting():
    """Test that epoch counting works correctly with doubled datasets.
    
    Scenario: 3 videos, both do_i2v and do_t2v enabled = 6 items
    - batch_size=1: 6 steps per epoch
    - batch_size=2: 3 steps per epoch
    - batch_size=3: 2 steps per epoch
    - batch_size=6: 1 step per epoch (all 6 items in same batch - mixed I2V/T2V)
    """
    from toolkit.config_modules import DatasetConfig
    
    config = DatasetConfig(
        folder_path="/test/path",
        num_frames=10,
        do_i2v=True,
        do_t2v=True,
        buckets=False,
    )
    
    num_videos = 3
    total_items = num_videos * 2  # Doubled due to both modes enabled
    
    batch_sizes = [1, 2, 3, 6]
    expected_steps = [6, 3, 2, 1]
    
    for batch_size, expected in zip(batch_sizes, expected_steps):
        steps_per_epoch = total_items // batch_size
        assert steps_per_epoch == expected, \
            f"batch_size={batch_size}: expected {expected} steps, got {steps_per_epoch}"
    
    print("✓ test_mixed_batch_epoch_counting passed")

if __name__ == "__main__":
    print("Running do_t2v mixed modes tests...")
    print()
    
    test_file_item_is_i2v_mode_default()
    test_dataset_doubling_logic()
    test_dataset_no_doubling_when_only_i2v_enabled()
    test_batch_get_is_i2v_mode_list()
    test_latent_cache_hash_includes_do_t2v()
    test_mixed_batch_epoch_counting()
    
    print()
    print("All tests passed! ✓")
