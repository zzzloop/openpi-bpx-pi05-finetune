from transcode_bpx_lerobot_images_to_video import _sample_offsets


def test_sample_offsets_cover_episode_boundaries_without_duplicates():
    assert _sample_offsets(1, 3) == [0]
    assert _sample_offsets(10, 3) == [0, 4, 9]
    assert _sample_offsets(3, 10) == [0, 1, 2]
    assert _sample_offsets(5, 0) == []
