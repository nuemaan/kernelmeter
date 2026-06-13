from kernelmeter import attrs


def test_known_attrs_are_labeled(fake_driver):
    dev = fake_driver.device(0)
    result = attrs.query_all(fake_driver, dev)
    assert result["multiprocessor_count"] == 82
    assert result["warp_size"] == 32
    assert result["compute_capability_major"] == 8
    assert result["compute_capability_minor"] == 6


def test_unsupported_ids_are_skipped(fake_driver):
    dev = fake_driver.device(0)
    result = attrs.query_all(fake_driver, dev)
    # id 2 (max_block_dim_x) is absent from the fake -> must not appear
    assert "max_block_dim_x" not in result


def test_unknown_but_supported_ids_get_generic_names(fake_driver):
    dev = fake_driver.device(0)
    result = attrs.query_all(fake_driver, dev)
    # id 160 succeeds in the fake but has no name in our table
    assert result["attribute_160"] == 7


def test_cuda12_range_names(fake_driver):
    dev = fake_driver.device(0)
    result = attrs.query_all(fake_driver, dev)
    assert result["numa_id"] == -1
    assert result["gpu_pci_device_id"] == 0x1EB810DE
    assert result["atomic_reduction_supported"] == 1
    # a CUDA 13.x attribute past the old 0.3.1 table
    assert result["dma_buf_mmap_supported"] == 1


def test_max_sentinel_is_not_named():
    # 155 is CU_DEVICE_ATTRIBUTE_MAX as of CUDA 13.x, not a real attribute
    assert 155 not in attrs.KNOWN_ATTRS
    # the last real attribute we name
    assert attrs.KNOWN_ATTRS[154] == "logical_endpoint_unicast_access_on_owner_device_supported"


def test_device_metadata(fake_driver):
    dev = fake_driver.device(0)
    assert dev.name == "NVIDIA GeForce RTX 3090"
    assert dev.total_mem_bytes == 25438126080
    assert fake_driver.driver_version() == (12, 4)
    assert fake_driver.device_count() == 1
