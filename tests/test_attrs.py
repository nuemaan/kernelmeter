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
    # id 150 succeeds in the fake but has no name in our table
    assert result["attribute_150"] == 7


def test_cuda12_range_names(fake_driver):
    dev = fake_driver.device(0)
    result = attrs.query_all(fake_driver, dev)
    assert result["numa_id"] == -1
    assert result["gpu_pci_device_id"] == 0x1EB810DE


def test_device_metadata(fake_driver):
    dev = fake_driver.device(0)
    assert dev.name == "NVIDIA GeForce RTX 3090"
    assert dev.total_mem_bytes == 25438126080
    assert fake_driver.driver_version() == (12, 4)
    assert fake_driver.device_count() == 1
