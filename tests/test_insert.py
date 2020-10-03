import pytest
import numpy as np
from PIL import Image

from h5_object import HDFConnection



@pytest.fixture
def make_hdf_object():
    objects = []

    def _make_hdf_object(filename, **kwargs):
        obj = HDFConnection(filename, **kwargs)
        objects.append(obj)
        return obj

    yield _make_hdf_object

    for obj in objects:
        obj.close()



@pytest.mark.parametrize("node_name, data_dir",
    [
        ("00", "tests/data/00.npy"),
        ("01", "tests/data/01.npy"),
        ("02", "tests/data/02.npy"),
        ("03", "tests/data/03.npy"),
        ("04", "tests/data/04.npy"),
    ]
)
def test_insert_single_dataset(make_hdf_object, node_name, data_dir):
    hdf = make_hdf_object("temp.hdf5", driver="core", backing_store=False)

    arr = np.load(data_dir)
    hdf.insert_node(node_name, data=arr)

    assert np.array_equal(hdf.connection[node_name][...], arr)



@pytest.mark.parametrize("node_name, data_dir",
    [
        ("01", "tests/data/01.npy"),
        ("02", "tests/data/02.npy"),
        ("03", "tests/data/03.npy"),
        ("04", "tests/data/04.npy"),
    ]
)
def test_insert_to_existing_hdf(make_hdf_object, node_name, data_dir):
    hdf = make_hdf_object("temp.hdf5", driver="core", backing_store=False)

    existing_arr = np.load("tests/data/00.npy")
    hdf.insert_node("00", data=existing_arr)

    arr = np.load(data_dir)
    hdf.insert_node(node_name, data=arr)

    assert np.array_equal(hdf.connection["00"][...], existing_arr)
    assert np.array_equal(hdf.connection[node_name][...], arr)


def test_insert_multiple_datasets(make_hdf_object):
    hdf = make_hdf_object("temp.hdf5", driver="core", backing_store=False)

    node_names = ["00", "01", "02"]
    data_paths = ["tests/data/00.npy", "tests/data/01.npy", "tests/data/02.npy"]
    node_to_arr = {n: np.load(d) for n, d in zip(node_names, data_paths)}

    for k, v in node_to_arr.items():
        hdf.insert_node(k, data=v)

    for name, arr in hdf.connection.items():
        assert np.array_equal(arr, node_to_arr[name])




@pytest.mark.parametrize("image_path", ["tests/data/images/original_3_16/3.16原片/20200316-_44A9267.jpg"])
def test_insert_directory(make_hdf_object, image_path):
    hdf = make_hdf_object("temp.hdf5", driver="core", backing_store=False)
    
    hdf.insert_directory("tests/data", "/")

    img_arr = np.array(Image.open(image_path))
    path_in_hdf5 = image_path.split(".jpg")[0].replace("tests/data", "")

    assert np.array_equal(hdf.connection[path_in_hdf5][...], img_arr)

