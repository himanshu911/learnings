from pathlib import Path


def test_write_and_read_file(tmp_path: Path):
    # tmp_path is a Path object pointing to a unique temp folder for this test

    file_path = tmp_path / "notes.txt"

    # write to the file
    file_path.write_text("hello pytest")

    # read it back
    content = file_path.read_text()

    assert content == "hello pytest"
