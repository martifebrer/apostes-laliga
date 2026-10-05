import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automatitzacio"))

import config


def test_env_file_is_loaded_without_overriding_existing_vars(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comentari\nTEST_NOVA=valor1\nTEST_EXISTENT="del_fitxer"\n\nlinia_sense_igual\n', encoding="utf-8")
    monkeypatch.delenv("TEST_NOVA", raising=False)
    monkeypatch.setenv("TEST_EXISTENT", "de_l_entorn")

    config._carrega_env(str(env))

    assert os.environ["TEST_NOVA"] == "valor1"
    assert os.environ["TEST_EXISTENT"] == "de_l_entorn"
    monkeypatch.delenv("TEST_NOVA")


def test_missing_env_file_is_ignored(tmp_path):
    config._carrega_env(str(tmp_path / "no_existeix.env"))
