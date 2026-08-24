from pathlib import Path

from openpyxl import Workbook

from app.codebook import TEXT_ONLY_FORMAT, load_codebook


def test_load_codebook_from_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "codebook.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Код", "Категория", "Подкатегория"])
    sheet.append(["A1", "Оплата", "Уровень зарплаты"])
    sheet.append(["B1", "Команда", "Отношения в коллективе"])
    workbook.save(path)

    entries = load_codebook(path, TEXT_ONLY_FORMAT)

    assert [entry.code for entry in entries] == ["A1", "B1"]
    assert entries[0].description == "Категория: Оплата. Подкатегория: Уровень зарплаты"

