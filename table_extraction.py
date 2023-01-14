# -*- coding: utf-8 -*-
import os

import pandas as pd
from docx.api import Document

path = "TARGET.docx"
outputs_folder = "outputs"
os.makedirs(outputs_folder, exist_ok=True)
document = Document(path)
tables = document.tables
extracted_table_path = os.path.join(outputs_folder, "output.xlsx")
writer = pd.ExcelWriter(extracted_table_path, engine='xlsxwriter')

incomplete_table = {}

c = 0
for i in range(len(tables)):
    table = tables[i]
    data = []
    keys = [" ".join(cell.text.strip().lower().split()) for cell in table.rows[0].cells]

    keys = [key.strip() for key in keys]
    if "" in keys:
        keys.remove("")

    print(keys)

    if not ({'study details', 'interventions', 'outcome measures', 'effect size'} <= set(keys) and len(keys) == len(
            set(keys))):
        incomplete_table = {}
        continue

    table_dict = {key: [] for key in keys}
    for j, key in enumerate(keys):
        text_list = [""]
        for k, row in enumerate(table.rows):
            if k == 0:
                continue
            text = row.cells[j].text
            if text != text_list[-1]:
                text_list.append(text)
        if len(text_list) > 1:
            table_dict[key] = "\n".join(text_list[1:])
        else:
            table_dict[key] = ""

    complete = False
    if "participants" in keys:
        if table_dict["participants"].startswith("People group:") and incomplete_table != {} or i == len(tables) - 1:
            complete = True
    elif "patients" in keys:
        print(keys)
        if table_dict["patients"].startswith("Patient group:") and incomplete_table != {} or i == len(tables) - 1:
            complete = True
    if complete:
        complete_table = {}
        for key in incomplete_table:
            complete_table[key] = "\n\n".join(incomplete_table[key])
        if not complete_table["participants"].startswith("People group:"):
            incomplete_table = {}
            continue
        df = pd.DataFrame(complete_table, index=[0])
        df.to_excel(writer, sheet_name=f"Table{c}", index=False)
        c += 1
        incomplete_table = {}
    if incomplete_table == {}:
        for key in keys:
            incomplete_table[key] = [table_dict[key].strip()]
    else:
        for key in keys:
            incomplete_table[key].append(table_dict[key].strip())

print("Total number of tables extracted: ", c)

writer.save()
