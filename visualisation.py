import py3Dmol

with open("wildtype_structure_prediction_af2.pdb", "r") as f:
    pdb_data = f.read()

viewer = py3Dmol.view(width=800, height=500)
viewer.addModel(pdb_data, "pdb")
viewer.setStyle({'cartoon': {'color': 'spectrum'}})
viewer.zoomTo()

html_content = viewer._make_html()


with open("protein_view.html", "w") as f:
    f.write(html_content)

print("loaded suscessfully")
import webbrowser

webbrowser.open("protein_view.html")