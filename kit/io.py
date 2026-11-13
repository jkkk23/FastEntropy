def save_ply_ascii_geo(coords, filedir):
    with open(filedir, "w", encoding="utf-8") as f:
        f.writelines(["ply\n", "format ascii 1.0\n"])
        f.write("element vertex " + str(coords.shape[0]) + "\n")
        f.writelines(["property float x\n", "property float y\n", "property float z\n"])
        f.write("end_header\n")
        coords = coords.astype("float32")
        for xyz in coords:
            f.writelines([str(xyz[0]), " ", str(xyz[1]), " ", str(xyz[2]), "\n"])
